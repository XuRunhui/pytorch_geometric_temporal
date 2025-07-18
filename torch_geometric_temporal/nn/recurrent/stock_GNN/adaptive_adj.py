# models/dyn_graph_module.py
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch import nn
from torch_geometric.nn import GCNConv, GATConv  # Add GATConv import
from torch_geometric.data import Batch
from torch_geometric_temporal.nn.recurrent.stock_GNN.adp_adj_loss import AccumulativeGainLoss  # 假设你有这个自定义损失函数
from torch_geometric_temporal.nn.recurrent.stock_GNN.dynamic_graph_core import DynamicGraphCore
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops

class DynamicGraphLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for DynamicGraphCore model.
    
    This class handles training, validation, and testing logic while delegating
    the forward pass to the standalone DynamicGraphCore model.
    """
    
    def __init__(
        self,
        node_feat_dim: int,
        gru_hidden_dim: int = 64,
        gnn_hidden_dim: int = 64,
        k_nn: int = 8,
        lr: float = 1e-3,
        loss_fn: nn.Module = None,
        add_self_loops: bool = True,
        metric_compute_frequency: int = 10,
        weight_decay: float = 1e-4,
        scheduler_config: dict = None,
        gnn_type: str = "gcn",
        gat_heads: int = 4,
        gat_dropout: float = 0.1,
        predict_return: bool = False,
        output_factor_dim: int = 32,
        pure_gru: bool = False,
    ):
        super().__init__()
        
        # Store training configuration
        self.lr = lr
        self.loss_fn = loss_fn or AccumulativeGainLoss(value_decay=0.9, penalty_weight=0.1, eps=1e-8)
        self.metric_compute_frequency = metric_compute_frequency
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config or {}
        
        # Initialize the core model
        self.core_model = DynamicGraphCore(
            node_feat_dim=node_feat_dim,
            gru_hidden_dim=gru_hidden_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            k_nn=k_nn,
            add_self_loops=add_self_loops,
            gnn_type=gnn_type,
            gat_heads=gat_heads,
            gat_dropout=gat_dropout,
            predict_return=predict_return,
            output_factor_dim=output_factor_dim,
            pure_gru=pure_gru,
        )
        
        # Store model configuration for easy access
        self.model_config = self.core_model.get_model_stats()
        
        # Automatically save hyperparameters
        self.save_hyperparameters()
        
        print(f"⚡ Lightning Wrapper for {self.model_config['gnn_type'].upper()} model initialized")
    
    def forward(self, data_input):
        """Delegate forward pass to core model"""
        return self.core_model(data_input)
    
    def _should_compute_metrics(self) -> bool:
        """Determine if metrics should be computed based on current epoch and frequency."""
        current_epoch = self.current_epoch
        return (current_epoch % self.metric_compute_frequency == 0) or (current_epoch == 0)
    
    def _compute_loss_and_log_stats(self, batch, stage: str = "train"):
        """Common loss computation and logging logic"""
        x_t, y_t = batch
        
        # Forward pass through core model
        if self.model_config['predict_return']:
            inter_feature, y_pred = self(x_t)
        else:
            y_pred = self(x_t)
        
        # Determine if we should compute metrics
        compute_metrics = (stage != "train") or self._should_compute_metrics()
        
        # Log some forward pass statistics
        if isinstance(y_pred, torch.Tensor):
            self.log(f'stats/{stage}_pred_mean', y_pred.mean().item(), on_step=True, on_epoch=False)
            self.log(f'stats/{stage}_pred_std', y_pred.std().item(), on_step=True, on_epoch=False)
        
        # Handle different output formats
        if isinstance(y_pred, list):
            # Variable length graphs
            total_loss = 0
            total_nodes = 0
            for i, (pred, target) in enumerate(zip(y_pred, y_t)):
                if isinstance(target, torch.Tensor):
                    loss_i = self.loss_fn(pred, target.float(), compute_metrics=compute_metrics)
                    total_loss += loss_i * pred.size(0)
                    total_nodes += pred.size(0)
            loss = total_loss / total_nodes if total_nodes > 0 else total_loss
        else:
            # Fixed size graphs
            if self.model_config['predict_return']:
                loss = self.loss_fn(inter_feature, y_pred, y_t.float(), compute_metrics=compute_metrics)
            else:
                loss = self.loss_fn(y_pred, y_t.float(), compute_metrics=compute_metrics)
            
            # Log metrics if computed
            if compute_metrics and hasattr(loss, 'rank_ic_info'):
                rank_ic_info = loss.rank_ic_info
                for metric_name, metric_value in rank_ic_info.items():
                    if not np.isnan(metric_value):
                        self.log(f'{stage}_{metric_name}', metric_value, 
                                on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                        
                        # Print test metrics for final evaluation
                        if stage == "test":
                            print(f"Final test {metric_name}: {metric_value:.6f}")
        
        # Log loss
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stage}_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        """Training step using core model"""
        return self._compute_loss_and_log_stats(batch, stage="train")
    
    def validation_step(self, batch, batch_idx):
        """Validation step using core model"""
        return self._compute_loss_and_log_stats(batch, stage="val")
    
    def test_step(self, batch, batch_idx):
        """Test step using core model"""
        return self._compute_loss_and_log_stats(batch, stage="test")
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        scheduler_type = self.scheduler_config.get('type', None)
        
        if scheduler_type is None:
            return optimizer
        
        elif scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_config.get('factor', 0.5),
                patience=self.scheduler_config.get('patience', 5),
                min_lr=self.scheduler_config.get('min_lr', 1e-6),
                verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        
        elif scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.get('T_max', 100),
                eta_min=self.scheduler_config.get('min_lr', 1e-6)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_config.get('step_size', 30),
                gamma=self.scheduler_config.get('gamma', 0.1)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        
        elif scheduler_type == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.scheduler_config.get('gamma', 0.95)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        
        else:
            print(f"Warning: Unknown scheduler type '{scheduler_type}', using no scheduler")
            return optimizer

