# dynamic_graph_core.py
"""
Core PyTorch model for Dynamic Graph Neural Networks
This module contains the standalone model logic separated from the Lightning trainer
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops
from typing import List, Union, Tuple, Optional


class DynamicGraphCore(nn.Module):
    """
    Core Dynamic Graph Neural Network model that can be used standalone or with any trainer.
    
    This model supports multiple architectures:
    - Pure GRU: Only temporal modeling without graph structure
    - GRU + GCN: Temporal + spatial modeling with Graph Convolutional Networks
    - GRU + GAT: Temporal + spatial modeling with Graph Attention Networks
    """
    
    def __init__(
        self,
        node_feat_dim: int,
        gru_hidden_dim: int = 64,
        gnn_hidden_dim: int = 64,
        k_nn: int = 8,
        add_self_loops: bool = True,
        gnn_type: str = "gcn",
        gat_heads: int = 4,
        gat_dropout: float = 0.1,
        predict_return: bool = False,
        output_factor_dim: int = 32,
        pure_gru: bool = False,
    ):
        super().__init__()
        
        # Store configuration
        self.node_feat_dim = node_feat_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.k_nn = k_nn
        self.add_self_loops = add_self_loops
        self.gnn_type = gnn_type.lower()
        self.gat_heads = gat_heads
        self.gat_dropout = gat_dropout
        self.predict_return = predict_return
        self.output_factor_dim = output_factor_dim
        self.pure_gru = pure_gru
        
        # Build model components
        self._build_model()
        
        # Log model configuration
        print(f"🏗️  Core Model Architecture: {'Pure GRU' if self.pure_gru else f'GRU + {self.gnn_type.upper()}'}")
        if self.gnn_type == "gat" and not self.pure_gru:
            print(f"   - GAT heads: {self.gat_heads}")
            print(f"   - GAT dropout: {self.gat_dropout}")
            print(f"   - Final GNN dim: {self.final_gnn_dim}")
    
    def _build_model(self):
        """Build the model architecture"""
        # 1. Temporal encoding with GRU
        self.gru = nn.GRU(
            input_size=self.node_feat_dim,
            hidden_size=self.gru_hidden_dim,
            batch_first=True,
        )
        
        # Additional GRU for similarity computation (dual GRU architecture)
        self.gru_sim = nn.GRU(
            input_size=self.node_feat_dim,
            hidden_size=self.gru_hidden_dim,
            batch_first=True,
        )
        
        # 2. Spatial modeling components
        if not self.pure_gru:
            if self.gnn_type == "gat":
                # GAT layers with multi-head attention
                self.gnn1 = GATConv(
                    self.gru_hidden_dim,
                    self.gnn_hidden_dim // self.gat_heads,
                    heads=self.gat_heads,
                    dropout=self.gat_dropout,
                    add_self_loops=self.add_self_loops,
                    concat=True
                )
                self.gnn2 = GATConv(
                    self.gnn_hidden_dim,
                    self.gnn_hidden_dim,
                    heads=1,
                    dropout=self.gat_dropout,
                    add_self_loops=self.add_self_loops,
                    concat=False
                )
                self.final_gnn_dim = self.gnn_hidden_dim
            else:
                # Default GCN layers
                self.gnn1 = GCNConv(self.gru_hidden_dim, self.gnn_hidden_dim, add_self_loops=self.add_self_loops)
                self.gnn2 = GCNConv(self.gnn_hidden_dim, self.gnn_hidden_dim, add_self_loops=self.add_self_loops)
                self.final_gnn_dim = self.gnn_hidden_dim
        else:
            # Pure GRU mode: final dimension is GRU hidden dimension
            self.final_gnn_dim = self.gru_hidden_dim
        
        # 3. Output layers
        self.batch_norm = nn.BatchNorm1d(self.final_gnn_dim)
        self.predictor = nn.Linear(self.final_gnn_dim, self.output_factor_dim)
        
        # Optional return prediction layers
        if self.predict_return:
            print("🎯 Building return prediction layers")
            self.return_predictor = nn.ModuleList([
                nn.ReLU(),
                nn.Linear(self.output_factor_dim, 1)
            ])
    
    def forward(self, data_input) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass supporting multiple input formats
        
        Args:
            data_input: Can be one of:
                1. torch.Tensor: [batch_size, seq_len, feat_dim, num_nodes] - fixed size graphs
                2. list[torch.Tensor]: Variable size graphs
        
        Returns:
            Predictions in matching format to input
        """
        if self.pure_gru:
            return self._forward_gru(data_input)
        
        if isinstance(data_input, torch.Tensor):
            return self._forward_batch_tensor(data_input)
        elif isinstance(data_input, list):
            return self._forward_graph_list(data_input)
        else:
            raise ValueError("Input must be either torch.Tensor or list of torch.Tensor")
    
    def _forward_gru(self, x_seq: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Pure GRU forward pass without graph structure"""
        # Handle input shape
        if x_seq.dim() == 3:
            b, n, f = x_seq.shape
            l = 1
            x_seq = x_seq.unsqueeze(1)  # [batch_size, 1, feat_dim, num_nodes]
        else:
            b, l, f, n = x_seq.shape
        
        # Reshape for GRU: [seq_len, batch_size * num_nodes, feat_dim]
        gru_in = x_seq.permute(1, 0, 3, 2).reshape(l, b * n, f)
        gru_out, _ = self.gru(gru_in)
        
        # Get last timestep output
        h = gru_out[-1].view(b, n, -1)  # [batch_size, num_nodes, gru_hidden_dim]
        
        # Apply BatchNorm and prediction
        h_flat = h.view(b * n, -1)
        out2_normalized = self.batch_norm(h_flat)
        out = self.predictor(out2_normalized)
        
        # Optional return prediction
        return_output = out
        if self.predict_return:
            return_output = self.forward_return(out)
        
        final_out = return_output.view(b, n, -1)
        
        if self.predict_return:
            return out.view(b, n, -1), final_out
        return final_out
    
    def _forward_batch_tensor(self, x_seq: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Full GNN forward pass for batch tensor input"""
        # Handle input shape
        if x_seq.dim() == 3:
            b, n, f = x_seq.shape
            l = 1
            x_seq = x_seq.unsqueeze(1)
        else:
            b, l, f, n = x_seq.shape
        
        # Dual GRU encoding
        gru_in = x_seq.permute(1, 0, 3, 2).reshape(l, b * n, f)
        
        # Similarity GRU for graph construction
        gru_out_sim, _ = self.gru_sim(gru_in)
        h = gru_out_sim[-1].view(b, n, -1)
        
        # Feature GRU for node representations
        gru_out, _ = self.gru(gru_in)
        h_out = gru_out[-1].view(b, n, -1)
        
        # Dynamic graph construction
        sim = torch.einsum("bni,bmi->bnm", h, h)  # [b, n, n]
        
        # Top-k edge selection
        sim_masked = sim.clone()
        eye_mask = torch.eye(n, device=sim.device).bool().unsqueeze(0).expand(b, -1, -1)
        sim_masked[eye_mask] = -1e9
        
        topk_vals, topk_idx = sim_masked.topk(self.k_nn, dim=-1, sorted=True)
        
        # Create edge indices
        edge_src = torch.arange(n, device=sim.device).unsqueeze(0).unsqueeze(2).expand(b, n, self.k_nn).contiguous()
        edge_dst = topk_idx.contiguous()
        edge_weight = topk_vals.contiguous()
        
        # Create batch data for GNN
        data_list = []
        for i in range(b):
            src_flat = edge_src[i].contiguous().view(-1)
            dst_flat = edge_dst[i].contiguous().view(-1)
            weight_flat = edge_weight[i].contiguous().view(-1)
            
            data = Data(
                x=h_out[i],
                edge_index=torch.stack([src_flat, dst_flat], dim=0),
                edge_weight=weight_flat
            )
            data_list.append(data)
        
        batch_data = Batch.from_data_list(data_list)
        x_all = batch_data.x
        e_idx = batch_data.edge_index
        e_w = batch_data.edge_weight
        
        # Add self loops for GCN
        if self.gnn_type == "gcn" and self.add_self_loops:
            e_idx, e_w = add_self_loops(
                batch_data.edge_index,
                batch_data.edge_weight,
                fill_value=1.0,
                num_nodes=batch_data.num_nodes
            )
        
        # GNN message passing
        if self.gnn_type == "gat":
            out1 = F.relu(self.gnn1(x_all, e_idx))
            out2 = F.relu(self.gnn2(out1, e_idx))
        else:
            out1 = F.relu(self.gnn1(x_all, e_idx, edge_weight=e_w))
            out2 = F.relu(self.gnn2(out1, e_idx, edge_weight=e_w))
        
        # Apply BatchNorm and prediction
        out2_normalized = self.batch_norm(out2)
        out = self.predictor(out2_normalized)
        
        # Optional return prediction
        return_output = out
        if self.predict_return:
            return_output = self.forward_return(out)
        
        final_out = return_output.view(b, n, -1)
        
        if self.predict_return:
            return out.view(b, n, -1), final_out
        return final_out
    
    def _forward_graph_list(self, graph_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass for variable size graphs"""
        results = []
        
        for i, graph_seq in enumerate(graph_list):
            # Handle graph sequence shape
            if graph_seq.dim() == 2:
                f, n = graph_seq.shape
                l = 1
                graph_seq = graph_seq.unsqueeze(0)
            else:
                l, f, n = graph_seq.shape
            
            # GRU encoding
            gru_in = graph_seq.permute(0, 2, 1)  # [seq_len, num_nodes, feat_dim]
            gru_out, _ = self.gru(gru_in)
            h = gru_out[-1]  # [num_nodes, gru_hidden_dim]
            
            # Dynamic graph construction for individual graph
            sim = torch.einsum("ni,mi->nm", h, h)
            
            # Ensure k_nn doesn't exceed available nodes
            k = min(self.k_nn, n - 1)
            if k <= 0:
                # Handle small graphs directly
                h_normalized = self.batch_norm(h)
                out = self.predictor(h_normalized)
                results.append(out)
                continue
            
            # Top-k selection and edge construction
            topk_vals, topk_idx = sim.topk(k + 1, dim=-1)
            
            # Remove self-loops
            node_idx = torch.arange(n, device=sim.device)[:, None]
            mask = topk_idx != node_idx
            
            # Build edges
            edge_weights = []
            edge_sources = []
            edge_targets = []
            
            for node in range(n):
                valid_neighbors = topk_idx[node][mask[node]][:k]
                valid_weights = topk_vals[node][mask[node]][:k]
                
                edge_sources.extend([node] * len(valid_neighbors))
                edge_targets.extend(valid_neighbors.tolist())
                edge_weights.extend(valid_weights.tolist())
            
            if len(edge_sources) == 0:
                # No edges, use linear layer directly
                h_normalized = self.batch_norm(h)
                out = self.predictor(h_normalized)
                results.append(out)
                continue
            
            # Create graph data
            edge_index = torch.tensor([edge_sources, edge_targets], 
                                    dtype=torch.long, device=h.device)
            edge_weight = torch.tensor(edge_weights, dtype=torch.float, device=h.device)
            
            # Add self loops for GCN
            if self.gnn_type == "gcn" and self.add_self_loops:
                edge_index, edge_weight = add_self_loops(
                    edge_index, edge_weight,
                    fill_value=1.0, num_nodes=n
                )
            
            # GNN message passing
            if self.gnn_type == "gat":
                out1 = F.relu(self.gnn1(h, edge_index))
                out2 = F.relu(self.gnn2(out1, edge_index))
            else:
                out1 = F.relu(self.gnn1(h, edge_index, edge_weight=edge_weight))
                out2 = F.relu(self.gnn2(out1, edge_index, edge_weight=edge_weight))
            
            # Apply BatchNorm and prediction
            out2_normalized = self.batch_norm(out2)
            out = self.predictor(out2_normalized)
            
            results.append(out)
        
        return results
    
    def forward_return(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through return prediction layers"""
        for layer in self.return_predictor:
            x = layer(x)
        return x
    
    def get_model_stats(self) -> dict:
        """Get model statistics for logging"""
        stats = {
            'node_feat_dim': self.node_feat_dim,
            'gru_hidden_dim': self.gru_hidden_dim,
            'output_factor_dim': self.output_factor_dim,
            'pure_gru': self.pure_gru,
            'gnn_type': self.gnn_type if not self.pure_gru else 'none',
            'k_nn': self.k_nn if not self.pure_gru else 0,
            'predict_return': self.predict_return,
        }
        
        if not self.pure_gru:
            stats.update({
                'gnn_hidden_dim': self.gnn_hidden_dim,
                'final_gnn_dim': self.final_gnn_dim,
            })
            
            if self.gnn_type == "gat":
                stats.update({
                    'gat_heads': self.gat_heads,
                    'gat_dropout': self.gat_dropout,
                })
        
        return stats
