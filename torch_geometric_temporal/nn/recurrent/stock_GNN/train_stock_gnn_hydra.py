#!/usr/bin/env python3
"""
Hydra-based training script for Stock GNN pipeline with modular architecture support

This script supports both integrated Lightning models and modular standalone core models:
- Integrated: All model logic contained within Lightning module (original approach)
- Modular: Standalone core model wrapped by Lightning trainer (new approach)

Usage:
    # Default integrated approach
    python train_stock_gnn_hydra.py
    
    # Using modular standalone core
    python train_stock_gnn_hydra.py model.use_standalone_core=true
    
    # Other examples
    python train_stock_gnn_hydra.py data=debug trainer=cpu_debug
    python train_stock_gnn_hydra.py model.lr=5e-4 trainer.max_epochs=100
    python train_stock_gnn_hydra.py model.pure_gru=true model.use_standalone_core=true
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from torch_geometric_temporal.nn.recurrent.stock_GNN.dataset.stock_dataset import StockDataModule
from torch_geometric_temporal.nn.recurrent.stock_GNN.adaptive_adj import DynamicGraphLightning
from torch_geometric_temporal.nn.recurrent.stock_GNN.dynamic_graph_core import DynamicGraphCore
from torch_geometric_temporal.nn.recurrent.stock_GNN.adp_adj_loss import AccumulativeGainLoss
from torch_geometric_temporal.nn.recurrent.stock_GNN.loss.return_loss import ReturnLoss

# Optimize RTX 4090 Tensor Core performance
torch.set_float32_matmul_precision('medium')


def print_config(config: OmegaConf):
    for cfg_name, cfg_object in config.items():  # Use .items() to iterate over key-value pairs
        print(f"Using {cfg_name}: {cfg_object}")


def setup_callbacks(cfg: DictConfig) -> list:
    """Setup PyTorch Lightning callbacks from config"""
    callbacks = []
    
    if "callbacks" in cfg and cfg.callbacks is not None:
        for callback_name, callback_cfg in cfg.callbacks.items():
            if callback_cfg is not None and "_target_" in callback_cfg:
                try:
                    callback = instantiate(callback_cfg)
                    callbacks.append(callback)
                    print(f"✓ Added callback: {callback_name}")
                except Exception as e:
                    print(f"⚠ Failed to instantiate callback {callback_name}: {e}")
    
    return callbacks


def setup_logger(cfg: DictConfig) -> Optional[pl.loggers.Logger]:
    """Setup PyTorch Lightning logger from config"""
    if "logger" not in cfg or cfg.logger is None:
        return None
    
    try:
        logger = instantiate(cfg.logger)
        print(f"✓ Using logger: {cfg.logger._target_}")
        return logger
    except Exception as e:
        print(f"⚠ Failed to instantiate logger: {e}")
        # Fallback to CSV logger
        try:
            from pytorch_lightning.loggers import CSVLogger
            logger = CSVLogger("logs", name="stock_gnn")
            print("📋 Fallback to CSV Logger")
            return logger
        except Exception:
            print("❌ No logger available")
            return None


def create_data_module(cfg: DictConfig) -> StockDataModule:
    """Create the data module from config using Hydra instantiate"""
    try:
        # Use Hydra's instantiate to create the data module
        dm = instantiate(cfg.data)
        
        print_config(cfg.data)
        
        return dm
        
    except Exception as e:
        print(f"❌ Failed to create data module: {e}")
        raise



def create_model(cfg: DictConfig, node_feat_dim: int) -> nn.Module:
    """Create the model from config using Hydra instantiate"""
    try:

        # Create core model first
        core_model_cfg = OmegaConf.structured({
            **cfg.model
        })
        
        core_model = instantiate(core_model_cfg)
    
        
        print(f"   ✓ Created standalone core model: {core_model.__class__.__name__}")

        print(f"🧠 Created model:")
        print_config(core_model_cfg)
        return core_model
        
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        raise

def create_loss(cfg: DictConfig) -> nn.Module:
    try:
        loss_cfg = cfg.loss

        loss_fn = instantiate(loss_cfg)

        print_config(loss_cfg)

        return loss_fn
    
    except Exception as e:
        print(f"❌ Failed to create loss function {e}")
        raise
        # model: nn.Module,  # Pass the model instance as an argument
        # lr: float = 1e-3,
        # loss_fn: nn.Module = None,
        # metric_compute_frequency: int = 10,
        # weight_decay: float = 1e-4,
        # scheduler_config: dict = None,
def create_training_module(cfg: DictConfig, model: nn.Module, loss_fn: nn.Module) -> pl.LightningModule:

    try:
        training_module_cfg = {
            "_target_": cfg.training_module._target_,  # Ensure the _target_ key is included
            "model": model,  # Pass the model object directly
            "loss_fn": loss_fn,  # Pass the loss function object directly
            **cfg.training_module  # Unpack the rest of the training module configuration
        }

        training_module = instantiate(training_module_cfg)

        print_config(training_module_cfg)

        return training_module

    except Exception as e:
        print(f"❌ Failed to create training_module: {e}")
        raise
    

def create_trainer(cfg: DictConfig, callbacks: list, logger) -> pl.Trainer:
    """Create PyTorch Lightning trainer from config using Hydra instantiate"""
    trainer_cfg = cfg.trainer
    
    # Handle devices configuration for DDP
    devices = trainer_cfg.devices
    strategy = trainer_cfg.strategy
    
    # For pure_gru or lightweight_gru mode, we need to handle unused parameters in DDP
    if (cfg.model.get("pure_gru", False) or cfg.model.get("lightweight_gru", False)) and strategy == "ddp":
        mode_name = "Pure GRU" if cfg.model.get("pure_gru", False) else "Lightweight GRU"
        print(f"🔧 {mode_name} mode detected with DDP - enabling find_unused_parameters")
        strategy = "ddp_find_unused_parameters_true"
    
    if isinstance(devices, int) and devices > 1 and "ddp" in strategy:
        print(f"🔥 Using DDP with {devices} GPUs, strategy: {strategy}")
    
        # Create trainer config for instantiation
    trainer_config = {
        "devices": devices,
        "strategy": strategy,
        "callbacks": callbacks,  # Pass the list of callback objects directly
        "logger": logger,  # Pass the logger object directly
        **trainer_cfg,  # Unpack the rest of the trainer configuration
    }
    
    try:
        trainer = instantiate(trainer_config)
        
        print(f"⚡ Created trainer:")
        print(f"   - Max epochs: {trainer_cfg.max_epochs}")
        print(f"   - Accelerator: {trainer_cfg.accelerator}")
        print(f"   - Devices: {devices}")
        print(f"   - Strategy: {strategy}")
        print(f"   - Precision: {trainer_cfg.get('precision', '32-true')}")
        
        return trainer
        
    except Exception as e:
        print(f"❌ Failed to create trainer: {e}")
        raise


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function"""
    
    # Print configuration
    print("=" * 80)
    print("🚀 Stock GNN Training Pipeline with Hydra (Modular Architecture)")
    print("=" * 80)
    
    print("📋 Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Set random seed for reproducibility
    if "seed" in cfg:
        seed_everything(cfg.seed, workers=True)
        print(f"🌱 Set random seed: {cfg.seed}")
    
    # Create output directories
    os.makedirs(cfg.get("output_dir", "outputs"), exist_ok=True)
    os.makedirs(cfg.get("log_dir", "logs"), exist_ok=True)
    
    try:
        # 1. Setup data module
        print("\n📊 Setting up data module...")
        dm = create_data_module(cfg)
        dm.prepare_data()
        dm.setup("fit")
        
        # Get feature dimension
        node_feat_dim = dm.get_feature_dim()
        stock_num = dm.get_stock_num()
        prediction_horizons = dm.get_prediction_horizons()
        
        print(f"   ✓ Feature dimension: {node_feat_dim}")
        print(f"   ✓ Number of stocks: {stock_num}")
        print(f"   ✓ Prediction horizons: {prediction_horizons}")
        
        # 2. Create model (with architecture choice)
        # print(f"\n🧠 Creating model ({architecture_mode})...")
        model = create_model(cfg, node_feat_dim)
        loss_fn = create_loss(cfg)
        training_module = create_training_module(cfg, model, loss_fn)
        # 3. Setup callbacks and logger
        print("\n⚙️ Setting up callbacks and logger...")
        callbacks = setup_callbacks(cfg)
        logger = setup_logger(cfg)
        
        # 4. Create trainer
        print("\n⚡ Creating trainer...")
        trainer = create_trainer(cfg, callbacks, logger)
        
        # 5. Start training
        print("\n🎯 Starting training...")
        print("-" * 60)
        trainer.fit(training_module, datamodule=dm)
        
        # 6. Test the model
        print("\n🧪 Testing model...")
        print("-" * 60)
        trainer.test(training_module, datamodule=dm)
        
        print("\n✅ Training completed successfully!")
        # print(f"🏗️  Final Architecture: {architecture_mode}")
        
        # Save final configuration
        config_path = Path(trainer.logger.log_dir) / "config.yaml"
        with open(config_path, "w") as f:
            OmegaConf.save(cfg, f)
        print(f"💾 Saved configuration to: {config_path}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise e


if __name__ == "__main__":
    main()
