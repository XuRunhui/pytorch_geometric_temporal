#!/usr/bin/env python3
"""
Prediction script for Stock GNN pipeline

This script loads a trained checkpoint and predicts factors on the test dataset,
saving results with date and stock code metadata to a file.

Usage:
    python predict.py --checkpoint path/to/checkpoint.ckpt --config path/to/config.yaml --output predictions.csv
    python predict.py --checkpoint logs/stock_gnn/version_0/checkpoints/epoch=49-step=1000.ckpt --output test_predictions.csv
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from torch_geometric_temporal.nn.recurrent.stock_GNN.dataset.stock_dataset import StockDataModule
from torch_geometric_temporal.nn.recurrent.stock_GNN.adaptive_adj import DynamicGraphLightning
from torch_geometric_temporal.nn.recurrent.stock_GNN.training_module import DynamicGraphLightning as TrainingModule


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file"""
    if config_path and os.path.exists(config_path):
        return OmegaConf.load(config_path)
    else:
        # Default configuration for prediction
        return OmegaConf.create({
            "data": {
                "_target_": "torch_geometric_temporal.nn.recurrent.stock_GNN.dataset.stock_dataset.StockDataModule",
                "data_dir": "/home/xu/clean_data_unaligned",
                "use_factors": True,
                "sequence_length": 30,
                "prediction_horizons": [1],
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "batch_size": 1,
                "num_workers": 4,
                "normalize_features": True,
                "normalize_targets": True,
                "debug": False
            }
        })


def load_model_from_checkpoint(checkpoint_path: str) -> pl.LightningModule:
    """Load the trained model from checkpoint"""
    try:
        # Try to load the training module (which contains the model)
        model = TrainingModule.load_from_checkpoint(checkpoint_path)
        model.eval()
        print(f"✓ Loaded model from checkpoint: {checkpoint_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model from checkpoint: {e}")
        raise


def predict_with_metadata(model: pl.LightningModule, 
                         datamodule: StockDataModule, 
                         device: torch.device) -> List[Dict[str, Any]]:
    """
    Run predictions on test dataset and collect results with metadata
    
    Returns:
        List of dictionaries containing predictions with metadata
    """
    model.to(device)
    model.eval()
    
    # Create prediction dataset with metadata
    prediction_dataset = datamodule.create_prediction_dataset(return_metadata=True)
    
    # Create dataloader for prediction
    prediction_loader = torch.utils.data.DataLoader(
        prediction_dataset,
        batch_size=datamodule.batch_size,
        shuffle=False,
        num_workers=datamodule.num_workers,
        pin_memory=True
    )
    
    print(f"📊 Prediction dataset info:")
    print(f"   - Number of batches: {len(prediction_loader)}")
    print(f"   - Number of samples: {len(prediction_dataset)}")
    if datamodule.date_index:
        val_end = int(len(datamodule.date_index) * (datamodule.train_ratio + datamodule.val_ratio))
        test_dates = datamodule.date_index[val_end:]
        if test_dates:
            print(f"   - Date range: {test_dates[0]} to {test_dates[-1]}")
    print(f"   - Number of stocks: {len(datamodule.get_stock_names())}")
    
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(prediction_loader):
            # Handle batch with metadata
            if len(batch) == 3:
                features, targets, metadata_list = batch
            elif len(batch) == 2:
                features, targets = batch
                metadata_list = None
            else:
                print(f"⚠️  Unexpected batch format: {type(batch)}, length: {len(batch)}")
                continue
            
            # Move to device
            features = features.to(device)
            if targets is not None:
                targets = targets.to(device)
            
            # Get predictions
            try:
                predictions = model(features)  # Shape: [batch_size, num_stocks, output_dim]
            except Exception as e:
                print(f"❌ Prediction failed for batch {batch_idx}: {e}")
                continue
            
            # Convert to numpy for processing
            predictions_np = predictions.cpu().numpy()
            if targets is not None:
                targets_np = targets.cpu().numpy()
            else:
                targets_np = None
            
            # Process each sample in the batch
            batch_size = predictions_np.shape[0]
            
            for sample_idx in range(batch_size):
                # Get metadata for this sample
                sample_metadata = None
                if metadata_list:
                    try:
                        if isinstance(metadata_list, dict):
                            # Extract sample-specific metadata
                            sample_metadata = {}
                            for key, value in metadata_list.items():
                                if isinstance(value, (list, tuple)) and len(value) > sample_idx:
                                    sample_metadata[key] = value[sample_idx]
                                elif not isinstance(value, (list, tuple)):
                                    sample_metadata[key] = value
                        else:
                            sample_metadata = metadata_list[sample_idx] if sample_idx < len(metadata_list) else None
                    except Exception as e:
                        print(f"⚠️  Failed to extract metadata for sample {sample_idx}: {e}")
                
                # Extract predictions for this sample
                sample_predictions = predictions_np[sample_idx]  # Shape: [num_stocks, output_dim]
                
                # Extract targets for this sample
                sample_targets = None
                if targets_np is not None:
                    sample_targets = targets_np[sample_idx]  # Shape: [num_stocks] or [num_stocks, output_dim]
                
                # Get valid stock information
                valid_stock_names = []
                if sample_metadata and "valid_stock_names" in sample_metadata:
                    valid_stock_names = sample_metadata["valid_stock_names"]
                elif sample_metadata and "all_stock_names" in sample_metadata:
                    # Use all stock names if valid names not available
                    valid_stock_names = sample_metadata["all_stock_names"]
                else:
                    # Fallback: use datamodule stock names
                    valid_stock_names = datamodule.get_stock_names()
                
                # Process each stock
                num_stocks = sample_predictions.shape[0]
                for stock_idx in range(num_stocks):
                    stock_code = valid_stock_names[stock_idx] if stock_idx < len(valid_stock_names) else f"stock_{stock_idx}"
                    
                    # Extract predictions for this stock
                    stock_predictions = sample_predictions[stock_idx]  # Shape: [output_dim]
                    
                    # Extract targets for this stock
                    stock_targets = None
                    if sample_targets is not None:
                        if sample_targets.ndim == 1:  # [num_stocks]
                            stock_targets = sample_targets[stock_idx] if stock_idx < len(sample_targets) else None
                        elif sample_targets.ndim == 2:  # [num_stocks, output_dim]
                            stock_targets = sample_targets[stock_idx] if stock_idx < sample_targets.shape[0] else None
                    
                    # Create result entry
                    result_entry = {
                        "batch_idx": batch_idx,
                        "sample_idx": sample_idx,
                        "stock_idx": stock_idx,
                        "stock_code": stock_code,
                    }
                    
                    # Add date information
                    if sample_metadata:
                        if "current_date" in sample_metadata:
                            result_entry["date"] = str(sample_metadata["current_date"])
                        if "sequence_start_date" in sample_metadata:
                            result_entry["sequence_start_date"] = str(sample_metadata["sequence_start_date"])
                        if "sequence_end_date" in sample_metadata:
                            result_entry["sequence_end_date"] = str(sample_metadata["sequence_end_date"])
                        if "target_start_date" in sample_metadata:
                            result_entry["target_start_date"] = str(sample_metadata["target_start_date"])
                        if "target_end_date" in sample_metadata:
                            result_entry["target_end_date"] = str(sample_metadata["target_end_date"])
                        if "current_idx" in sample_metadata:
                            result_entry["time_idx"] = sample_metadata["current_idx"]
                    
                    # Add predictions (factors)
                    if stock_predictions.ndim == 0:  # Scalar
                        result_entry["factor_0"] = float(stock_predictions)
                    else:  # Vector
                        for factor_idx, factor_value in enumerate(stock_predictions):
                            result_entry[f"factor_{factor_idx}"] = float(factor_value)
                    
                    # Add targets if available
                    if stock_targets is not None:
                        if isinstance(stock_targets, (list, np.ndarray)) and hasattr(stock_targets, '__len__') and len(stock_targets) > 1:
                            for target_idx, target_value in enumerate(stock_targets):
                                result_entry[f"target_{target_idx}"] = float(target_value)
                        else:
                            result_entry["target"] = float(stock_targets)
                    
                    results.append(result_entry)
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"   Processed batch {batch_idx}/{len(prediction_loader)}")
    
    print(f"✓ Completed predictions. Total results: {len(results)}")
    return results


def save_predictions(results: List[Dict[str, Any]], output_path: str):
    """Save predictions to a CSV file"""
    if not results:
        print("⚠️  No results to save")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by date and stock_code for better organization
    df = df.sort_values(["date", "stock_code"])
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"💾 Saved {len(results)} predictions to: {output_path}")
    print(f"📊 Data shape: {df.shape}")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"🏢 Number of unique stocks: {df['stock_code'].nunique()}")
    print(f"📈 Factor columns: {[col for col in df.columns if col.startswith('factor_')]}")


def main():
    parser = argparse.ArgumentParser(description="Predict factors using trained Stock GNN model")
    parser.add_argument("--checkpoint", required=True, type=str, 
                       help="Path to the model checkpoint (.ckpt file)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to the configuration file (optional)")
    parser.add_argument("--output", type=str, default="predictions.csv",
                       help="Output CSV file path")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔮 Stock GNN Prediction Pipeline")
    print("=" * 80)
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🖥️  Using device: {device}")
    
    # Load configuration
    print("\n📋 Loading configuration...")
    cfg = load_config(args.config)
    print(f"   ✓ Configuration loaded")
    
    # Create data module
    print("\n📊 Setting up data module...")
    try:
        dm = instantiate(cfg.data)
        dm.prepare_data()
        dm.setup("test")
        print(f"   ✓ Data module ready")
        print(f"   - Feature dimension: {dm.get_feature_dim()}")
        print(f"   - Number of stocks: {dm.get_stock_num()}")
        print(f"   - Prediction horizons: {dm.get_prediction_horizons()}")
    except Exception as e:
        print(f"   ❌ Failed to setup data module: {e}")
        return
    
    # Load model
    print(f"\n🧠 Loading model from checkpoint...")
    try:
        model = load_model_from_checkpoint(args.checkpoint)
        print(f"   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return
    
    # Run predictions
    print(f"\n🔮 Running predictions on test dataset...")
    try:
        results = predict_with_metadata(model, dm, device)
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        return
    
    # Save results
    print(f"\n💾 Saving predictions...")
    try:
        save_predictions(results, args.output)
        print(f"   ✓ Predictions saved successfully")
    except Exception as e:
        print(f"   ❌ Failed to save predictions: {e}")
        return
    
    print(f"\n✅ Prediction pipeline completed successfully!")
    print(f"📄 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
