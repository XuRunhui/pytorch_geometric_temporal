#!/usr/bin/env python3
"""
Feature Extraction Script for Trained Stock GNN Model

This script loads a trained model checkpoint and extracts 32-dimensional features
for the entire dataset, maintaining original data structure and handling missing values.
"""

import os
import sys
import gzip
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from tqdm import tqdm
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

# Import required modules
from torch_geometric_temporal.nn.recurrent.stock_GNN.dataset.stock_dataset import StockDataModule, StockDataset
from torch_geometric_temporal.nn.recurrent.stock_GNN.training_module import DynamicGraphLightning

# Set up safe globals for PyTorch 2.6+ compatibility
try:
    import torch_geometric_temporal.nn.recurrent.stock_GNN.models.dynamic_gat as dynamic_gat_module
    torch.serialization.add_safe_globals([
        dynamic_gat_module.Dynamic_Gat,
        DynamicGraphLightning
    ])
except (ImportError, AttributeError) as e:
    # If safe globals are not available or modules not found, continue anyway
    pass


class FeatureExtractor:
    """Extract features from trained Stock GNN model"""
    
    def __init__(self, 
                 checkpoint_path: str,
                 data_dir: str,
                 output_dir: str,
                 device: str = 'auto',
                 batch_size: int = 1,
                 save_mode: str = 'complete'):
        """
        Initialize feature extractor
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            data_dir: Directory containing input data
            output_dir: Directory to save extracted features
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            batch_size: Batch size for inference
            save_mode: How to save features ('complete', 'separate', 'both')
        """
        self.checkpoint_path = checkpoint_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.save_mode = save_mode
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.datamodule = None
        self.original_data_shape = None
        self.original_dates = None
        self.original_stocks = None
        
    def load_model(self) -> None:
        """Load trained model from checkpoint"""
        self.logger.info(f"Loading model from {self.checkpoint_path}")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Load model from checkpoint with PyTorch 2.6+ compatibility
        try:
            self.model = DynamicGraphLightning.load_from_checkpoint(
                self.checkpoint_path,
                map_location=self.device
            )
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.logger.info("Tip: This might be due to PyTorch 2.6+ security changes.")
            self.logger.info("Try setting PYTORCH_ENABLE_MPS_FALLBACK=1 or use an earlier PyTorch version.")
            raise
        
        self.model.eval()
        self.model.to(self.device)
        
        self.logger.info(f"Model loaded on device: {self.device}")
        self.logger.info(f"Model architecture: {type(self.model.core_model).__name__}")
    
    def setup_data(self) -> None:
        """Setup data module and load original data structure"""
        self.logger.info("Setting up data module...")
        
        # Create data module with same parameters as training
        self.datamodule = StockDataModule(
            data_dir=self.data_dir,
            use_factors=True,
            sequence_length=30,  # Use same as training
            prediction_horizons=[1],
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            batch_size=self.batch_size,
            num_workers=0,  # Avoid multiprocessing issues
            normalize_features=True,
            normalize_targets=True,
            debug=False
        )
        
        # Setup data
        self.datamodule.prepare_data()
        self.datamodule.setup()
        
        # Store original data structure info
        self.original_dates = self.datamodule.date_index
        self.original_stocks = self.datamodule.stock_names
        self.original_data_shape = (len(self.original_dates), len(self.original_stocks))
        
        self.logger.info(f"Original data shape: {self.original_data_shape}")
        self.logger.info(f"Date range: {self.original_dates[0]} to {self.original_dates[-1]}")
        self.logger.info(f"Number of stocks: {len(self.original_stocks)}")
    
    def extract_features_from_dataset(self, dataset: StockDataset, dataset_name: str) -> Dict[str, np.ndarray]:
        """
        Extract features from a dataset
        
        Args:
            dataset: Dataset to extract features from
            dataset_name: Name of dataset (train/val/test)
            
        Returns:
            Dictionary containing extracted features and metadata
        """
        self.logger.info(f"Extracting features from {dataset_name} dataset...")
        
        # Create dataloader with custom collate function to handle metadata issues
        def safe_collate(batch):
            """Custom collate function to handle pandas Timestamp objects in metadata"""
            if len(batch[0]) == 3:
                # Has metadata - need to handle it carefully
                features_list = []
                targets_list = []
                metadata_list = []
                
                for features, targets, metadata in batch:
                    features_list.append(features)
                    targets_list.append(targets)
                    
                    # Convert problematic types in metadata
                    if isinstance(metadata, dict):
                        safe_metadata = {}
                        for key, value in metadata.items():
                            if hasattr(value, 'timestamp'):  # pandas Timestamp
                                safe_metadata[key] = str(value)
                            elif hasattr(value, 'tolist'):  # numpy array or tensor
                                safe_metadata[key] = value
                            else:
                                safe_metadata[key] = value
                        metadata_list.append(safe_metadata)
                    else:
                        metadata_list.append(metadata)
                
                return (
                    torch.stack(features_list),
                    torch.stack(targets_list),
                    metadata_list
                )
            else:
                # No metadata, use default collate
                return torch.utils.data.default_collate(batch)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False,  # Keep order for alignment
            num_workers=0,
            collate_fn=safe_collate
        )
        
        # Storage for features and metadata
        all_features = []
        all_sample_indices = []
        all_valid_stocks_info = []
        
        # Extract features
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Processing {dataset_name}")):
                if len(batch_data) == 3:
                    # With metadata
                    features, targets, metadata = batch_data
                    metadata_dict = metadata[0] if isinstance(metadata, list) else metadata
                    sample_idx = metadata_dict['current_idx']
                    valid_stock_names = metadata_dict.get('valid_stock_names', [])
                else:
                    # Without metadata, need to compute sample index
                    features, targets = batch_data
                    sample_idx = dataset.valid_indices[batch_idx]
                    # Get valid stocks for this sample - fallback method
                    valid_stock_mask = dataset._get_valid_stocks_for_sequence(features[0])
                    # Convert mask to stock names
                    valid_stock_names = [dataset.stock_names[i] for i in range(len(valid_stock_mask)) if valid_stock_mask[i]]
                
                # Move to device
                features = features.to(self.device)
                
                # Extract features from model
                batch_features = self._extract_model_features(features)
                
                # Store results
                all_features.append(batch_features.cpu().numpy())
                all_sample_indices.append(sample_idx)
                all_valid_stocks_info.append({
                    'valid_stock_names': valid_stock_names,
                    'n_valid_stocks': len(valid_stock_names)
                })
        
        return {
            'features': all_features,
            'sample_indices': all_sample_indices,
            'valid_stocks_info': all_valid_stocks_info,
            'dataset_name': dataset_name
        }
    
    def _extract_model_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from model using direct forward pass
        
        Args:
            x: Input features [batch_size, sequence_length, feature_dim, n_stocks]
            
        Returns:
            Features from model output [batch_size, n_stocks, feature_dim]
        """
        # Simply use the model's forward pass - much simpler!
        core_model = self.model.core_model
        features = core_model(x)
        
        self.logger.info(f"Model output shape: {features.shape}")
        return features
    
    def align_features_to_original_grid(self, extracted_data: Dict) -> List[pd.DataFrame]:
        """
        Align extracted features to original data grid using valid_stock_names from metadata
        
        Args:
            extracted_data: Dictionary containing extracted features and metadata
            
        Returns:
            List of DataFrames with proper date/stock indexing and NaN for missing values
        """
        self.logger.info(f"Aligning features for {extracted_data['dataset_name']} dataset...")
        
        # Get feature dimensions from first sample
        first_features = extracted_data['features'][0]
        if len(first_features.shape) == 3:  # [batch_size, n_stocks, feature_dim]
            feature_dim = first_features.shape[-1]
        elif len(first_features.shape) == 2:  # [n_stocks, feature_dim]
            feature_dim = first_features.shape[-1]
        else:
            raise ValueError(f"Unexpected feature shape: {first_features.shape}")
        
        self.logger.info(f"Detected feature dimension: {feature_dim}")
        
        # Create result array filled with NaN
        # Shape: [n_dates, n_stocks, feature_dim]
        result_array = np.full(
            (len(self.original_dates), len(self.original_stocks), feature_dim), 
            np.nan, 
            dtype=np.float32
        )
        
        # Statistics for debugging
        filled_positions = 0
        total_positions = len(self.original_dates) * len(self.original_stocks)
        
        # Fill in available features using valid_stock_names
        for i, (features, sample_idx, stock_info) in enumerate(zip(
            extracted_data['features'],
            extracted_data['sample_indices'], 
            extracted_data['valid_stocks_info']
        )):
            if sample_idx < len(self.original_dates):
                # Get valid stock names from metadata
                valid_stock_names = stock_info.get('valid_stock_names', [])
                
                # Handle different feature shapes
                if len(features.shape) == 3:  # [batch_size, n_stocks, feature_dim]
                    features = features[0]  # Take first batch element
                elif len(features.shape) == 2:  # [n_stocks, feature_dim]
                    pass  # Already correct shape
                else:
                    self.logger.warning(f"Unexpected feature shape: {features.shape}")
                    continue
                
                # Debug info for first few samples
                if i < 5:
                    self.logger.info(f"Sample {i}: date_idx={sample_idx}, date={self.original_dates[sample_idx]}")
                    self.logger.info(f"  Valid stocks: {len(valid_stock_names)}")
                    self.logger.info(f"  Feature shape: {features.shape}")
                    self.logger.info(f"  Sample valid stocks: {valid_stock_names[:5] if len(valid_stock_names) > 5 else valid_stock_names}")
                
                # Map features to stock positions using stock names
                if len(valid_stock_names) == features.shape[0]:
                    for j, stock_name in enumerate(valid_stock_names):
                        if stock_name in self.original_stocks:
                            stock_idx = self.original_stocks.index(stock_name)
                            result_array[sample_idx, stock_idx, :] = features[j]
                            filled_positions += 1
                        else:
                            self.logger.warning(f"Stock {stock_name} not found in original stocks")
                else:
                    self.logger.warning(f"Mismatch in stock count at index {sample_idx}: "
                                      f"expected {len(valid_stock_names)}, got {features.shape[0]}")
        
        # Log coverage statistics
        coverage = filled_positions / total_positions * 100
        self.logger.info(f"Data coverage: {filled_positions}/{total_positions} ({coverage:.2f}%)")
        
        # Convert to list of DataFrames (one per feature dimension)
        feature_dfs = []
        for feat_dim in range(feature_dim):
            df = pd.DataFrame(
                result_array[:, :, feat_dim],
                index=pd.Index(self.original_dates, name='date'),
                columns=pd.Index(self.original_stocks, name='stock')
            )
            
            # Log statistics for this feature dimension
            non_nan_count = df.notna().sum().sum()
            total_count = df.size
            coverage_dim = non_nan_count / total_count * 100
            
            self.logger.info(f"Feature dim {feat_dim}: {non_nan_count}/{total_count} non-NaN ({coverage_dim:.2f}%)")
            
            feature_dfs.append(df)
        
        return feature_dfs
    
    def save_features(self, feature_dfs: List[pd.DataFrame], dataset_name: str) -> None:
        """
        Save extracted features in the same format as original data
        
        Args:
            feature_dfs: List of DataFrames (one per feature dimension)
            dataset_name: Name of dataset (train/val/test)
        """
        self.logger.info(f"Saving {dataset_name} features...")
        
        feature_dim = len(feature_dfs)
        
        for feat_dim, df in enumerate(feature_dfs):
            filename = f"extracted_features_{dataset_name}_dim{feat_dim:02d}.pkl"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save with gzip compression like original data
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(df, f)
        
        # Also save a summary file
        summary = {
            'dataset_name': dataset_name,
            'feature_dimensions': feature_dim,
            'data_shape': feature_dfs[0].shape,
            'date_range': (str(feature_dfs[0].index[0]), str(feature_dfs[0].index[-1])),
            'n_stocks': len(feature_dfs[0].columns),
            'extraction_time': datetime.now().isoformat(),
            'checkpoint_path': self.checkpoint_path
        }
        
        summary_path = os.path.join(self.output_dir, f"extraction_summary_{dataset_name}.json")
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Saved {dataset_name} features to {self.output_dir}")
        self.logger.info(f"Feature dimensions: {feature_dim}")
    
    def extract_all_features(self) -> None:
        """Extract features from all datasets and combine them into a single output"""
        self.logger.info("Starting feature extraction from all datasets...")
        
        if self.save_mode == 'complete':
            self._extract_complete_dataset()
        elif self.save_mode == 'separate':
            self._extract_separate_datasets()
        elif self.save_mode == 'both':
            self._extract_complete_dataset()
            self._extract_separate_datasets()
        else:
            raise ValueError(f"Invalid save_mode: {self.save_mode}")
    
    def _extract_complete_dataset(self) -> None:
        """Extract and save complete dataset"""
        self.logger.info("Extracting features from complete dataset...")
        
        all_features = []
        all_sample_indices = []
        all_valid_stocks_info = []
        
        # Extract from all datasets
        for dataset_name, dataset in [
            ('train', self.datamodule.train_dataset),
            ('val', self.datamodule.val_dataset), 
            ('test', self.datamodule.test_dataset)
        ]:
            self.logger.info(f"Processing {dataset_name} dataset...")
            
            dataset_features = self.extract_features_from_dataset(dataset, dataset_name)
            
            # Combine all features
            all_features.extend(dataset_features['features'])
            all_sample_indices.extend(dataset_features['sample_indices'])
            all_valid_stocks_info.extend(dataset_features['valid_stocks_info'])
        
        # Create combined dataset dictionary
        combined_data = {
            'features': all_features,
            'sample_indices': all_sample_indices,
            'valid_stocks_info': all_valid_stocks_info,
            'dataset_name': 'complete'
        }
        
        # Align to original grid
        self.logger.info("Aligning combined features to original data grid...")
        combined_dfs = self.align_features_to_original_grid(combined_data)
        
        # Save combined features
        self.save_features(combined_dfs, 'complete')
        
        # Also save dataset split information for reference
        self._save_split_info()
        
        self.logger.info("Complete dataset feature extraction completed!")
    
    def _extract_separate_datasets(self) -> None:
        """Extract and save separate train/val/test datasets"""
        self.logger.info("Starting feature extraction from separate datasets...")
        
        # Extract from train dataset
        train_features = self.extract_features_from_dataset(
            self.datamodule.train_dataset, 'train'
        )
        train_dfs = self.align_features_to_original_grid(train_features)
        self.save_features(train_dfs, 'train')
        
        # Extract from validation dataset
        val_features = self.extract_features_from_dataset(
            self.datamodule.val_dataset, 'val'
        )
        val_dfs = self.align_features_to_original_grid(val_features)
        self.save_features(val_dfs, 'val')
        
        # Extract from test dataset
        test_features = self.extract_features_from_dataset(
            self.datamodule.test_dataset, 'test'
        )
        test_dfs = self.align_features_to_original_grid(test_features)
        self.save_features(test_dfs, 'test')
        
        self.logger.info("Separate dataset feature extraction completed!")
    
    def _save_split_info(self) -> None:
        """Save information about train/val/test splits for reference"""
        total_samples = len(self.original_dates)
        train_end = int(total_samples * 0.7)
        val_end = int(total_samples * 0.85)
        
        split_info = {
            'total_samples': total_samples,
            'train_range': {
                'start_idx': 0,
                'end_idx': train_end,
                'start_date': str(self.original_dates[0]),
                'end_date': str(self.original_dates[train_end-1]) if train_end > 0 else str(self.original_dates[0]),
                'n_samples': train_end
            },
            'val_range': {
                'start_idx': train_end,
                'end_idx': val_end,
                'start_date': str(self.original_dates[train_end]) if train_end < len(self.original_dates) else str(self.original_dates[-1]),
                'end_date': str(self.original_dates[val_end-1]) if val_end > train_end else str(self.original_dates[train_end]),
                'n_samples': val_end - train_end
            },
            'test_range': {
                'start_idx': val_end,
                'end_idx': total_samples,
                'start_date': str(self.original_dates[val_end]) if val_end < len(self.original_dates) else str(self.original_dates[-1]),
                'end_date': str(self.original_dates[-1]),
                'n_samples': total_samples - val_end
            }
        }
        
        split_info_path = os.path.join(self.output_dir, "dataset_split_info.json")
        import json
        with open(split_info_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        self.logger.info(f"Dataset split information saved to {split_info_path}")
        
        self.logger.info("Feature extraction completed!")
    
    def run(self) -> None:
        """Run the complete feature extraction pipeline"""
        self.logger.info("Starting feature extraction pipeline...")
        
        # Load model
        self.load_model()
        
        # Setup data
        self.setup_data()
        
        # Extract features
        self.extract_all_features()
        
        self.logger.info("Feature extraction pipeline completed successfully!")


def load_complete_features(feature_dir: str, feature_dims: List[int] = None) -> Tuple[List[pd.DataFrame], Dict]:
    """
    Load extracted features from the complete dataset
    
    Args:
        feature_dir: Directory containing extracted features
        feature_dims: Specific feature dimensions to load (None for all)
        
    Returns:
        Tuple of (feature_dataframes, metadata)
    """
    import json
    
    # Load metadata
    summary_path = os.path.join(feature_dir, "extraction_summary_complete.json")
    split_info_path = os.path.join(feature_dir, "dataset_split_info.json")
    
    metadata = {}
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            metadata['extraction_summary'] = json.load(f)
    
    if os.path.exists(split_info_path):
        with open(split_info_path, 'r') as f:
            metadata['split_info'] = json.load(f)
    
    # Determine feature dimensions to load
    if feature_dims is None:
        # Load all available dimensions
        feature_files = [f for f in os.listdir(feature_dir) 
                        if f.startswith('extracted_features_complete_dim') and f.endswith('.pkl')]
        feature_dims = sorted([int(f.split('_dim')[1].split('.')[0]) for f in feature_files])
    
    # Load feature DataFrames
    feature_dfs = []
    for dim in feature_dims:
        filename = f"extracted_features_complete_dim{dim:02d}.pkl"
        filepath = os.path.join(feature_dir, filename)
        
        if os.path.exists(filepath):
            with gzip.open(filepath, 'rb') as f:
                df = pickle.load(f)
                feature_dfs.append(df)
        else:
            print(f"Warning: Feature dimension {dim} not found at {filepath}")
    
    return feature_dfs, metadata


def split_features_by_dataset(feature_dfs: List[pd.DataFrame], 
                             split_info: Dict) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Split the complete feature DataFrames back into train/val/test
    
    Args:
        feature_dfs: List of feature DataFrames for the complete dataset
        split_info: Split information dictionary
        
    Returns:
        Tuple of (train_dfs, val_dfs, test_dfs)
    """
    train_range = split_info['train_range']
    val_range = split_info['val_range'] 
    test_range = split_info['test_range']
    
    train_dfs = [df.iloc[train_range['start_idx']:train_range['end_idx']] for df in feature_dfs]
    val_dfs = [df.iloc[val_range['start_idx']:val_range['end_idx']] for df in feature_dfs]
    test_dfs = [df.iloc[test_range['start_idx']:test_range['end_idx']] for df in feature_dfs]
    
    return train_dfs, val_dfs, test_dfs


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Extract features from trained Stock GNN model")
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=False,
        help="Path to trained model checkpoint (.ckpt file)",
        default="logs/gat_experiment/version_0/checkpoints/stock-gnn-00-50.8377.ckpt"
    )
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="/root/pytorch_geometric_temporal/data",
        help="Directory containing input data"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./extracted_features",
        help="Directory to save extracted features"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1,
        help="Batch size for inference"
    )
    
    parser.add_argument(
        "--save-mode", 
        type=str, 
        default="complete",
        choices=["complete", "separate", "both"],
        help="How to save features: 'complete' (single dataset), 'separate' (train/val/test), 'both'"
    )
    
    args = parser.parse_args()
    
    # Create feature extractor
    extractor = FeatureExtractor(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        save_mode=args.save_mode
    )
    
    # Run extraction
    extractor.run()


if __name__ == "__main__":
    main()
