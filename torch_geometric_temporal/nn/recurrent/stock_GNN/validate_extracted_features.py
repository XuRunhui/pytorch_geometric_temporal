#!/usr/bin/env python3
"""
Validation script to check extracted features
"""

import os
import gzip
import pickle
import pandas as pd
import numpy as np
import json
from pathlib import Path

def load_extracted_features(feature_dir: str, dataset_name: str = 'complete') -> tuple:
    """
    Load extracted features and check their properties
    
    Args:
        feature_dir: Directory containing extracted features
        dataset_name: Name of dataset ('complete', 'train', 'val', 'test')
        
    Returns:
        Tuple of (feature_list, summary_info)
    """
    print(f"\n📊 Checking {dataset_name} features...")
    
    # Load summary
    summary_path = os.path.join(feature_dir, f"extraction_summary_{dataset_name}.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        print(f"   ✓ Summary loaded: {summary['data_shape']} shape")
    else:
        summary = None
        print("   ⚠️  Summary file not found")
    
    # Detect available feature dimensions
    feature_files = [f for f in os.listdir(feature_dir) 
                    if f.startswith(f'extracted_features_{dataset_name}_dim') and f.endswith('.pkl')]
    
    if not feature_files:
        print(f"   ❌ No feature files found for {dataset_name}")
        return None, None
    
    # Extract dimension numbers and sort
    available_dims = sorted([int(f.split('_dim')[1].split('.')[0]) for f in feature_files])
    print(f"   📋 Available dimensions: {available_dims}")
    
    # Load all available feature dimensions
    features = []
    for dim in available_dims:
        filename = f"extracted_features_{dataset_name}_dim{dim:02d}.pkl"
        filepath = os.path.join(feature_dir, filename)
        
        if os.path.exists(filepath):
            try:
                with gzip.open(filepath, 'rb') as f:
                    df = pickle.load(f)
                features.append(df)
                print(f"   ✓ Loaded dim {dim:02d}: {df.shape}")
            except Exception as e:
                print(f"   ❌ Error loading {filename}: {e}")
                return None, None
        else:
            print(f"   ❌ Missing file: {filename}")
            return None, None
    
    print(f"   ✅ Successfully loaded {len(features)} feature dimensions")
    return features, summary

def validate_feature_structure(features: list, dataset_name: str):
    """Validate the structure and properties of extracted features"""
    print(f"\n🔍 Validating {dataset_name} feature structure...")
    
    if not features:
        print("   ❌ No features to validate")
        return
    
    # Check consistency across dimensions
    first_shape = features[0].shape
    first_index = features[0].index
    first_columns = features[0].columns
    
    print(f"   📐 Data shape: {first_shape}")
    print(f"   📅 Date range: {first_index[0]} to {first_index[-1]}")
    print(f"   📈 Number of stocks: {len(first_columns)}")
    
    # Check consistency across all dimensions
    all_consistent = True
    for i, df in enumerate(features):
        if df.shape != first_shape:
            print(f"   ❌ Dimension {i}: shape mismatch {df.shape} vs {first_shape}")
            all_consistent = False
        if not df.index.equals(first_index):
            print(f"   ❌ Dimension {i}: index mismatch")
            all_consistent = False
        if not df.columns.equals(first_columns):
            print(f"   ❌ Dimension {i}: columns mismatch")
            all_consistent = False
    
    if all_consistent:
        print("   ✅ All dimensions have consistent structure")
    
    # Check data quality
    for i, df in enumerate(features[:3]):  # Check first 3 dimensions
        nan_count = df.isna().sum().sum()
        total_count = df.size
        nan_percentage = (nan_count / total_count) * 100
        
        value_range = (df.min().min(), df.max().max())
        
        print(f"   📊 Dim {i:02d}: {nan_percentage:.1f}% NaN, range: [{value_range[0]:.4f}, {value_range[1]:.4f}]")
    
    if len(features) > 3:
        print(f"   ... (showing first 3 of {len(features)} dimensions)")

def compare_with_original_data(features: list, original_data_dir: str, dataset_name: str):
    """Compare extracted features with original data structure"""
    print(f"\n🔄 Comparing with original data structure...")
    
    # Load a sample original file
    sample_file = os.path.join(original_data_dir, "filtered_close_adj.pkl")
    if not os.path.exists(sample_file):
        print(f"   ⚠️  Original data not found: {sample_file}")
        return
    
    try:
        with gzip.open(sample_file, 'rb') as f:
            original_df = pickle.load(f)
    except Exception as e:
        print(f"   ❌ Error loading original data: {e}")
        return
    
    print(f"   📐 Original data shape: {original_df.shape}")
    print(f"   📐 Extracted features shape: {features[0].shape}")
    
    # Check index alignment
    if features[0].index.equals(original_df.index):
        print("   ✅ Date indices perfectly aligned")
    else:
        print("   ⚠️  Date indices not perfectly aligned")
        print(f"      Original dates: {len(original_df.index)}, Features dates: {len(features[0].index)}")
        
        # Check overlap
        common_dates = set(features[0].index) & set(original_df.index)
        print(f"      Common dates: {len(common_dates)}")
    
    # Check column alignment
    if features[0].columns.equals(original_df.columns):
        print("   ✅ Stock columns perfectly aligned")
    else:
        print("   ⚠️  Stock columns not perfectly aligned")
        print(f"      Original stocks: {len(original_df.columns)}, Features stocks: {len(features[0].columns)}")
        
        # Check overlap
        common_stocks = set(features[0].columns) & set(original_df.columns)
        print(f"      Common stocks: {len(common_stocks)}")

def load_split_info(feature_dir: str) -> dict:
    """Load dataset split information for complete features"""
    split_info_path = os.path.join(feature_dir, "dataset_split_info.json")
    if os.path.exists(split_info_path):
        with open(split_info_path, 'r') as f:
            split_info = json.load(f)
        print(f"   ✓ Split info loaded:")
        print(f"      Train: {split_info['train_range']['start_date']} to {split_info['train_range']['end_date']} ({split_info['train_range']['n_samples']} samples)")
        print(f"      Val:   {split_info['val_range']['start_date']} to {split_info['val_range']['end_date']} ({split_info['val_range']['n_samples']} samples)")
        print(f"      Test:  {split_info['test_range']['start_date']} to {split_info['test_range']['end_date']} ({split_info['test_range']['n_samples']} samples)")
        return split_info
    else:
        print("   ⚠️  Split info not found")
        return None

def analyze_temporal_coverage(features: list, split_info: dict = None):
    """Analyze temporal coverage patterns in features"""
    print(f"\n� Analyzing temporal coverage patterns...")
    
    if not features:
        return
    
    # Use first feature dimension for analysis
    df = features[0]
    
    # Calculate coverage per date
    coverage_per_date = df.notna().sum(axis=1) / len(df.columns)
    
    print(f"   📊 Overall temporal statistics:")
    print(f"      Average daily coverage: {coverage_per_date.mean():.1%}")
    print(f"      Min daily coverage: {coverage_per_date.min():.1%}")
    print(f"      Max daily coverage: {coverage_per_date.max():.1%}")
    
    # Find first and last dates with data
    dates_with_data = coverage_per_date[coverage_per_date > 0]
    if len(dates_with_data) > 0:
        first_data_date = dates_with_data.index[0]
        last_data_date = dates_with_data.index[-1]
        print(f"      First data date: {first_data_date}")
        print(f"      Last data date: {last_data_date}")
        print(f"      Days with data: {len(dates_with_data)}/{len(coverage_per_date)}")
    
    # Analyze early period (should be mostly NaN due to 30-day requirement)
    early_period = coverage_per_date.iloc[:50]  # First 50 days
    early_with_data = (early_period > 0).sum()
    print(f"      Early period (first 50 days) with data: {early_with_data}/50")
    
    if early_with_data > 30:
        print("      ⚠️  Warning: Many early dates have data, check 30-day filtering")
    else:
        print("      ✅ Early dates properly filtered (as expected)")
    
    # Split-based analysis if split info is available
    if split_info:
        print(f"\n   📊 Coverage by split:")
        
        for split_name in ['train', 'val', 'test']:
            split_range = split_info[f'{split_name}_range']
            start_idx = split_range['start_idx']
            end_idx = split_range['end_idx']
            
            split_coverage = coverage_per_date.iloc[start_idx:end_idx]
            avg_coverage = split_coverage.mean()
            days_with_data = (split_coverage > 0).sum()
            
            print(f"      {split_name.upper():>5}: {avg_coverage:.1%} avg coverage, {days_with_data}/{len(split_coverage)} days with data")

def analyze_stock_coverage(features: list):
    """Analyze which stocks have features most frequently"""
    print(f"\n📈 Analyzing stock coverage patterns...")
    
    if not features:
        return
    
    # Use first feature dimension
    df = features[0]
    
    # Calculate coverage per stock
    coverage_per_stock = df.notna().sum(axis=0) / len(df.index)
    
    print(f"   📊 Stock coverage statistics:")
    print(f"      Average stock coverage: {coverage_per_stock.mean():.1%}")
    print(f"      Min stock coverage: {coverage_per_stock.min():.1%}")
    print(f"      Max stock coverage: {coverage_per_stock.max():.1%}")
    
    # Find stocks with different coverage levels
    high_coverage_stocks = coverage_per_stock[coverage_per_stock >= 0.8]
    medium_coverage_stocks = coverage_per_stock[(coverage_per_stock >= 0.5) & (coverage_per_stock < 0.8)]
    low_coverage_stocks = coverage_per_stock[coverage_per_stock < 0.5]
    
    print(f"      High coverage (≥80%): {len(high_coverage_stocks)} stocks")
    print(f"      Medium coverage (50-80%): {len(medium_coverage_stocks)} stocks")
    print(f"      Low coverage (<50%): {len(low_coverage_stocks)} stocks")
    
    # Show top and bottom stocks
    top_stocks = coverage_per_stock.nlargest(5)
    bottom_stocks = coverage_per_stock.nsmallest(5)
    
    print(f"\n   📊 Top 5 stocks by coverage:")
    for stock, coverage in top_stocks.items():
        print(f"      {stock}: {coverage:.1%}")
    
    print(f"\n   � Bottom 5 stocks by coverage:")
    for stock, coverage in bottom_stocks.items():
        print(f"      {stock}: {coverage:.1%}")

def check_data_consistency(features: list):
    """Check consistency across different feature dimensions"""
    print(f"\n🔄 Checking data consistency across dimensions...")
    
    if len(features) < 2:
        print("   ⚠️  Need at least 2 dimensions for consistency check")
        return
    
    # Check NaN pattern consistency
    first_nan_mask = features[0].isna()
    consistent_nan_pattern = True
    
    for i, df in enumerate(features[1:], 1):
        if not df.isna().equals(first_nan_mask):
            consistent_nan_pattern = False
            break
    
    if consistent_nan_pattern:
        print("   ✅ NaN patterns are consistent across all dimensions")
    else:
        print("   ⚠️  NaN patterns differ across dimensions")
        # Count differences for each dimension
        for i, df in enumerate(features):
            diff_count = (df.isna() != first_nan_mask).sum().sum()
            if diff_count > 0:
                print(f"      Dim {i:02d}: {diff_count} differences with dim 00")

def validate_complete_features():
    """Main validation function for complete features"""
    print("🔍 Complete Feature Validation Tool")
    print("=" * 50)
    
    # Configuration
    feature_dir = "./extracted_features"
    original_data_dir = "/root/pytorch_geometric_temporal/data"
    
    if not os.path.exists(feature_dir):
        print(f"❌ Feature directory not found: {feature_dir}")
        print("Please run feature extraction first or update the feature_dir path")
        return
    
    # Load complete features
    features, summary = load_extracted_features(feature_dir, 'complete')
    
    if features is None:
        print("❌ Could not load complete features")
        return
    
    # Load split information
    split_info = load_split_info(feature_dir)
    
    # Run all validation checks
    validate_feature_structure(features, 'complete')
    compare_with_original_data(features, original_data_dir, 'complete')
    analyze_temporal_coverage(features, split_info)
    analyze_stock_coverage(features)
    check_data_consistency(features)
    feature_stats = generate_feature_statistics(features, 'complete')
    
    print("\n✅ Complete feature validation finished!")
    print("\n📋 Summary:")
    print(f"   - Feature dimensions: {len(features)}")
    print(f"   - Data shape: {features[0].shape}")
    print(f"   - Date range: {features[0].index[0]} to {features[0].index[-1]}")
    print(f"   - Stock count: {len(features[0].columns)}")
    
    return features, summary, split_info

def generate_feature_statistics(features: list, dataset_name: str):
    """Generate comprehensive statistics for features"""
    print(f"\n📈 Generating statistics for {dataset_name} features...")
    
    # Combine all features into a single array for analysis
    all_features = np.stack([df.values for df in features], axis=-1)  # [dates, stocks, features]
    
    print(f"   📐 Combined array shape: {all_features.shape}")
    
    # Overall statistics
    valid_mask = ~np.isnan(all_features)
    valid_ratio = valid_mask.sum() / all_features.size
    print(f"   ✅ Valid data ratio: {valid_ratio:.1%}")
    
    # Per-dimension statistics
    feature_stats = []
    for dim in range(all_features.shape[2]):
        dim_data = all_features[:, :, dim]
        valid_data = dim_data[~np.isnan(dim_data)]
        
        if len(valid_data) > 0:
            stats = {
                'mean': np.mean(valid_data),
                'std': np.std(valid_data),
                'min': np.min(valid_data),
                'max': np.max(valid_data),
                'valid_ratio': len(valid_data) / dim_data.size
            }
        else:
            stats = {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'valid_ratio': 0.0}
        
        feature_stats.append(stats)
    
    # Show statistics for first few dimensions
    print("\n   📊 Per-dimension statistics (first 5):")
    print("   Dim |   Mean   |   Std    |   Min    |   Max    | Valid%")
    print("   ----|----------|----------|----------|----------|-------")
    for i in range(min(50, len(feature_stats))):
        stats = feature_stats[i]
        print(f"   {i:2d}  | {stats['mean']:8.4f} | {stats['std']:8.4f} | {stats['min']:8.4f} | {stats['max']:8.4f} | {stats['valid_ratio']:5.1%}")
    
    return feature_stats

def main():
    """Main validation function - supports both complete and separate validation"""
    print("🔍 Feature Extraction Validation Tool")
    print("=" * 50)
    
    # Configuration
    feature_dir = "./extracted_features"
    original_data_dir = "/root/pytorch_geometric_temporal/data"
    
    if not os.path.exists(feature_dir):
        print(f"❌ Feature directory not found: {feature_dir}")
        print("Please run feature extraction first or update the feature_dir path")
        return
    
    # Check what type of features are available
    complete_files = [f for f in os.listdir(feature_dir) if f.startswith('extracted_features_complete_')]
    separate_files = [f for f in os.listdir(feature_dir) if f.startswith('extracted_features_train_')]
    
    if complete_files:
        print("\n📦 Complete features detected - validating complete dataset...")
        validate_complete_features()
    elif separate_files:
        print("\n📦 Separate features detected - validating individual datasets...")
        
        # Validate each dataset separately
        for dataset_name in ['train', 'val', 'test']:
            features, summary = load_extracted_features(feature_dir, dataset_name)
            
            if features is not None:
                validate_feature_structure(features, dataset_name)
                compare_with_original_data(features, original_data_dir, dataset_name)
                feature_stats = generate_feature_statistics(features, dataset_name)
            else:
                print(f"❌ Could not load {dataset_name} features")
    else:
        print("❌ No extracted features found in the directory")
        return
    
    print("\n✅ Validation completed!")
    print("\n💡 Usage tips:")
    print("1. Features are saved as separate files per dimension")
    print("2. Each file contains a pandas DataFrame with date×stock structure")
    print("3. NaN values indicate missing/filtered data points")
    print("4. For complete features, use load_complete_features() helper function")
    print("5. Check temporal coverage to verify 30-day filtering worked correctly")

if __name__ == "__main__":
    main()
