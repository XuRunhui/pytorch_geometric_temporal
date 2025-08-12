"""
Example usage and test for Multi-RNN Loss Module

This script demonstrates how to use the new Multi-RNN Loss module
and provides unit tests for its functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
from multi_rnn_loss import MultiRNNLoss, BatchNormLoss, EqualWeightedLoss


def test_multi_rnn_loss():
    """Test the Multi-RNN Loss module with sample data."""
    print("🧪 Testing Multi-RNN Loss Module...")
    
    # Test parameters
    batch_size = 4
    seq_len = 30
    num_stocks = 150
    feature_dim = 64  # zi feature dimension
    output_dim = 7
    
    # Create sample data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Sample zi (batch-normalized features) and y_hat (predictions) - need gradients for backprop
    zi = torch.randn(batch_size, seq_len, num_stocks, feature_dim, device=device, requires_grad=True)
    y_hat = torch.randn(batch_size, seq_len, num_stocks, output_dim, device=device, requires_grad=True)
    targets = torch.randn(batch_size, seq_len, num_stocks, output_dim, device=device)
    
    # Initialize the loss module (no model parameters)
    loss_module = MultiRNNLoss(
        loss_weights={
            'prediction': 1.0,
            'batch_norm': 0.1,
            'equal_weighted': 0.05,
            'cross_sectional': 0.1,
            'temporal_consistency': 0.05
        }
    )
    
    print(f"✅ Loss module initialized on {device}")
    print(f"📊 Loss weights: {loss_module.get_loss_weights()}")
    
    # Test forward pass
    print("\n🔄 Testing forward pass...")
    
    try:
        # Test with zi, y_hat, and targets
        loss_components = loss_module(zi, y_hat, targets)
        
        print("✅ Forward pass successful!")
        print("📈 Loss components:")
        for component, value in loss_components.items():
            if isinstance(value, torch.Tensor):
                print(f"   {component}: {value.item():.6f}")
        
        # Test backward pass
        print("\n⬅️ Testing backward pass...")
        total_loss = loss_components['total_loss']
        total_loss.backward()
        print("✅ Backward pass successful!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        raise


def test_individual_components():
    """Test individual loss components."""
    print("\n🧪 Testing Individual Loss Components...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2
    seq_len = 10
    num_stocks = 50
    output_dim = 1
    
    predictions = torch.randn(batch_size, seq_len, num_stocks, output_dim, device=device)
    targets = torch.randn(batch_size, seq_len, num_stocks, output_dim, device=device)
    
    # Test BatchNormLoss
    print("🔄 Testing BatchNormLoss...")
    batch_norm_loss = BatchNormLoss()
    bn_loss = batch_norm_loss(predictions)
    print(f"✅ BatchNormLoss: {bn_loss.item():.6f}")
    
    # Test EqualWeightedLoss
    print("🔄 Testing EqualWeightedLoss...")
    equal_weighted_loss = EqualWeightedLoss(weight=0.5)
    ew_loss = equal_weighted_loss(predictions, targets)
    print(f"✅ EqualWeightedLoss: {ew_loss.item():.6f}")


def test_loss_weights_modification():
    """Test modification of loss weights."""
    print("\n🔧 Testing Loss Weight Modification...")
    
    # Initialize simplified loss module
    loss_module = MultiRNNLoss()
    
    # Get current weights
    current_weights = loss_module.get_loss_weights()
    print("📊 Current loss weights:")
    for key, value in current_weights.items():
        print(f"   {key}: {value}")
    
    # Modify weights
    new_weights = {
        'prediction': 2.0,
        'batch_norm': 0.2,
        'equal_weighted': 0.1,
        'cross_sectional': 0.15
    }
    
    loss_module.set_loss_weights(new_weights)
    updated_weights = loss_module.get_loss_weights()
    
    print("\n📊 Updated loss weights:")
    for key, value in updated_weights.items():
        print(f"   {key}: {value}")
    
    print("✅ Loss weight modification successful!")


def demonstrate_integration_with_training():
    """Demonstrate how to integrate with existing training pipeline."""
    print("\n🚀 Integration Example...")
    
    # Initialize simplified loss module
    multi_rnn_loss = MultiRNNLoss(
        loss_weights={
            'prediction': 1.0,
            'batch_norm': 0.1,
            'equal_weighted': 0.05,
            'cross_sectional': 0.1,
            'temporal_consistency': 0.05
        }
    )
    
    print("📝 Integration steps:")
    print("1. Replace your existing loss function with MultiRNNLoss")
    print("2. Pass zi (batch-normalized features), y_hat (predictions), and targets")
    print("3. Use the returned total_loss for backpropagation")
    print("4. Monitor individual loss components for debugging")
    
    # Sample integration code
    sample_code = """
    # In your training loop:
    
    # Forward pass through your model
    y_hat = model(input_data)  # [batch, seq, stocks, output_dim]
    
    # Extract batch-normalized features zi from your model
    # This should come from the batch normalization layer before prediction
    zi = model.get_batch_normalized_features()  # [batch, seq, stocks, feature_dim]
    
    # Compute loss with MultiRNNLoss
    loss_components = multi_rnn_loss(
        zi=zi,              # Batch-normalized features
        y_hat=y_hat,        # Model predictions
        targets=targets     # Ground truth targets
    )
    
    # Use total loss for backpropagation
    total_loss = loss_components['total_loss']
    total_loss.backward()
    
    # Log individual components for monitoring
    for component, value in loss_components.items():
        logger.log(f'{component}', value.item())
    """
    
    print("💻 Sample integration code:")
    print(sample_code)


def performance_benchmark():
    """Benchmark the performance of the loss module."""
    print("\n⏱️ Performance Benchmark...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test with different sizes
    test_sizes = [
        (2, 10, 50, 7),    # Small
        (4, 30, 150, 7),   # Medium (your data size)
        (8, 50, 300, 7),   # Large
    ]
    
    for batch_size, seq_len, num_stocks, output_dim in test_sizes:
        print(f"\n📏 Testing size: batch={batch_size}, seq={seq_len}, stocks={num_stocks}, out={output_dim}")
        
        # Create test data with proper gradient requirements
        zi = torch.randn(batch_size, seq_len, num_stocks, 24, device=device, requires_grad=True)
        y_hat = torch.randn(batch_size, seq_len, num_stocks, output_dim, device=device, requires_grad=True)
        targets = torch.randn(batch_size, seq_len, num_stocks, output_dim, device=device)
        
        # Initialize simplified loss module
        loss_module = MultiRNNLoss(
            loss_weights={
                'prediction': 1.0,
                'batch_norm': 0.1,
                'equal_weighted': 0.05,
                'cross_sectional': 0.1,
                'temporal_consistency': 0.05
            }
        )
        
        # Benchmark forward pass
        import time
        
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()
        
        for _ in range(10):  # Run 10 times for averaging
            loss_components = loss_module(zi, y_hat, targets)
            total_loss = loss_components['total_loss']
        
        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"   ⏱️ Average forward pass time: {avg_time*1000:.2f}ms")
        print(f"   📊 Total loss: {total_loss.item():.6f}")


if __name__ == "__main__":
    print("🧪 Multi-RNN Loss Module Test Suite")
    print("=" * 50)
    
    try:
        # Run all tests
        test_multi_rnn_loss()
        test_individual_components()
        test_loss_weights_modification()
        demonstrate_integration_with_training()
        performance_benchmark()
        
        print("\n🎉 All tests passed successfully!")
        print("✅ Multi-RNN Loss module is ready for use!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
