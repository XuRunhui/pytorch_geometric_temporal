"""
Multi-RNN Loss Integration Example

This example demonstrates how to integrate the simplified Multi-RNN loss module
into your existing Stock GNN training pipeline.

Based on the architectural diagram showing:
- zi: batch-normalized features
- ŷi: model predictions  
- Multiple loss components with equal weighting
"""

import torch
import torch.nn.functional as F
from multi_rnn_loss import MultiRNNLoss


def example_training_loop():
    """
    Example of how to integrate Multi-RNN Loss into your training loop.
    """
    print("🚀 Multi-RNN Loss Integration Example")
    print("=" * 50)
    
    # Initialize the simplified loss module
    multi_rnn_loss = MultiRNNLoss(
        loss_weights={
            'prediction': 1.0,          # Main prediction loss
            'batch_norm': 0.1,          # Batch normalization consistency
            'equal_weighted': 0.05,     # Equal weighting factor loss
            'cross_sectional': 0.1,     # Cross-sectional consistency
            'temporal_consistency': 0.05 # Temporal consistency
        }
    )
    
    print(f"✅ Loss module initialized")
    print(f"📊 Loss weights: {multi_rnn_loss.get_loss_weights()}")
    
    # Simulate typical training data dimensions
    batch_size = 4
    sequence_length = 30
    num_stocks = 150
    feature_dim = 24
    output_dim = 7
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n📏 Data dimensions:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Number of stocks: {num_stocks}")
    print(f"   Feature dimension: {feature_dim}")
    print(f"   Output dimension: {output_dim}")
    
    # Simulate training loop
    print(f"\n🔄 Simulating training loop...")
    
    for epoch in range(3):
        print(f"\n📈 Epoch {epoch + 1}")
        
        # Simulate batch of training data
        # In real training, these would come from your DataLoader
        
        # zi: batch-normalized features extracted from your model
        # This should be the output of batch normalization layer before prediction
        zi = torch.randn(batch_size, sequence_length, num_stocks, feature_dim, 
                        device=device, requires_grad=True)
        
        # y_hat: predictions from your Stock GNN model
        y_hat = torch.randn(batch_size, sequence_length, num_stocks, output_dim,
                           device=device, requires_grad=True)
        
        # targets: ground truth targets
        targets = torch.randn(batch_size, sequence_length, num_stocks, output_dim,
                             device=device)
        
        # Forward pass through Multi-RNN Loss
        loss_components = multi_rnn_loss(zi, y_hat, targets)
        
        # Extract total loss for backpropagation
        total_loss = loss_components['total_loss']
        
        # Log individual loss components (useful for monitoring)
        print(f"   📊 Loss components:")
        for component, value in loss_components.items():
            if component != 'total_loss':
                print(f"      {component}: {value.item():.6f}")
        print(f"      TOTAL: {total_loss.item():.6f}")
        
        # Backward pass (in real training, you'd have optimizer.zero_grad() first)
        total_loss.backward()
        print(f"   ✅ Backward pass completed")


def example_model_integration():
    """
    Example of how the loss integrates with your existing Stock GNN model.
    """
    print(f"\n🏗️ Model Integration Example")
    print("=" * 40)
    
    # This is pseudo-code showing how you would modify your existing model
    integration_code = '''
class StockGNNWithMultiRNNLoss:
    def __init__(self, model_config, loss_config):
        self.model = YourStockGNNModel(model_config)
        self.multi_rnn_loss = MultiRNNLoss(loss_config['loss_weights'])
        
    def training_step(self, batch):
        # Forward pass through your model
        y_hat = self.model(batch.x, batch.edge_index)  # [batch, seq, stocks, output_dim]
        
        # Extract batch-normalized features zi from your model
        # This requires modifying your model to expose zi after batch norm
        zi = self.model.get_batch_normalized_features()  # [batch, seq, stocks, feature_dim]
        
        # Compute loss using Multi-RNN Loss
        loss_components = self.multi_rnn_loss(
            zi=zi,                    # Batch-normalized features
            y_hat=y_hat,              # Model predictions  
            targets=batch.y           # Ground truth targets
        )
        
        # Use total loss for optimization
        total_loss = loss_components['total_loss']
        
        # Log individual components for monitoring
        self.log_dict({
            f'train_{k}': v for k, v in loss_components.items()
        })
        
        return total_loss
    '''
    
    print("💻 Integration code structure:")
    print(integration_code)
    
    print("\n📝 Key integration points:")
    print("1. Extract zi (batch-normalized features) from your model")
    print("2. Pass zi, y_hat, and targets to MultiRNNLoss")
    print("3. Use total_loss for backpropagation")
    print("4. Monitor individual loss components")
    print("5. Adjust loss weights based on validation performance")


def example_loss_weight_tuning():
    """
    Example of how to tune loss weights for optimal performance.
    """
    print(f"\n⚖️ Loss Weight Tuning Example")
    print("=" * 35)
    
    # Initialize with conservative weights
    loss_module = MultiRNNLoss(
        loss_weights={
            'prediction': 1.0,          # Always keep this as baseline
            'batch_norm': 0.1,          # Start with 10% of prediction loss
            'equal_weighted': 0.05,     # Start with 5% of prediction loss
            'cross_sectional': 0.1,     # Start with 10% of prediction loss
            'temporal_consistency': 0.05 # Start with 5% of prediction loss
        }
    )
    
    print("🎯 Weight tuning strategies:")
    print("1. Start with conservative weights (small regularization terms)")
    print("2. Monitor individual loss components during training")
    print("3. Increase weights for components that help validation performance")
    print("4. Decrease weights for components that hurt convergence")
    
    # Example of adjusting weights based on training progress
    tuning_strategies = [
        {
            'name': 'Conservative (Start)',
            'weights': {'prediction': 1.0, 'batch_norm': 0.1, 'equal_weighted': 0.05, 
                       'cross_sectional': 0.1, 'temporal_consistency': 0.05}
        },
        {
            'name': 'Balanced',
            'weights': {'prediction': 1.0, 'batch_norm': 0.2, 'equal_weighted': 0.1, 
                       'cross_sectional': 0.15, 'temporal_consistency': 0.1}
        },
        {
            'name': 'High Regularization',
            'weights': {'prediction': 1.0, 'batch_norm': 0.3, 'equal_weighted': 0.2, 
                       'cross_sectional': 0.25, 'temporal_consistency': 0.2}
        }
    ]
    
    for strategy in tuning_strategies:
        print(f"\n📊 {strategy['name']} weights:")
        for component, weight in strategy['weights'].items():
            print(f"   {component}: {weight}")


if __name__ == "__main__":
    print("🧪 Multi-RNN Loss Integration Examples")
    print("=" * 60)
    
    # Run examples
    example_training_loop()
    example_model_integration() 
    example_loss_weight_tuning()
    
    print(f"\n🎉 Integration examples completed!")
    print("💡 Use these patterns to integrate Multi-RNN Loss into your Stock GNN training.")
