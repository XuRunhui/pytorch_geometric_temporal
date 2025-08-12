"""
Multi-RNN Loss Module

This module implements a pure loss function based on the multi-RNN architecture diagram.
It calculates loss given zi (features) and ŷi (predictions) without containing model parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np


class MultiRNNLoss(nn.Module):
    """
    Multi-RNN Loss calculation module.
    
    This loss function implements loss calculations based on:
    - Batch normalization consistency
    - Equal-weighted factor processing  
    - Cross-sectional and time-series loss components
    - Multiple loss terms as shown in the architecture diagram
    
    The module does not contain trainable parameters, only loss calculation logic.
    """
    
    def __init__(
        self,
        loss_weights: Optional[Dict[str, float]] = None,
        eps: float = 1e-8
    ):
        """
        Initialize Multi-RNN Loss module.
        
        Args:
            loss_weights: Dictionary of loss component weights
            eps: Small epsilon for numerical stability
        """
        super(MultiRNNLoss, self).__init__()
        
        self.eps = eps
        
        # Default loss weights
        if loss_weights is None:
            self.loss_weights = {
                'prediction': 1.0,
                'batch_norm': 0.1,
                'equal_weighted': 0.05,
                'cross_sectional': 0.1,
                'temporal_consistency': 0.05
            }
        else:
            self.loss_weights = loss_weights
    
    def forward(self, zi: torch.Tensor, y_hat: torch.Tensor, 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass and loss computation.
        
        Args:
            zi: Features from batch normalization [batch_size, seq_len, num_stocks, feature_dim]
            y_hat: Predictions from model [batch_size, seq_len, num_stocks, output_dim] 
            targets: Target values [batch_size, seq_len, num_stocks, output_dim]
            
        Returns:
            Dictionary containing loss components and total loss
        """
        # Initialize loss components
        loss_components = {}
        
        # 1. Basic prediction loss: Loss = (1/N) * Σ(batchnorm(ci) - ŷi)²
        pred_loss = self._compute_prediction_loss(y_hat, targets)
        loss_components['prediction_loss'] = pred_loss
        
        # 2. Batch normalization consistency loss based on zi
        batch_norm_loss = self._compute_batch_norm_consistency_loss(zi)
        loss_components['batch_norm_loss'] = batch_norm_loss
        
        # 3. Equal-weighted factor loss: ci = (1/K) * Σ zik
        equal_weighted_loss = self._compute_equal_weighted_loss(zi, y_hat, targets)
        loss_components['equal_weighted_loss'] = equal_weighted_loss
        
        # 4. Cross-sectional consistency loss
        cross_sectional_loss = self._compute_cross_sectional_loss(y_hat, targets)
        loss_components['cross_sectional_loss'] = cross_sectional_loss
        
        # 5. Temporal consistency loss
        temporal_consistency_loss = self._compute_temporal_consistency_loss(y_hat, targets)
        loss_components['temporal_consistency_loss'] = temporal_consistency_loss
        
        # Compute total weighted loss
        total_loss = self._compute_weighted_loss(loss_components)
        loss_components['total_loss'] = total_loss
        
        return loss_components
    
    def _compute_prediction_loss(self, y_hat: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute basic prediction loss: (1/N) * Σ(prediction - target)²
        
        Args:
            y_hat: Predictions [batch_size, seq_len, num_stocks, output_dim]
            targets: Targets [batch_size, seq_len, num_stocks, output_dim]
            
        Returns:
            Mean squared error loss
        """
        return F.mse_loss(y_hat, targets)
    
    def _compute_batch_norm_consistency_loss(self, zi: torch.Tensor) -> torch.Tensor:
        """
        Compute batch normalization consistency loss based on zi features.
        
        This ensures that zi = batchnorm(hik) maintains proper statistical properties.
        
        Args:
            zi: Batch-normalized features [batch_size, seq_len, num_stocks, feature_dim]
            
        Returns:
            Batch normalization consistency loss
        """
        # Flatten spatial and temporal dimensions for batch statistics
        batch_size, seq_len, num_stocks, feature_dim = zi.shape
        zi_flat = zi.reshape(-1, feature_dim)  # [batch_size * seq_len * num_stocks, feature_dim]
        
        # Compute batch statistics
        batch_mean = zi_flat.mean(dim=0)  # [feature_dim]
        batch_var = zi_flat.var(dim=0, unbiased=False)  # [feature_dim]
        
        # Batch norm should have approximately zero mean and unit variance
        mean_penalty = torch.mean(torch.abs(batch_mean))
        var_penalty = torch.mean(torch.abs(batch_var - 1.0))
        
        return mean_penalty + var_penalty
    
    def _compute_equal_weighted_loss(self, zi: torch.Tensor, y_hat: torch.Tensor, 
                                   targets: torch.Tensor) -> torch.Tensor:
        """
        Compute equal-weighted factor loss: ci = (1/K) * Σ zik
        
        This implements the equal-weighted aggregation shown in the diagram.
        
        Args:
            zi: Features [batch_size, seq_len, num_stocks, feature_dim]
            y_hat: Predictions [batch_size, seq_len, num_stocks, output_dim]
            targets: Targets [batch_size, seq_len, num_stocks, output_dim]
            
        Returns:
            Equal-weighted factor loss
        """
        # Compute equal-weighted features: ci = (1/K) * Σ zik
        K = zi.shape[-1]  # number of features
        ci = zi.mean(dim=-1, keepdim=True)  # [batch_size, seq_len, num_stocks, 1]
        
        # If output_dim > 1, expand ci to match
        if y_hat.shape[-1] > 1:
            ci = ci.expand(-1, -1, -1, y_hat.shape[-1])
        
        # Loss between equal-weighted features and predictions
        equal_weighted_pred_loss = F.mse_loss(ci, y_hat)
        
        # Also compute cross-sectional equal-weighted portfolio loss
        # Equal-weighted portfolio across stocks
        portfolio_pred = y_hat.mean(dim=2, keepdim=True)  # [batch_size, seq_len, 1, output_dim]
        portfolio_target = targets.mean(dim=2, keepdim=True)
        portfolio_loss = F.mse_loss(portfolio_pred, portfolio_target)
        
        return equal_weighted_pred_loss + portfolio_loss
    
    def _compute_cross_sectional_loss(self, y_hat: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-sectional consistency loss.
        
        Ensures predictions are consistent across stocks at each time step.
        
        Args:
            y_hat: Predictions [batch_size, seq_len, num_stocks, output_dim]
            targets: Targets [batch_size, seq_len, num_stocks, output_dim]
            
        Returns:
            Cross-sectional consistency loss
        """
        # Compute cross-sectional means and variances
        pred_cross_mean = y_hat.mean(dim=2, keepdim=True)  # [batch_size, seq_len, 1, output_dim]
        target_cross_mean = targets.mean(dim=2, keepdim=True)
        
        pred_cross_var = y_hat.var(dim=2, keepdim=True, unbiased=False)
        target_cross_var = targets.var(dim=2, keepdim=True, unbiased=False)
        
        # Loss on cross-sectional means
        mean_loss = F.mse_loss(pred_cross_mean, target_cross_mean)
        
        # Loss on cross-sectional variances (scale consistency)
        var_loss = F.mse_loss(pred_cross_var, target_cross_var)
        
        return mean_loss + 0.1 * var_loss
    
    def _compute_temporal_consistency_loss(self, y_hat: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Ensures predictions are smooth over time.
        
        Args:
            y_hat: Predictions [batch_size, seq_len, num_stocks, output_dim]
            targets: Targets [batch_size, seq_len, num_stocks, output_dim]
            
        Returns:
            Temporal consistency loss
        """
        if y_hat.shape[1] < 2:  # Need at least 2 time steps
            return torch.tensor(0.0, device=y_hat.device)
        
        # Compute temporal differences
        pred_diff = y_hat[:, 1:] - y_hat[:, :-1]  # [batch_size, seq_len-1, num_stocks, output_dim]
        target_diff = targets[:, 1:] - targets[:, :-1]
        
        # Loss on temporal differences (smoothness)
        temporal_loss = F.mse_loss(pred_diff, target_diff)
        
        # Penalize large temporal jumps in predictions
        smoothness_penalty = torch.mean(torch.abs(pred_diff))
        
        return temporal_loss + 0.01 * smoothness_penalty
    
    
    def _compute_weighted_loss(self, loss_components: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted total loss."""
        total_loss = 0.0
        
        # Basic prediction loss
        if 'prediction_loss' in loss_components:
            total_loss += self.loss_weights['prediction'] * loss_components['prediction_loss']
        
        # Batch normalization loss
        if 'batch_norm_loss' in loss_components:
            total_loss += self.loss_weights['batch_norm'] * loss_components['batch_norm_loss']
        
        # Equal-weighted loss
        if 'equal_weighted_loss' in loss_components:
            total_loss += self.loss_weights['equal_weighted'] * loss_components['equal_weighted_loss']
        
        # Cross-sectional loss
        if 'cross_sectional_loss' in loss_components:
            total_loss += self.loss_weights['cross_sectional'] * loss_components['cross_sectional_loss']
        
        # Temporal consistency loss
        if 'temporal_consistency_loss' in loss_components:
            total_loss += self.loss_weights['temporal_consistency'] * loss_components['temporal_consistency_loss']
        
        return total_loss
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return self.loss_weights.copy()
    
    def set_loss_weights(self, new_weights: Dict[str, float]) -> None:
        """Set new loss weights."""
        self.loss_weights.update(new_weights)


class BatchNormLoss(nn.Module):
    """
    Batch normalization loss component for ensuring stable training dynamics.
    """
    
    def __init__(self, momentum: float = 0.1, eps: float = 1e-5):
        super(BatchNormLoss, self).__init__()
        self.momentum = momentum
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute batch normalization loss.
        
        Args:
            x: Input tensor [batch_size, *, feature_dim]
            
        Returns:
            Batch normalization loss
        """
        # Flatten all dimensions except the last one
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1])
        
        # Compute batch statistics
        batch_mean = x_flat.mean(dim=0)
        batch_var = x_flat.var(dim=0, unbiased=False)
        
        # Normalize
        x_norm = (x_flat - batch_mean) / torch.sqrt(batch_var + self.eps)
        
        # Loss encourages normalized values to have unit variance and zero mean
        mean_loss = torch.mean(torch.abs(x_norm.mean(dim=0)))
        var_loss = torch.mean(torch.abs(x_norm.var(dim=0, unbiased=False) - 1.0))
        
        return mean_loss + var_loss


class EqualWeightedLoss(nn.Module):
    """
    Equal-weighted loss component for cross-sectional consistency.
    """
    
    def __init__(self, weight: float = 1.0):
        super(EqualWeightedLoss, self).__init__()
        self.weight = weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute equal-weighted loss.
        
        Args:
            predictions: Predictions [batch_size, seq_len, num_stocks, output_dim]
            targets: Targets [batch_size, seq_len, num_stocks, output_dim]
            
        Returns:
            Equal-weighted loss
        """
        # Cross-sectional equal-weighted portfolios
        pred_portfolio = predictions.mean(dim=2, keepdim=True)  # [batch_size, seq_len, 1, output_dim]
        target_portfolio = targets.mean(dim=2, keepdim=True)
        
        portfolio_loss = F.mse_loss(pred_portfolio, target_portfolio)
        
        return self.weight * portfolio_loss
