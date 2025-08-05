import torch
import torch.nn as nn
from typing import Any

class AccumulativeGainLoss(nn.Module):
    def __init__(self, value_decay: float = 0.9, penalty_weight: float = 0.1, eps: float = 1e-8, importance_weights: Any = [1.0, 0.0, 0.0]):
        super().__init__()
        self.value_decay = value_decay
        self.penalty_weight = penalty_weight
        self.eps = eps
        # 将列表或 ListConfig 转为 Tensor
        if not isinstance(importance_weights, torch.Tensor):
            self.importance_weights = torch.tensor(importance_weights, dtype=torch.float32)
        else:
            self.importance_weights = importance_weights
        
        # 存储训练阶段拟合的线性模型参数
        self.fitted_models = None  # 用于存储预训练的线性模型

    def forward(self, preds: torch.Tensor, y_ts: torch.Tensor, use_pretrained_model: bool = False, compute_metrics: bool = False) -> torch.Tensor:
        """
        Args:
            preds: [B, N, K] 模型输出因子（每个 batch 一个图）
            y_ts:  [B, T, N] 每个 batch 的未来 T 天收益
            use_pretrained_model: 是否使用预训练的线性模型（验证时使用）

        Returns:
            scalar loss
        """
        B, N, K = preds.shape
        _, T, _ = y_ts.shape  # 移除了 D 维度
        device = preds.device
        
        # 确保 importance 张量在正确的设备上
        self.importance_weights = self.importance_weights.to(device)

        total_loss_r2 = 0.0
        total_loss_corr = 0.0

        for b in range(B):
            F_b = preds[b]          # [N, K]
            y_b = y_ts[b]           # [T, N] (移除了 D 维度)

            if use_pretrained_model and self.fitted_models is not None:
                # 使用预训练的线性模型进行验证
                beta = self.fitted_models[b] if b < len(self.fitted_models) else self.fitted_models[0]  # [K]
                # 对于预训练模型，我们需要重新计算预测方式
                use_beta_model = True
            else:
                # 训练阶段：用当前batch的数据拟合线性模型
                # === 计算 pseudo-inverse: (F^T F)^(-1) F^T ===
                FtF = F_b.T @ F_b
                inv_FtF = torch.inverse(FtF + self.eps * torch.eye(K, device=device))
                pseudo_inv = inv_FtF @ F_b.T     # [K, N]
                use_beta_model = False
                
                # 存储拟合的模型（用于后续验证）
                if self.fitted_models is None:
                    self.fitted_models = []
                if len(self.fitted_models) <= b:
                    self.fitted_models.append(pseudo_inv.detach().clone())
                else:
                    self.fitted_models[b] = pseudo_inv.detach().clone()

            total_r2 = 0.0
            for t in range(T):
                weight_t = self.value_decay ** t
                y_t = y_b[t]       # [N] (移除了 D 维度)
                
                if use_beta_model:
                    # 使用预训练的beta系数进行预测
                    y_hat = F_b @ beta  # [N] = [N, K] @ [K]
                else:
                    # 使用pseudo-inverse进行预测
                    y_hat = F_b @ (pseudo_inv @ y_t)  # [N]

                ss_res = ((y_t - y_hat) ** 2).sum()  # scalar
                y_mean = y_t.mean()
                ss_tot = ((y_t - y_mean) ** 2).sum() + self.eps  # scalar

                r2 = 1 - ss_res / ss_tot       # scalar
                # 由于移除了 D 维度，直接使用 r2 值
                weighted_r2 = r2  # scalar

                total_r2 += weight_t * weighted_r2

            # 最小化负的 R²
            loss_r2 = - total_r2 / T
            total_loss_r2 += loss_r2

            # === 计算信息冗余惩罚 corr(F_b.T) ===
            corr_mat = torch.corrcoef(F_b.T)         # [K, K]
            eye = torch.eye(K, device=device)
            off_diag = corr_mat[~eye.bool()]         # [K*K - K]
            loss_corr = (off_diag ** 2).sum()
            total_loss_corr += loss_corr

        # 求所有 batch 平均
        mean_loss_r2 = total_loss_r2 / B
        loss = mean_loss_r2 + self.penalty_weight * (total_loss_corr / B)
        
        return loss
    
    def fit_linear_models_from_training_data(self, train_preds: torch.Tensor, train_targets: torch.Tensor):
        """
        从训练数据中拟合线性模型，用于后续验证
        
        Args:
            train_preds: [B, N, K] 训练集的因子预测
            train_targets: [B, T, N] 训练集的收益率
        """
        B, N, K = train_preds.shape
        device = train_preds.device
        
        self.fitted_models = []
        
        for b in range(B):
            F_b = train_preds[b]  # [N, K]
            y_b = train_targets[b]  # [T, N]
            
            # 将所有时间步的数据组合起来进行拟合
            # F_expanded: [T*N, K], y_flat: [T*N]
            F_expanded = F_b.unsqueeze(0).repeat(y_b.shape[0], 1, 1).view(-1, K)  # [T*N, K]
            y_flat = y_b.view(-1)  # [T*N]
            
            # 拟合线性模型: y = F * beta
            FtF = F_expanded.T @ F_expanded
            inv_FtF = torch.inverse(FtF + self.eps * torch.eye(K, device=device))
            beta = inv_FtF @ F_expanded.T @ y_flat  # [K]
            
            self.fitted_models.append(beta.detach().clone())
    
    def clear_fitted_models(self):
        """清除已拟合的模型"""
        self.fitted_models = None
    
    def has_fitted_models(self):
        """检查是否有已拟合的模型"""
        return self.fitted_models is not None and len(self.fitted_models) > 0

# 使用示例:
# 
# # 1. 训练阶段
# loss_fn = AccumulativeGainLoss()
# train_loss = loss_fn(train_preds, train_targets, use_pretrained_model=False)
# 
# # 2. 在训练完成后，使用训练数据拟合线性模型
# loss_fn.fit_linear_models_from_training_data(train_preds, train_targets)
# 
# # 3. 验证阶段 - 使用预训练的线性模型
# val_loss = loss_fn(val_preds, val_targets, use_pretrained_model=True)
# 
# # 这样验证阶段就使用了基于训练数据拟合的线性模型，更符合实际预测场景
