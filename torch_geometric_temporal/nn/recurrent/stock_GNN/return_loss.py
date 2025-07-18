import torch
import torch.nn as nn
from typing import Any, Dict, Tuple
import scipy.stats as stats
import numpy as np  

class ReturnLoss(nn.Module):
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
            
        # 用于存储历史RankIC值，计算ICIR
        self.rank_ic_history = []
        self.abs_rank_ic_history = []

    def compute_rank_ic(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """
        计算 RankIC (Rank Information Coefficient)
        
        Args:
            preds: [B, N] 预测值
            targets: [B, N] 真实值
            
        Returns:
            RankIC 值
        """
        B, N = preds.shape
        rank_ics = []
        
        for b in range(B):
            pred_b = preds[b].detach().cpu().numpy()
            target_b = targets[b].detach().cpu().numpy()
            
            # 过滤掉 NaN 值
            valid_mask = ~(np.isnan(pred_b) | np.isnan(target_b))
            if valid_mask.sum() < 3:  # 至少需要3个有效样本
                continue
                
            pred_valid = pred_b[valid_mask]
            target_valid = target_b[valid_mask]
            
            # 计算 Spearman 相关系数
            rank_ic, _ = stats.spearmanr(pred_valid, target_valid)
            if not np.isnan(rank_ic):
                rank_ics.append(rank_ic)
        
        return np.mean(rank_ics) if rank_ics else 0.0
    
    def compute_decile_analysis(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        按预测值将股票分成十组，计算top组和bottom组的平均收益率
        
        Args:
            preds: [B, N] 预测值
            targets: [B, N] 真实值
            
        Returns:
            包含top组和bottom组收益率的字典
        """
        B, N = preds.shape
        
        top_pred_returns = []
        bottom_pred_returns = []
        top_actual_returns = []
        bottom_actual_returns = []
        
        # print(f"DEBUG: Starting decile analysis with B={B}, N={N}")  # 调试信息
        
        for b in range(B):
            pred_b = preds[b].detach().cpu().numpy()
            target_b = targets[b].detach().cpu().numpy()
            
            # 过滤掉 NaN 值
            valid_mask = ~(np.isnan(pred_b) | np.isnan(target_b))
            if valid_mask.sum() < 10:  # 至少需要10个有效样本才能分组
                # print(f"DEBUG: Batch {b} has only {valid_mask.sum()} valid samples, skipping")  # 调试信息
                continue
                
            pred_valid = pred_b[valid_mask]
            target_valid = target_b[valid_mask]
            
            # 按预测值排序
            sorted_indices = np.argsort(pred_valid)
            n_valid = len(pred_valid)
            
            # 计算十分位分组
            decile_size = n_valid // 10
            if decile_size == 0:
                # print(f"DEBUG: Batch {b} decile_size is 0 (n_valid={n_valid}), skipping")  # 调试信息
                continue
            
            # print(f"DEBUG: Batch {b} processing {n_valid} valid samples with decile_size={decile_size}")  # 调试信息
            
            # Top组 (预测值最高的10%)
            top_indices = sorted_indices[-decile_size:]
            top_pred_returns.extend(pred_valid[top_indices])
            top_actual_returns.extend(target_valid[top_indices])
            
            # Bottom组 (预测值最低的10%)
            bottom_indices = sorted_indices[:decile_size]
            bottom_pred_returns.extend(pred_valid[bottom_indices])
            bottom_actual_returns.extend(target_valid[bottom_indices])
        
        return {
            'top_pred_return': np.mean(top_pred_returns) if top_pred_returns else 0.0,
            'top_actual_return': np.mean(top_actual_returns) if top_actual_returns else 0.0,
            'bottom_pred_return': np.mean(bottom_pred_returns) if bottom_pred_returns else 0.0,
            'bottom_actual_return': np.mean(bottom_actual_returns) if bottom_actual_returns else 0.0,
            'long_short_pred': (np.mean(top_pred_returns) - np.mean(bottom_pred_returns)) if (top_pred_returns and bottom_pred_returns) else 0.0,
            'long_short_actual': (np.mean(top_actual_returns) - np.mean(bottom_actual_returns)) if (top_actual_returns and bottom_actual_returns) else 0.0
        }

    def get_ic_statistics(self) -> Dict[str, float]:
        """
        获取 RankIC 的历史统计信息
        
        Returns:
            包含 RankIC 统计信息的字典
        """
        if not self.rank_ic_history:
            return {
                'rank_ic_mean': 0.0,
                'rank_ic_std': 0.0,
                'abs_rank_ic_mean': 0.0,
                'icir': 0.0,
                'rank_ic_count': 0
            }
        
        rank_ic_array = np.array(self.rank_ic_history)
        abs_rank_ic_array = np.array(self.abs_rank_ic_history)
        
        return {
            'rank_ic_mean': np.mean(rank_ic_array),
            'rank_ic_std': np.std(rank_ic_array),
            'abs_rank_ic_mean': np.mean(abs_rank_ic_array),
            'icir': np.mean(rank_ic_array) / (np.std(rank_ic_array) + self.eps),
            'rank_ic_count': len(self.rank_ic_history)
        }
    
    def reset_ic_history(self):
        """重置 RankIC 历史记录"""
        self.rank_ic_history = []
        self.abs_rank_ic_history = []


    def forward(self, inter_feature: torch.Tensor, preds: torch.Tensor, y_ts: torch.Tensor, compute_metrics: bool = False) -> torch.Tensor:  
        """
        Args:
            inter_feature: [B, N, K] 中间特征表示
            preds: [B, N, 1] 模型输出收益率（每个 batch 一个图）
            y_ts:  [B, T, N, D] 每个 batch 的未来 T 天收益（D 维）
            compute_metrics: 是否计算额外的评估指标

        Returns:
            scalar loss
        """
        B, N, _ = preds.shape
        _, T, _, D = y_ts.shape
        device = preds.device
        
        # 确保 importance 张量在正确的设备上
        self.importance_weights = self.importance_weights.to(device)
        _, _, K = inter_feature.shape

        # 初始化相关性损失
        total_loss_corr = 0.0

        # 计算信息冗余惩罚
        for b in range(B):
            F_b = inter_feature[b]  # [N, K]

            # === 计算信息冗余惩罚 corr(F_b.T) ===
            corr_mat = torch.corrcoef(F_b.T)         # [K, K]
            eye = torch.eye(K, device=device)
            off_diag = corr_mat[~eye.bool()]         # [K*K - K]
            loss_corr = (off_diag ** 2).sum()
            total_loss_corr += loss_corr

        # 只计算 T 维度上第一个和 D 维度上第一个的 MSE loss
        y_ts_selected = y_ts[:, 0, :, 0]  # [B, N] 选择 T 维度第一个和 D 维度第一个
        preds_selected = preds.squeeze(-1)  # [B, N] 去掉最后一个维度
        assert preds_selected.shape == y_ts_selected.shape, f"Shape mismatch: {preds_selected.shape} vs {y_ts_selected.shape}"
        
        # MSE loss
        mse_loss = torch.mean((preds_selected - y_ts_selected) ** 2)
        total_loss = mse_loss + self.penalty_weight * (total_loss_corr / B)

        # 计算评估指标
        if compute_metrics:
            with torch.no_grad():
                # 计算 RankIC
                rank_ic = self.compute_rank_ic(preds_selected, y_ts_selected)
                self.rank_ic_history.append(rank_ic)
                self.abs_rank_ic_history.append(abs(rank_ic))
                
                # 计算十分位分析
                decile_analysis = self.compute_decile_analysis(preds_selected, y_ts_selected)
                # print(f"DEBUG: Decile analysis results: {decile_analysis}")  # 调试信息
                
                # 计算 ICIR (Information Coefficient Information Ratio)
                icir = np.mean(self.rank_ic_history) / (np.std(self.rank_ic_history) + self.eps) if len(self.rank_ic_history) > 1 else 0.0
                
                # 将指标信息附加到 loss tensor 上
                total_loss.rank_ic_info = {
                    'rank_ic': rank_ic,
                    'abs_rank_ic': abs(rank_ic),
                    'icir': icir,
                    'rank_ic_mean': np.mean(self.rank_ic_history),
                    'abs_rank_ic_mean': np.mean(self.abs_rank_ic_history),
                    'top_pred_return': decile_analysis['top_pred_return'],
                    'top_actual_return': decile_analysis['top_actual_return'],
                    'bottom_pred_return': decile_analysis['bottom_pred_return'],
                    'bottom_actual_return': decile_analysis['bottom_actual_return'],
                    'long_short_pred': decile_analysis['long_short_pred'],
                    'long_short_actual': decile_analysis['long_short_actual'],
                    'mse_loss': mse_loss.item(),
                    'corr_penalty': (total_loss_corr / B).item()
                }
        
        return total_loss
