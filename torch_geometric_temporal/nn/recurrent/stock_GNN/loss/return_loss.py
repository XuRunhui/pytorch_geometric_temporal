import torch
import torch.nn as nn
from typing import Any

class ReturnLoss(nn.Module):
    def __init__(
            self, 
            value_decay: float = 0.9, 
            penalty_weight: float = 0.1, 
            eps: float = 1e-8, 
            importance_weights: Any = [1.0, 0.0, 0.0]
        ):

        super().__init__()
        self.value_decay = value_decay
        self.penalty_weight = penalty_weight
        self.eps = eps
        # 将列表或 ListConfig 转为 Tensor
        if not isinstance(importance_weights, torch.Tensor):
            self.importance_weights = torch.tensor(importance_weights, dtype=torch.float32)
        else:
            self.importance_weights = importance_weights

    def forward(self, preds: torch.Tensor, y_ts: torch.Tensor) -> torch.Tensor:  
        """
        Args:
            preds: [B, N, T] 模型输出收益率（每个 batch 一个图，未来T期收益）
            y_ts:  [B, T, N] 每个 batch 的未来 T 天收益

        Returns:
            scalar loss
        """
        B, N, T = preds.shape
        device = preds.device
        
        # 确保 importance 张量在正确的设备上
        self.importance_weights = self.importance_weights.to(device)

        # 计算时间衰减权重: decay^(t-1) for t=1,2,...,T
        time_weights = torch.tensor([self.value_decay ** t for t in range(T)], device=device)  # [T]
        
        # 计算时间衰减的MSE loss
        # preds: [B, N, T], y_ts: [B, T, N] -> 需要对齐维度
        y_ts_aligned = y_ts.transpose(1, 2)  # [B, N, T]
        
        # 计算每个时间步的MSE，然后加权平均
        mse_per_time = torch.mean((preds - y_ts_aligned) ** 2, dim=(0, 1))  # [T]
        weighted_mse = torch.sum(mse_per_time * time_weights) / torch.sum(time_weights)
        
        # 计算时间衰减的预测收益率和实际收益率（用于分组分析）
        # 对每个样本在时间维度上加权平均
        time_weights_expanded = time_weights.view(1, 1, T)  # [1, 1, T]
        weighted_preds = torch.sum(preds * time_weights_expanded, dim=2) / torch.sum(time_weights)  # [B, N]
        weighted_targets = torch.sum(y_ts_aligned * time_weights_expanded, dim=2) / torch.sum(time_weights)  # [B, N]
        
        # 按预测收益率进行分组分析（记录top和bottom组的实际收益）
        # 将所有batch的数据flatten进行分组
        flat_preds = weighted_preds.view(-1)  # [B*N]
        flat_targets = weighted_targets.view(-1)  # [B*N]
        
        # 计算RankIC和ICIR（使用weighted_preds作为因子）
        # 过滤掉NaN值
        valid_mask = ~(torch.isnan(flat_preds) | torch.isnan(flat_targets))
        valid_preds = flat_preds[valid_mask]
        valid_targets = flat_targets[valid_mask]
        
        if len(valid_preds) > 1:
            # 计算RankIC（Spearman相关系数的近似）
            # 对预测值和目标值分别排序
            pred_ranks = torch.argsort(torch.argsort(valid_preds, descending=True), descending=True).float()
            target_ranks = torch.argsort(torch.argsort(valid_targets, descending=True), descending=True).float()
            
            # 计算Pearson相关系数（排序后的值）作为RankIC
            pred_ranks_centered = pred_ranks - pred_ranks.mean()
            target_ranks_centered = target_ranks - target_ranks.mean()
            
            numerator = torch.sum(pred_ranks_centered * target_ranks_centered)
            pred_std = torch.sqrt(torch.sum(pred_ranks_centered ** 2))
            target_std = torch.sqrt(torch.sum(target_ranks_centered ** 2))
            
            rank_ic = numerator / (pred_std * target_std + self.eps)
            
            # 存储RankIC用于计算ICIR（需要多个时间点的RankIC来计算标准差）
            if not hasattr(self, 'rank_ic_history'):
                self.rank_ic_history = []
            self.rank_ic_history.append(rank_ic.detach().cpu().item())
            
            # 保持历史记录在合理长度内（最近100个）
            if len(self.rank_ic_history) > 100:
                self.rank_ic_history = self.rank_ic_history[-100:]
            
            # 计算ICIR（RankIC的均值/标准差）
            if len(self.rank_ic_history) > 1:
                rank_ic_tensor = torch.tensor(self.rank_ic_history)
                icir = rank_ic_tensor.mean() / (rank_ic_tensor.std() + self.eps)
            else:
                icir = torch.tensor(0.0)
        else:
            rank_ic = torch.tensor(0.0)
            icir = torch.tensor(0.0)
        
        # 按预测收益率排序
        sorted_indices = torch.argsort(flat_preds, descending=True)
        n_total = len(flat_preds)
        n_group = max(1, n_total // 10)  # 取前10%和后10%
        
        # Top 10% 和 Bottom 10% 的实际收益
        top_indices = sorted_indices[:n_group]
        bottom_indices = sorted_indices[-n_group:]
        
        top_actual_returns = flat_targets[top_indices].mean()
        bottom_actual_returns = flat_targets[bottom_indices].mean()
        
        # 可选：添加分组收益差异作为额外信号（这里只是记录，不加入loss）
        group_spread = top_actual_returns - bottom_actual_returns
        
        # 存储分组信息（可用于后续分析）
        self.last_top_returns = top_actual_returns.detach()
        self.last_bottom_returns = bottom_actual_returns.detach()
        self.last_group_spread = group_spread.detach()
        self.last_rank_ic = rank_ic.detach() if isinstance(rank_ic, torch.Tensor) else torch.tensor(rank_ic)
        self.last_icir = icir.detach() if isinstance(icir, torch.Tensor) else torch.tensor(icir)

        total_loss = weighted_mse
        
        total_loss.rank_ic_info = {
            'top_returns': top_actual_returns.item(),
            'bottom_returns': bottom_actual_returns.item(),
            'group_spread': group_spread.item(),
            'rank_ic': rank_ic.item() if isinstance(rank_ic, torch.Tensor) else rank_ic,
            'icir': icir.item() if isinstance(icir, torch.Tensor) else icir
        }       
        
        return total_loss
