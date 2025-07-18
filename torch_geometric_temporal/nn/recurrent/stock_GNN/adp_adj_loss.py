import torch
import torch.nn as nn
from typing import Any
import scipy.stats as stats
import numpy as np  

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
            
        # 用于存储历史RankIC值，计算ICIR
        self.rank_ic_history = []
        self.abs_rank_ic_history = []

    def compute_rank_ic_and_icir(self, factors, returns):
        """
        计算RankIC和ICIR指标
        
        Args:
            factors: [N, K] 当天的因子值
            returns: [T, N, D] 未来T天的收益率，D为不同收益率类型
            
        Returns:
            rank_ic: 当天因子与未来收益率的平均RankIC
            icir: 历史RankIC序列的信息比率
        """
        factors_np = factors.detach().cpu().numpy()  # [N, K]
        returns_np = returns.detach().cpu().numpy()  # [T, N, D]
        
        T, N, D = returns_np.shape
        K = factors_np.shape[1]
        
        # 计算每个因子与未来每天收益率的RankIC
        daily_rank_ics = []
        daily_abs_rank_ics = []
        
        for t in range(T):  # 遍历未来每一天
            for d in range(D):  # 遍历每种收益率类型
                day_returns = returns_np[t, :, d]  # [N] 第t天第d种收益率
                
                # 检查是否有有效数据
                valid_mask = ~(np.isnan(day_returns) | np.any(np.isnan(factors_np), axis=1))
                if valid_mask.sum() < 10:  # 至少需要10个有效样本
                    continue
                    
                valid_returns = day_returns[valid_mask]
                valid_factors = factors_np[valid_mask]  # [N_valid, K]
                
                # 计算每个因子与收益率的RankIC
                factor_rank_ics = []
                abs_factor_rank_ics = []
                for k in range(K):
                    factor_k = valid_factors[:, k]
                    
                    # 计算Spearman秩相关系数
                    try:
                        rank_ic, _ = stats.spearmanr(factor_k, valid_returns)
                        if not np.isnan(rank_ic):
                            factor_rank_ics.append(rank_ic)
                            abs_factor_rank_ics.append(abs(rank_ic))
                    except:
                        continue
                
                if factor_rank_ics:
                    # Get the importance weight as a CPU float
                    importance_weight = self.importance_weights[d].cpu().item()
                    # 取所有因子的平均RankIC
                    daily_rank_ics.append(np.mean(factor_rank_ics) * importance_weight)
                    daily_abs_rank_ics.append(np.mean(abs_factor_rank_ics) * importance_weight)
        
        if not daily_rank_ics:
            return {'rank_ic': 0.0, 'abs_rank_ic': 0.0, 'icir': 0.0, 'abs_icir': 0.0}
            
        # 当前的平均RankIC
        current_rank_ic = np.mean(daily_rank_ics)
        current_abs_rank_ic = np.mean(daily_abs_rank_ics)
        
        # 更新历史RankIC记录
        self.rank_ic_history.append(current_rank_ic)
        self.abs_rank_ic_history.append(current_abs_rank_ic)
        
        # 保持历史记录在合理范围内（例如最近100个值）
        if len(self.rank_ic_history) > 100:
            self.rank_ic_history = self.rank_ic_history[-100:]
            self.abs_rank_ic_history = self.abs_rank_ic_history[-100:]
        
        # 计算ICIR（需要至少5个历史值）
        if len(self.rank_ic_history) >= 5:
            rank_ic_series = np.array(self.rank_ic_history)
            abs_rank_ic_series = np.array(self.abs_rank_ic_history)
            
            icir = np.mean(rank_ic_series) / (np.std(rank_ic_series) + 1e-8)
            abs_icir = np.mean(abs_rank_ic_series) / (np.std(abs_rank_ic_series) + 1e-8)
        else:
            icir = 0.0
            abs_icir = 0.0
            
        return {
            'rank_ic': current_rank_ic,
            'abs_rank_ic': current_abs_rank_ic,
            'icir': icir,
            'abs_icir': abs_icir
        }

    def compute_decile_analysis(self, predictions, returns):
        """
        按预测值将股票分成十组，计算top组和bottom组的平均收益率
        
        Args:
            predictions: [T, N, D] 线性预测值
            returns: [T, N, D] 未来T天的收益率，D为不同收益率类型
            
        Returns:
            包含top组和bottom组收益率的字典
        """
        predictions_np = predictions.detach().cpu().numpy()  # [T, N, D]
        returns_np = returns.detach().cpu().numpy()  # [T, N, D]
        
        T, N, D = returns_np.shape
        
        top_pred_returns = []
        bottom_pred_returns = []
        top_actual_returns = []
        bottom_actual_returns = []
        
        for t in range(T):  # 遍历每一天
            for d in range(D):  # 遍历每种收益率类型
                pred_values = predictions_np[t, :, d]  # [N] 第t天第d种预测值
                actual_returns = returns_np[t, :, d]  # [N] 第t天第d种收益率
                
                # 检查是否有有效数据
                valid_mask = ~(np.isnan(pred_values) | np.isnan(actual_returns))
                if valid_mask.sum() < 20:  # 至少需要20个有效样本才能分十组
                    continue
                    
                valid_preds = pred_values[valid_mask]
                valid_returns = actual_returns[valid_mask]
                
                # 按预测值排序
                sorted_indices = np.argsort(valid_preds)
                n_valid = len(valid_preds)
                
                # 计算十分位分组
                decile_size = n_valid // 10
                if decile_size == 0:
                    continue
                
                # Top组 (预测值最高的10%)
                top_indices = sorted_indices[-decile_size:]
                top_pred_returns.extend(valid_preds[top_indices])
                top_actual_returns.extend(valid_returns[top_indices])
                
                # Bottom组 (预测值最低的10%)
                bottom_indices = sorted_indices[:decile_size]
                bottom_pred_returns.extend(valid_preds[bottom_indices])
                bottom_actual_returns.extend(valid_returns[bottom_indices])
        
        return {
            'top_pred_return': np.mean(top_pred_returns) if top_pred_returns else 0.0,
            'top_actual_return': np.mean(top_actual_returns) if top_actual_returns else 0.0,
            'bottom_pred_return': np.mean(bottom_pred_returns) if bottom_pred_returns else 0.0,
            'bottom_actual_return': np.mean(bottom_actual_returns) if bottom_actual_returns else 0.0,
            'long_short_pred': (np.mean(top_pred_returns) - np.mean(bottom_pred_returns)) if (top_pred_returns and bottom_pred_returns) else 0.0,
            'long_short_actual': (np.mean(top_actual_returns) - np.mean(bottom_actual_returns)) if (top_actual_returns and bottom_actual_returns) else 0.0
        }

    def compute_linear_rank_ic_and_icir(self, predictions, returns):
        """
        计算线性预测值的RankIC和ICIR指标
        
        Args:
            predictions: [T, N, D] 线性预测值
            returns: [T, N, D] 未来T天的收益率，D为不同收益率类型
            
        Returns:
            rank_ic: 预测值与收益率的平均RankIC
            icir: 历史RankIC序列的信息比率
        """
        predictions_np = predictions.detach().cpu().numpy()  # [T, N, D]
        returns_np = returns.detach().cpu().numpy()  # [T, N, D]
        
        T, N, D = returns_np.shape
        
        # 计算每个时间和维度的RankIC
        daily_rank_ics = []
        daily_abs_rank_ics = []
        
        for t in range(T):  # 遍历每一天
            for d in range(D):  # 遍历每种收益率类型
                pred_values = predictions_np[t, :, d]  # [N] 第t天第d种预测值
                actual_returns = returns_np[t, :, d]  # [N] 第t天第d种收益率
                
                # 检查是否有有效数据
                valid_mask = ~(np.isnan(pred_values) | np.isnan(actual_returns))
                if valid_mask.sum() < 10:  # 至少需要10个有效样本
                    continue
                    
                valid_preds = pred_values[valid_mask]
                valid_returns = actual_returns[valid_mask]
                
                # 计算Spearman秩相关系数
                try:
                    rank_ic, _ = stats.spearmanr(valid_preds, valid_returns)
                    if not np.isnan(rank_ic):
                        # Get the importance weight as a CPU float
                        importance_weight = self.importance_weights[d].cpu().item()
                        daily_rank_ics.append(rank_ic * importance_weight)
                        daily_abs_rank_ics.append(abs(rank_ic) * importance_weight)
                except:
                    continue
        
        if not daily_rank_ics:
            return {'rank_ic': 0.0, 'abs_rank_ic': 0.0, 'icir': 0.0, 'abs_icir': 0.0}
            
        # 当前的平均RankIC
        current_rank_ic = np.mean(daily_rank_ics)
        current_abs_rank_ic = np.mean(daily_abs_rank_ics)
        
        # 更新历史RankIC记录
        self.rank_ic_history.append(current_rank_ic)
        self.abs_rank_ic_history.append(current_abs_rank_ic)
        
        # 保持历史记录在合理范围内（例如最近100个值）
        if len(self.rank_ic_history) > 100:
            self.rank_ic_history = self.rank_ic_history[-100:]
            self.abs_rank_ic_history = self.abs_rank_ic_history[-100:]
        
        # 计算ICIR（需要至少5个历史值）
        if len(self.rank_ic_history) >= 5:
            rank_ic_series = np.array(self.rank_ic_history)
            abs_rank_ic_series = np.array(self.abs_rank_ic_history)
            
            icir = np.mean(rank_ic_series) / (np.std(rank_ic_series) + 1e-8)
            abs_icir = np.mean(abs_rank_ic_series) / (np.std(abs_rank_ic_series) + 1e-8)
        else:
            icir = 0.0
            abs_icir = 0.0
            
        return {
            'rank_ic': current_rank_ic,
            'abs_rank_ic': current_abs_rank_ic,
            'icir': icir,
            'abs_icir': abs_icir
        }

    def forward(self, preds: torch.Tensor, y_ts: torch.Tensor, compute_metrics: bool = False) -> torch.Tensor:
        """
        Args:
            preds: [B, N, K] 模型输出因子（每个 batch 一个图）
            y_ts:  [B, T, N, D] 每个 batch 的未来 T 天收益（D 维）
            importance: [D] 各收益维度的重要性权重

        Returns:
            scalar loss
        """
        B, N, K = preds.shape
        _, T, _, D = y_ts.shape
        device = preds.device
        
        # 确保 importance 张量在正确的设备上
        self.importance_weights = self.importance_weights.to(device)

        total_loss_r2 = 0.0
        total_loss_corr = 0.0
        total_loss_mse = 0.0  # 用于记录线性规划的MSE
        
        # 用于存储线性预测因子的RankIC值
        batch_linear_rank_ics = []
        batch_linear_abs_rank_ics = []
        batch_linear_icirs = []
        batch_linear_abs_icirs = []
        
        # 用于存储十分组分析结果
        batch_decile_results = []

        for b in range(B):
            F_b = preds[b]          # [N, K]
            y_b = y_ts[b]           # [T, N, D]

            # === 计算 pseudo-inverse: (F^T F)^(-1) F^T ===
            FtF = F_b.T @ F_b
            inv_FtF = torch.inverse(FtF + self.eps * torch.eye(K, device=device))
            pseudo_inv = inv_FtF @ F_b.T     # [K, N]

            total_r2 = 0.0
            for t in range(T):
                weight_t = self.value_decay ** t
                y_t = y_b[t]       # [N, D]
                y_hat = F_b @ (pseudo_inv @ y_t)  # [N, D]

                ss_res = ((y_t - y_hat) ** 2).sum(dim=0)  # [D]
                y_mean = y_t.mean(dim=0)
                ss_tot = ((y_t - y_mean) ** 2).sum(dim=0) + self.eps  # [D]

                r2_d = 1 - ss_res / ss_tot       # [D]
                weighted_r2 = (r2_d * self.importance_weights).sum()  # scalar

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

            # === 计算线性预测的MSE和RankIC ===
            # 选择第一个时间和维度计算MSE
            y_t_selected = y_b[0, :, 0]  # 选择 T 维度第一个和 D 维度第一个 [N]
            y_hat_selected = F_b @ (pseudo_inv @ y_t_selected)  # [N]
            mse_loss = torch.mean((y_hat_selected - y_t_selected) ** 2)  # MSE loss
            total_loss_mse += mse_loss
            
            # 计算所有T和D维度的线性预测值作为因子
            if compute_metrics:
                # 计算所有T和D维度的线性预测值
                y_hat_all = torch.zeros_like(y_b)  # [T, N, D]
                for t in range(T):
                    for d in range(D):
                        y_t_d = y_b[t, :, d]  # [N]
                        y_hat_all[t, :, d] = F_b @ (pseudo_inv @ y_t_d)  # [N]
                
                # 使用compute_linear_rank_ic_and_icir函数计算线性预测的RankIC
                linear_ic_metrics = self.compute_linear_rank_ic_and_icir(y_hat_all, y_b)
                batch_linear_rank_ics.append(linear_ic_metrics['rank_ic'])
                batch_linear_abs_rank_ics.append(linear_ic_metrics['abs_rank_ic'])
                batch_linear_icirs.append(linear_ic_metrics['icir'])
                batch_linear_abs_icirs.append(linear_ic_metrics['abs_icir'])
                
                # 计算十分组分析
                decile_metrics = self.compute_decile_analysis(y_hat_all, y_b)
                batch_decile_results.append(decile_metrics)

        # 求所有 batch 平均
        mean_loss_r2 = total_loss_r2 / B
        mean_loss_mse = total_loss_mse / B
        loss = mean_loss_r2 + self.penalty_weight * (total_loss_corr / B)
        
        # 将RankIC和ICIR信息作为属性附加到loss tensor上
        # 这样外部可以直接访问而不需要重新计算
        if compute_metrics:
            # 计算十分组分析的平均值
            if batch_decile_results:
                avg_decile_metrics = {}
                for key in batch_decile_results[0].keys():
                    avg_decile_metrics[key] = np.mean([result[key] for result in batch_decile_results])
            else:
                avg_decile_metrics = {
                    'top_pred_return': 0.0,
                    'top_actual_return': 0.0,
                    'bottom_pred_return': 0.0,
                    'bottom_actual_return': 0.0,
                    'long_short_pred': 0.0,
                    'long_short_actual': 0.0
                }
            
            loss.rank_ic_info = {
                'loss_mse': float(mean_loss_mse.item()),  # 转换为Python float
                'loss_r2': float(mean_loss_r2.item()),  # 转换为Python float
                'linear_rank_ic': np.mean(batch_linear_rank_ics) if batch_linear_rank_ics else 0.0,
                'linear_abs_rank_ic': np.mean(batch_linear_abs_rank_ics) if batch_linear_abs_rank_ics else 0.0,
                'linear_icir': np.mean(batch_linear_icirs) if batch_linear_icirs else 0.0,
                'linear_abs_icir': np.mean(batch_linear_abs_icirs) if batch_linear_abs_icirs else 0.0,
                'top_pred_return': avg_decile_metrics['top_pred_return'],
                'top_actual_return': avg_decile_metrics['top_actual_return'],
                'bottom_pred_return': avg_decile_metrics['bottom_pred_return'],
                'bottom_actual_return': avg_decile_metrics['bottom_actual_return'],
                'long_short_pred': avg_decile_metrics['long_short_pred'],
                'long_short_actual': avg_decile_metrics['long_short_actual']
            }
        
        return loss
