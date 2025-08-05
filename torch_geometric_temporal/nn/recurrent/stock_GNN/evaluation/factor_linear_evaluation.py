import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class FactorLinearEvaluator:
    """
    评估器：使用训练集的32个因子和累积收益率拟合线性模型，在验证集和测试集上评估
    """
    
    def __init__(self, value_decay: float = 0.9):
        """
        Args:
            value_decay: 时间衰减因子，用于计算累积收益率
        """
        self.value_decay = value_decay
        self.linear_model = None
        self.train_factors = None
        self.train_targets = None
        self.val_factors = None
        self.val_targets = None
        self.test_factors = None
        self.test_targets = None
        
    def compute_accumulative_returns(self, factors: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """
        计算累积收益率
        
        Args:
            factors: [B, N, K] 因子预测
            returns: [B, T, N] 收益率
            
        Returns:
            accumulative_returns: [B*N] 累积收益率
        """
        B, N, K = factors.shape
        _, T, _ = returns.shape
        device = factors.device
        
        all_accumulative_returns = []
        
        for b in range(B):
            F_b = factors[b]  # [N, K]
            y_b = returns[b]  # [T, N]
            
            # 计算pseudo-inverse
            FtF = F_b.T @ F_b
            eps = 1e-8
            inv_FtF = torch.inverse(FtF + eps * torch.eye(K, device=device))
            pseudo_inv = inv_FtF @ F_b.T  # [K, N]
            
            # 计算每个股票的累积收益率
            stock_accumulative_returns = []
            for n in range(N):
                total_return = 0.0
                for t in range(T):
                    weight_t = self.value_decay ** t
                    y_t = y_b[t]  # [N]
                    y_hat = F_b @ (pseudo_inv @ y_t)  # [N]
                    
                    # 该股票在时间t的预测收益
                    stock_return = y_hat[n].item()
                    total_return += weight_t * stock_return
                
                stock_accumulative_returns.append(total_return / T)
            
            all_accumulative_returns.extend(stock_accumulative_returns)
        
        return torch.tensor(all_accumulative_returns, device=device)
    
    def prepare_training_data(self, train_factors: torch.Tensor, train_returns: torch.Tensor):
        """
        准备训练数据：计算因子特征和累积收益率标签
        
        Args:
            train_factors: [B, N, K] 训练集因子
            train_returns: [B, T, N] 训练集收益率
        """
        print("=== 准备训练数据 ===")
        
        B, N, K = train_factors.shape
        
        # 展平因子作为特征 [B*N, K]
        self.train_factors = train_factors.view(-1, K).cpu().numpy()
        
        # 计算累积收益率作为标签 [B*N]
        accumulative_returns = self.compute_accumulative_returns(train_factors, train_returns)
        self.train_targets = accumulative_returns.cpu().numpy()
        
        print(f"训练特征形状: {self.train_factors.shape}")
        print(f"训练标签形状: {self.train_targets.shape}")
        print(f"训练标签统计: 均值={self.train_targets.mean():.6f}, 标准差={self.train_targets.std():.6f}")
        
    def fit_linear_model(self):
        """拟合线性模型"""
        print("=== 拟合线性模型 ===")
        
        if self.train_factors is None or self.train_targets is None:
            raise ValueError("请先调用 prepare_training_data() 准备训练数据")
        
        # 使用sklearn的线性回归
        self.linear_model = LinearRegression(fit_intercept=True)
        self.linear_model.fit(self.train_factors, self.train_targets)
        
        # 训练集上的表现
        train_pred = self.linear_model.predict(self.train_factors)
        train_r2 = r2_score(self.train_targets, train_pred)
        train_mse = mean_squared_error(self.train_targets, train_pred)
        
        print(f"线性模型训练完成")
        print(f"训练集 R²: {train_r2:.4f}")
        print(f"训练集 MSE: {train_mse:.6f}")
        print(f"模型权重范围: [{self.linear_model.coef_.min():.4f}, {self.linear_model.coef_.max():.4f}]")
        print(f"模型偏置: {self.linear_model.intercept_:.6f}")
        
    def prepare_evaluation_data(self, val_factors: torch.Tensor, val_returns: torch.Tensor,
                              test_factors: torch.Tensor, test_returns: torch.Tensor):
        """
        准备验证和测试数据
        
        Args:
            val_factors: [B, N, K] 验证集因子
            val_returns: [B, T, N] 验证集收益率
            test_factors: [B, N, K] 测试集因子  
            test_returns: [B, T, N] 测试集收益率
        """
        print("=== 准备评估数据 ===")
        
        # 验证集
        B_val, N_val, K = val_factors.shape
        self.val_factors = val_factors.view(-1, K).cpu().numpy()
        val_accumulative_returns = self.compute_accumulative_returns(val_factors, val_returns)
        self.val_targets = val_accumulative_returns.cpu().numpy()
        
        # 测试集
        B_test, N_test, K = test_factors.shape
        self.test_factors = test_factors.view(-1, K).cpu().numpy()
        test_accumulative_returns = self.compute_accumulative_returns(test_factors, test_returns)
        self.test_targets = test_accumulative_returns.cpu().numpy()
        
        print(f"验证集特征形状: {self.val_factors.shape}")
        print(f"验证集标签形状: {self.val_targets.shape}")
        print(f"测试集特征形状: {self.test_factors.shape}")
        print(f"测试集标签形状: {self.test_targets.shape}")
        
    def evaluate(self) -> Dict[str, float]:
        """
        评估线性模型在验证集和测试集上的表现
        
        Returns:
            评估结果字典
        """
        print("=== 评估线性模型 ===")
        
        if self.linear_model is None:
            raise ValueError("请先调用 fit_linear_model() 拟合模型")
        
        results = {}
        
        # 验证集评估
        if self.val_factors is not None and self.val_targets is not None:
            val_pred = self.linear_model.predict(self.val_factors)
            val_r2 = r2_score(self.val_targets, val_pred)
            val_mse = mean_squared_error(self.val_targets, val_pred)
            val_corr = np.corrcoef(self.val_targets, val_pred)[0, 1]
            
            results['val_r2'] = val_r2
            results['val_mse'] = val_mse
            results['val_correlation'] = val_corr
            
            print(f"验证集 R²: {val_r2:.4f}")
            print(f"验证集 MSE: {val_mse:.6f}")
            print(f"验证集相关系数: {val_corr:.4f}")
        
        # 测试集评估
        if self.test_factors is not None and self.test_targets is not None:
            test_pred = self.linear_model.predict(self.test_factors)
            test_r2 = r2_score(self.test_targets, test_pred)
            test_mse = mean_squared_error(self.test_targets, test_pred)
            test_corr = np.corrcoef(self.test_targets, test_pred)[0, 1]
            
            results['test_r2'] = test_r2
            results['test_mse'] = test_mse
            results['test_correlation'] = test_corr
            
            print(f"测试集 R²: {test_r2:.4f}")
            print(f"测试集 MSE: {test_mse:.6f}")
            print(f"测试集相关系数: {test_corr:.4f}")
        
        return results
    
    def analyze_factor_importance(self, top_k: int = 10) -> Dict[str, np.ndarray]:
        """
        分析因子重要性
        
        Args:
            top_k: 显示前k个重要因子
            
        Returns:
            因子重要性分析结果
        """
        if self.linear_model is None:
            raise ValueError("请先拟合模型")
        
        print(f"=== 因子重要性分析 (Top {top_k}) ===")
        
        # 获取因子权重
        weights = self.linear_model.coef_
        abs_weights = np.abs(weights)
        
        # 排序获取最重要的因子
        importance_indices = np.argsort(abs_weights)[::-1]
        
        print("最重要的因子:")
        for i in range(min(top_k, len(importance_indices))):
            idx = importance_indices[i]
            print(f"因子 {idx:2d}: 权重={weights[idx]:8.4f}, 绝对值={abs_weights[idx]:8.4f}")
        
        return {
            'weights': weights,
            'abs_weights': abs_weights,
            'importance_ranking': importance_indices
        }
    
    def plot_predictions_vs_actual(self, save_path: Optional[str] = None):
        """
        绘制预测值vs实际值的散点图
        
        Args:
            save_path: 保存图片的路径
        """
        if self.linear_model is None:
            raise ValueError("请先拟合模型")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 验证集
        if self.val_factors is not None:
            val_pred = self.linear_model.predict(self.val_factors)
            axes[0].scatter(self.val_targets, val_pred, alpha=0.6, s=20)
            axes[0].plot([self.val_targets.min(), self.val_targets.max()], 
                        [self.val_targets.min(), self.val_targets.max()], 'r--', lw=2)
            axes[0].set_xlabel('Actual Accumulative Returns')
            axes[0].set_ylabel('Predicted Accumulative Returns')
            axes[0].set_title('Validation Set: Predicted vs Actual')
            axes[0].grid(True, alpha=0.3)
        
        # 测试集
        if self.test_factors is not None:
            test_pred = self.linear_model.predict(self.test_factors)
            axes[1].scatter(self.test_targets, test_pred, alpha=0.6, s=20)
            axes[1].plot([self.test_targets.min(), self.test_targets.max()], 
                        [self.test_targets.min(), self.test_targets.max()], 'r--', lw=2)
            axes[1].set_xlabel('Actual Accumulative Returns')
            axes[1].set_ylabel('Predicted Accumulative Returns')
            axes[1].set_title('Test Set: Predicted vs Actual')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        
        plt.show()
    
    def plot_factor_weights(self, save_path: Optional[str] = None):
        """
        绘制因子权重分布
        
        Args:
            save_path: 保存图片的路径
        """
        if self.linear_model is None:
            raise ValueError("请先拟合模型")
        
        weights = self.linear_model.coef_
        factor_indices = np.arange(len(weights))
        
        plt.figure(figsize=(12, 6))
        
        # 权重柱状图
        plt.subplot(1, 2, 1)
        colors = ['red' if w < 0 else 'blue' for w in weights]
        plt.bar(factor_indices, weights, color=colors, alpha=0.7)
        plt.xlabel('Factor Index')
        plt.ylabel('Weight')
        plt.title('Linear Model Factor Weights')
        plt.grid(True, alpha=0.3)
        
        # 权重绝对值排序
        plt.subplot(1, 2, 2)
        abs_weights = np.abs(weights)
        sorted_indices = np.argsort(abs_weights)[::-1]
        plt.bar(range(len(weights)), abs_weights[sorted_indices], alpha=0.7)
        plt.xlabel('Factor Rank (by absolute weight)')
        plt.ylabel('Absolute Weight')
        plt.title('Factor Importance Ranking')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        
        plt.show()

def main():
    """
    使用示例
    """
    print("=== Factor Linear Model Evaluation ===")
    
    # 假设你已经有了训练好的模型预测结果
    # 这里需要替换为实际的数据加载代码
    
    # 示例数据形状 (需要替换为实际数据)
    B_train, N, K = 100, 500, 32  # 100个batch, 500只股票, 32个因子
    T = 20  # 20天预测期
    
    # 模拟数据 (实际使用时替换为真实数据)
    train_factors = torch.randn(B_train, N, K)
    train_returns = torch.randn(B_train, T, N) * 0.02  # 2%的收益率标准差
    
    val_factors = torch.randn(30, N, K)
    val_returns = torch.randn(30, T, N) * 0.02
    
    test_factors = torch.randn(50, N, K)
    test_returns = torch.randn(50, T, N) * 0.02
    
    # 创建评估器
    evaluator = FactorLinearEvaluator(value_decay=0.9)
    
    # 步骤1: 准备训练数据
    evaluator.prepare_training_data(train_factors, train_returns)
    
    # 步骤2: 拟合线性模型
    evaluator.fit_linear_model()
    
    # 步骤3: 准备评估数据
    evaluator.prepare_evaluation_data(val_factors, val_returns, test_factors, test_returns)
    
    # 步骤4: 评估模型
    results = evaluator.evaluate()
    
    # 步骤5: 分析因子重要性
    importance_analysis = evaluator.analyze_factor_importance(top_k=10)
    
    # 步骤6: 可视化结果
    # evaluator.plot_predictions_vs_actual(save_path='predictions_vs_actual.png')
    # evaluator.plot_factor_weights(save_path='factor_weights.png')
    
    print("\n=== 评估完成 ===")
    return evaluator, results

if __name__ == "__main__":
    evaluator, results = main()
