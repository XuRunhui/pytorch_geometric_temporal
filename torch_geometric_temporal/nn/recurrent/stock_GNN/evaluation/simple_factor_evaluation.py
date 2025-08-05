import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def evaluate_factor_linear_model(train_factors, train_returns, val_factors, val_returns, 
                                test_factors, test_returns, value_decay=0.9, eps=1e-8):
    """
    使用训练集的32个因子和累积收益率拟合线性模型，在验证集和测试集上评估
    
    Args:
        train_factors: [B, N, K] 训练集因子
        train_returns: [B, T, N] 训练集收益率
        val_factors: [B, N, K] 验证集因子
        val_returns: [B, T, N] 验证集收益率
        test_factors: [B, N, K] 测试集因子
        test_returns: [B, T, N] 测试集收益率
        value_decay: 时间衰减因子
        eps: 数值稳定性参数
        
    Returns:
        evaluation_results: 评估结果字典
        linear_model: 拟合的线性模型
    """
    
    def compute_accumulative_returns(factors, returns):
        """计算累积收益率"""
        B, N, K = factors.shape
        _, T, _ = returns.shape
        device = factors.device
        
        all_returns = []
        
        for b in range(B):
            F_b = factors[b]  # [N, K]
            y_b = returns[b]  # [T, N]
            
            # 计算pseudo-inverse
            FtF = F_b.T @ F_b
            inv_FtF = torch.inverse(FtF + eps * torch.eye(K, device=device))
            pseudo_inv = inv_FtF @ F_b.T  # [K, N]
            
            # 计算每个股票的累积收益率
            for n in range(N):
                total_return = 0.0
                for t in range(T):
                    weight_t = value_decay ** t
                    y_t = y_b[t]  # [N]
                    y_hat = F_b @ (pseudo_inv @ y_t)  # [N]
                    
                    stock_return = y_hat[n].item()
                    total_return += weight_t * stock_return
                
                all_returns.append(total_return / T)
        
        return np.array(all_returns)
    
    print("=== 因子线性模型评估 ===")
    
    # 步骤1: 准备训练数据
    print("1. 准备训练数据...")
    B_train, N, K = train_factors.shape
    
    # 因子特征 [B*N, K]
    train_X = train_factors.view(-1, K).cpu().numpy()
    # 累积收益率标签 [B*N]
    train_y = compute_accumulative_returns(train_factors, train_returns)
    
    print(f"   训练特征形状: {train_X.shape}")
    print(f"   训练标签形状: {train_y.shape}")
    print(f"   训练标签统计: 均值={train_y.mean():.6f}, 标准差={train_y.std():.6f}")
    
    # 步骤2: 拟合线性模型
    print("2. 拟合线性模型...")
    linear_model = LinearRegression(fit_intercept=True)
    linear_model.fit(train_X, train_y)
    
    # 训练集表现
    train_pred = linear_model.predict(train_X)
    train_r2 = r2_score(train_y, train_pred)
    train_mse = mean_squared_error(train_y, train_pred)
    
    print(f"   训练集 R²: {train_r2:.4f}")
    print(f"   训练集 MSE: {train_mse:.6f}")
    
    # 步骤3: 验证集评估
    print("3. 验证集评估...")
    val_X = val_factors.view(-1, K).cpu().numpy()
    val_y = compute_accumulative_returns(val_factors, val_returns)
    
    val_pred = linear_model.predict(val_X)
    val_r2 = r2_score(val_y, val_pred)
    val_mse = mean_squared_error(val_y, val_pred)
    val_corr = np.corrcoef(val_y, val_pred)[0, 1]
    
    print(f"   验证集 R²: {val_r2:.4f}")
    print(f"   验证集 MSE: {val_mse:.6f}")
    print(f"   验证集相关系数: {val_corr:.4f}")
    
    # 步骤4: 测试集评估
    print("4. 测试集评估...")
    test_X = test_factors.view(-1, K).cpu().numpy()
    test_y = compute_accumulative_returns(test_factors, test_returns)
    
    test_pred = linear_model.predict(test_X)
    test_r2 = r2_score(test_y, test_pred)
    test_mse = mean_squared_error(test_y, test_pred)
    test_corr = np.corrcoef(test_y, test_pred)[0, 1]
    
    print(f"   测试集 R²: {test_r2:.4f}")
    print(f"   测试集 MSE: {test_mse:.6f}")
    print(f"   测试集相关系数: {test_corr:.4f}")
    
    # 步骤5: 因子重要性分析
    print("5. 因子重要性分析...")
    weights = linear_model.coef_
    abs_weights = np.abs(weights)
    importance_ranking = np.argsort(abs_weights)[::-1]
    
    print("   前10个重要因子:")
    for i in range(min(10, len(importance_ranking))):
        idx = importance_ranking[i]
        print(f"     因子 {idx:2d}: 权重={weights[idx]:8.4f}, 绝对值={abs_weights[idx]:8.4f}")
    
    # 返回结果
    results = {
        'train_r2': train_r2,
        'train_mse': train_mse,
        'val_r2': val_r2,
        'val_mse': val_mse,
        'val_correlation': val_corr,
        'test_r2': test_r2,
        'test_mse': test_mse,
        'test_correlation': test_corr,
        'factor_weights': weights,
        'factor_importance_ranking': importance_ranking,
        'train_predictions': train_pred,
        'val_predictions': val_pred,
        'test_predictions': test_pred,
        'train_targets': train_y,
        'val_targets': val_y,
        'test_targets': test_y
    }
    
    print("=== 评估完成 ===")
    return results, linear_model

def plot_evaluation_results(results, save_path=None):
    """绘制评估结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 预测vs实际 - 验证集
    axes[0, 0].scatter(results['val_targets'], results['val_predictions'], alpha=0.6, s=20)
    axes[0, 0].plot([results['val_targets'].min(), results['val_targets'].max()], 
                   [results['val_targets'].min(), results['val_targets'].max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Accumulative Returns')
    axes[0, 0].set_ylabel('Predicted Accumulative Returns')
    axes[0, 0].set_title(f'Validation Set (R²={results["val_r2"]:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 预测vs实际 - 测试集
    axes[0, 1].scatter(results['test_targets'], results['test_predictions'], alpha=0.6, s=20)
    axes[0, 1].plot([results['test_targets'].min(), results['test_targets'].max()], 
                   [results['test_targets'].min(), results['test_targets'].max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual Accumulative Returns')
    axes[0, 1].set_ylabel('Predicted Accumulative Returns')
    axes[0, 1].set_title(f'Test Set (R²={results["test_r2"]:.3f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 因子权重
    weights = results['factor_weights']
    factor_indices = np.arange(len(weights))
    colors = ['red' if w < 0 else 'blue' for w in weights]
    axes[1, 0].bar(factor_indices, weights, color=colors, alpha=0.7)
    axes[1, 0].set_xlabel('Factor Index')
    axes[1, 0].set_ylabel('Weight')
    axes[1, 0].set_title('Linear Model Factor Weights')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 因子重要性排序
    abs_weights = np.abs(weights)
    sorted_indices = results['factor_importance_ranking']
    axes[1, 1].bar(range(len(weights)), abs_weights[sorted_indices], alpha=0.7)
    axes[1, 1].set_xlabel('Factor Rank (by absolute weight)')
    axes[1, 1].set_ylabel('Absolute Weight')
    axes[1, 1].set_title('Factor Importance Ranking')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 模拟数据 (替换为你的实际数据)
    B_train, N, K = 100, 500, 32
    T = 20
    
    train_factors = torch.randn(B_train, N, K)
    train_returns = torch.randn(B_train, T, N) * 0.02
    
    val_factors = torch.randn(30, N, K)
    val_returns = torch.randn(30, T, N) * 0.02
    
    test_factors = torch.randn(50, N, K)
    test_returns = torch.randn(50, T, N) * 0.02
    
    # 运行评估
    results, model = evaluate_factor_linear_model(
        train_factors, train_returns,
        val_factors, val_returns,
        test_factors, test_returns
    )
    
    # 绘制结果
    plot_evaluation_results(results, save_path='factor_evaluation_results.png')
