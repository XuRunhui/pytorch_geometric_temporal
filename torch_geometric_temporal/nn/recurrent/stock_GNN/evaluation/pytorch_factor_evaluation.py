import torch
import numpy as np

def evaluate_factor_linear_model_pytorch(train_factors, train_returns, val_factors, val_returns, 
                                        test_factors, test_returns, value_decay=0.9, eps=1e-8):
    """
    使用PyTorch实现的因子线性模型评估（不依赖sklearn）
    
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
    """
    
    def compute_accumulative_returns(factors, returns):
        """计算累积收益率 - 直接从真实收益率计算"""
        B, N, K = factors.shape
        _, T, _ = returns.shape
        device = factors.device
        
        all_returns = []
        
        for b in range(B):
            y_b = returns[b]  # [T, N]
            
            # 计算每个股票的累积收益率
            for n in range(N):
                total_return = 0.0
                for t in range(T):
                    weight_t = value_decay ** t
                    stock_return_t = y_b[t, n].item()  # 第t天第n只股票的真实收益率
                    total_return += weight_t * stock_return_t
                
                all_returns.append(total_return)
        
        return torch.tensor(all_returns, device=factors.device)
    
    def fit_linear_model_pytorch(X, y):
        """使用PyTorch拟合线性模型 y = X * w + b"""
        # 添加偏置项
        X_with_bias = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
        
        # 最小二乘解: w = (X^T X)^(-1) X^T y
        XtX = X_with_bias.T @ X_with_bias
        XtX_inv = torch.inverse(XtX + eps * torch.eye(XtX.shape[0], device=X.device))
        weights = XtX_inv @ X_with_bias.T @ y
        
        return weights  # [K+1] 包含偏置
    
    def predict_linear_model(X, weights):
        """使用线性模型预测"""
        X_with_bias = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
        return X_with_bias @ weights
    
    def compute_r2(y_true, y_pred):
        """计算R²分数"""
        y_mean = y_true.mean()
        ss_tot = ((y_true - y_mean) ** 2).sum()
        ss_res = ((y_true - y_pred) ** 2).sum()
        return 1 - ss_res / ss_tot
    
    def compute_correlation(x, y):
        """计算相关系数"""
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        numerator = (x_centered * y_centered).sum()
        denominator = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
        return numerator / (denominator + eps)
    
    print("=== 因子线性模型评估 (PyTorch版本) ===")
    
    device = train_factors.device
    
    # 步骤1: 准备训练数据
    print("1. 准备训练数据...")
    B_train, N, K = train_factors.shape
    
    # 因子特征 [B*N, K]
    train_X = train_factors.view(-1, K)
    # 累积收益率标签 [B*N]
    train_y = compute_accumulative_returns(train_factors, train_returns)
    
    print(f"   训练特征形状: {train_X.shape}")
    print(f"   训练标签形状: {train_y.shape}")
    print(f"   训练标签统计: 均值={train_y.mean():.6f}, 标准差={train_y.std():.6f}")
    
    # 步骤2: 拟合线性模型
    print("2. 拟合线性模型...")
    model_weights = fit_linear_model_pytorch(train_X, train_y)
    
    # 训练集表现
    train_pred = predict_linear_model(train_X, model_weights)
    train_r2 = compute_r2(train_y, train_pred)
    train_mse = ((train_y - train_pred) ** 2).mean()
    
    print(f"   训练集 R²: {train_r2:.4f}")
    print(f"   训练集 MSE: {train_mse:.6f}")
    
    # 步骤3: 验证集评估
    print("3. 验证集评估...")
    val_X = val_factors.view(-1, K)
    val_y = compute_accumulative_returns(val_factors, val_returns)
    
    val_pred = predict_linear_model(val_X, model_weights)
    val_r2 = compute_r2(val_y, val_pred)
    val_mse = ((val_y - val_pred) ** 2).mean()
    val_corr = compute_correlation(val_y, val_pred)
    
    print(f"   验证集 R²: {val_r2:.4f}")
    print(f"   验证集 MSE: {val_mse:.6f}")
    print(f"   验证集相关系数: {val_corr:.4f}")
    
    # 步骤4: 测试集评估
    print("4. 测试集评估...")
    test_X = test_factors.view(-1, K)
    test_y = compute_accumulative_returns(test_factors, test_returns)
    
    test_pred = predict_linear_model(test_X, model_weights)
    test_r2 = compute_r2(test_y, test_pred)
    test_mse = ((test_y - test_pred) ** 2).mean()
    test_corr = compute_correlation(test_y, test_pred)
    
    print(f"   测试集 R²: {test_r2:.4f}")
    print(f"   测试集 MSE: {test_mse:.6f}")
    print(f"   测试集相关系数: {test_corr:.4f}")
    
    # 步骤5: 因子重要性分析
    print("5. 因子重要性分析...")
    factor_weights = model_weights[:-1]  # 除去偏置项
    bias = model_weights[-1]
    
    abs_weights = torch.abs(factor_weights)
    _, importance_ranking = torch.sort(abs_weights, descending=True)
    
    print("   前10个重要因子:")
    for i in range(min(10, len(importance_ranking))):
        idx = importance_ranking[i].item()
        weight = factor_weights[idx].item()
        abs_weight = abs_weights[idx].item()
        print(f"     因子 {idx:2d}: 权重={weight:8.4f}, 绝对值={abs_weight:8.4f}")
    
    print(f"   模型偏置: {bias:.6f}")
    
    # 返回结果
    results = {
        'train_r2': train_r2.item(),
        'train_mse': train_mse.item(),
        'val_r2': val_r2.item(),
        'val_mse': val_mse.item(),
        'val_correlation': val_corr.item(),
        'test_r2': test_r2.item(),
        'test_mse': test_mse.item(),
        'test_correlation': test_corr.item(),
        'factor_weights': factor_weights.cpu().numpy(),
        'bias': bias.item(),
        'importance_ranking': importance_ranking.cpu().numpy(),
        'model_weights': model_weights,  # 包含偏置的完整权重
        'train_predictions': train_pred.cpu().numpy(),
        'val_predictions': val_pred.cpu().numpy(),
        'test_predictions': test_pred.cpu().numpy(),
        'train_targets': train_y.cpu().numpy(),
        'val_targets': val_y.cpu().numpy(),
        'test_targets': test_y.cpu().numpy()
    }
    
    print("=== 评估完成 ===")
    return results

def print_evaluation_summary(results):
    """打印评估结果摘要"""
    print("\n" + "="*50)
    print("           评估结果摘要")
    print("="*50)
    print(f"训练集 R²:     {results['train_r2']:.4f}")
    print(f"验证集 R²:     {results['val_r2']:.4f}")
    print(f"测试集 R²:     {results['test_r2']:.4f}")
    print("-"*30)
    print(f"训练集 MSE:    {results['train_mse']:.6f}")
    print(f"验证集 MSE:    {results['val_mse']:.6f}")
    print(f"测试集 MSE:    {results['test_mse']:.6f}")
    print("-"*30)
    print(f"验证集相关系数: {results['val_correlation']:.4f}")
    print(f"测试集相关系数: {results['test_correlation']:.4f}")
    print("="*50)

# 使用示例
if __name__ == "__main__":
    # 模拟数据 (替换为你的实际数据)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    B_train, N, K = 100, 500, 32
    T = 20
    
    train_factors = torch.randn(B_train, N, K, device=device)
    train_returns = torch.randn(B_train, T, N, device=device) * 0.02
    
    val_factors = torch.randn(30, N, K, device=device)
    val_returns = torch.randn(30, T, N, device=device) * 0.02
    
    test_factors = torch.randn(50, N, K, device=device)
    test_returns = torch.randn(50, T, N, device=device) * 0.02
    
    # 运行评估
    results = evaluate_factor_linear_model_pytorch(
        train_factors, train_returns,
        val_factors, val_returns,
        test_factors, test_returns
    )
    
    # 打印摘要
    print_evaluation_summary(results)
