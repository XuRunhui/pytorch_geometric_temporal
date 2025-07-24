import pandas as pd
import numpy as np
import pickle
import os
import gzip
from datetime import datetime

# 数据目录
DATA_DIR = '/home/xu/clean_data_unaligned'

def load_pkl_safe(filename):
    """安全加载pkl文件"""
    filepath = os.path.join(DATA_DIR, filename)
    try:
        with gzip.open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"加载 {filename} 失败: {e}")
        return None

def save_pkl_safe(data, filename):
    """安全保存pkl文件"""
    filepath = os.path.join(DATA_DIR, filename)
    try:
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ 保存 {filename} 成功，形状: {data.shape}")
        return True
    except Exception as e:
        print(f"✗ 保存 {filename} 失败: {e}")
        return False

def calculate_returns(close_adj_df, periods=[1, 5, 10, 20]):
    """
    计算多期收益率
    
    Args:
        close_adj_df: 复权收盘价DataFrame，index为日期，columns为股票代码
        periods: 收益率计算期数列表，如[1, 5, 10, 20]表示1日、5日、10日、20日收益率
    
    Returns:
        dict: 各期收益率DataFrame字典，key为期数，value为收益率DataFrame
    """
    if close_adj_df is None or close_adj_df.empty:
        print("输入数据为空")
        return {}
    
    returns_dict = {}
    
    for period in periods:
        print(f"计算 {period} 日收益率...")
        
        # 计算收益率: (price_t+n / price_t) - 1
        # 使用 shift(-period) 获取未来 period 天的价格
        future_prices = close_adj_df.shift(-period)
        returns = (future_prices / close_adj_df) - 1
        
        # 移除最后 period 行（因为没有未来数据）
        returns = returns.iloc[:-period]
        
        # 统计信息
        total_data = returns.shape[0] * returns.shape[1]
        valid_data = (~returns.isnull()).sum().sum()
        valid_ratio = valid_data / total_data
        
        print(f"  {period}日收益率形状: {returns.shape}")
        print(f"  有效数据比例: {valid_ratio:.2%}")
        print(f"  收益率范围: [{returns.min().min():.4f}, {returns.max().max():.4f}]")
        
        returns_dict[period] = returns
    
    return returns_dict

def calculate_and_save_returns(close_adj_filename='close_adj.pkl', periods=[1, 5, 10, 20]):
    """
    从close_adj.pkl计算收益率并保存
    
    Args:
        close_adj_filename: 复权收盘价文件名
        periods: 收益率计算期数列表
    """
    print("=== 计算股票收益率 ===")
    
    # 1. 加载复权收盘价数据
    print(f"1. 加载 {close_adj_filename}...")
    close_adj_df = load_pkl_safe(close_adj_filename)
    
    if close_adj_df is None:
        print("无法加载复权收盘价数据")
        return False
    
    print(f"复权收盘价数据形状: {close_adj_df.shape}")
    print(f"时间范围: {close_adj_df.index.min()} 到 {close_adj_df.index.max()}")
    print(f"股票数量: {close_adj_df.shape[1]}")
    
    # 2. 计算收益率
    print(f"\n2. 计算收益率...")
    returns_dict = calculate_returns(close_adj_df, periods)
    
    if not returns_dict:
        print("收益率计算失败")
        return False
    
    # 3. 保存收益率文件
    print(f"\n3. 保存收益率文件...")
    success_count = 0
    
    for period, returns_df in returns_dict.items():
        filename = f"returns_{period}d.pkl"
        if save_pkl_safe(returns_df, filename):
            success_count += 1
        else:
            print(f"保存 {filename} 失败")
    
    # 4. 额外保存一个综合文件（可选）
    print(f"\n4. 保存综合收益率文件...")
    if save_pkl_safe(returns_dict, "returns_all.pkl"):
        print("✓ 综合收益率文件保存成功")
    
    # 5. 输出统计信息
    print(f"\n=== 收益率计算完成 ===")
    print(f"✓ 成功计算并保存 {success_count} 个收益率文件")
    print(f"保存的文件:")
    for period in periods:
        print(f"  - returns_{period}d.pkl ({period}日收益率)")
    print(f"  - returns_all.pkl (综合文件)")
    
    return True

def load_returns(period=1):
    """
    加载特定期数的收益率数据
    
    Args:
        period: 收益率期数
    
    Returns:
        DataFrame: 收益率数据
    """
    filename = f"returns_{period}d.pkl"
    return load_pkl_safe(filename)

def analyze_returns(returns_df, name="收益率"):
    """
    分析收益率数据的统计特征
    
    Args:
        returns_df: 收益率DataFrame
        name: 数据名称
    """
    if returns_df is None or returns_df.empty:
        print(f"{name} 数据为空")
        return
    
    print(f"\n=== {name} 统计分析 ===")
    print(f"数据形状: {returns_df.shape}")
    print(f"时间范围: {returns_df.index.min()} 到 {returns_df.index.max()}")
    
    # 整体统计
    returns_flat = returns_df.values.flatten()
    returns_flat = returns_flat[~np.isnan(returns_flat)]
    
    print(f"\n整体统计:")
    print(f"  有效数据量: {len(returns_flat):,}")
    print(f"  均值: {np.mean(returns_flat):.4f}")
    print(f"  标准差: {np.std(returns_flat):.4f}")
    print(f"  最小值: {np.min(returns_flat):.4f}")
    print(f"  最大值: {np.max(returns_flat):.4f}")
    
    # 分位数
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n分位数:")
    for p in percentiles:
        value = np.percentile(returns_flat, p)
        print(f"  {p:2d}%: {value:.4f}")
    
    # 正负收益率比例
    positive_ratio = (returns_flat > 0).mean()
    negative_ratio = (returns_flat < 0).mean()
    zero_ratio = (returns_flat == 0).mean()
    
    print(f"\n收益率分布:")
    print(f"  正收益率: {positive_ratio:.2%}")
    print(f"  负收益率: {negative_ratio:.2%}")
    print(f"  零收益率: {zero_ratio:.2%}")

def main():
    """主函数"""
    # 计算并保存收益率
    success = calculate_and_save_returns(
        close_adj_filename='filtered_close_adj.pkl',
        periods=[1, 5, 10, 20]  # 1日、5日、10日、20日收益率
    )
    
    if success:
        print("\n=== 收益率数据分析 ===")
        # 分析各期收益率
        for period in [1, 5, 10, 20]:
            returns_df = load_returns(period)
            analyze_returns(returns_df, f"{period}日收益率")
    
    print("\n=== 使用说明 ===")
    print("计算完成后，你可以这样使用:")
    print("```python")
    print("# 加载1日收益率")
    print("returns_1d = load_pkl_safe('returns_1d.pkl')")
    print("# 加载5日收益率")
    print("returns_5d = load_pkl_safe('returns_5d.pkl')")
    print("# 加载所有收益率")
    print("returns_all = load_pkl_safe('returns_all.pkl')")
    print("```")

if __name__ == "__main__":
    main()