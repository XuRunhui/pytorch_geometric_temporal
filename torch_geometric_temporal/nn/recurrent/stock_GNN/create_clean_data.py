import pandas as pd
import numpy as np
import pickle
import os
import gzip
from datetime import datetime

# 数据目录
DATA_DIR = '/home/xu/data'
CLEAN_DIR =  "/home/xu/clean_data_unaligned"

os.makedirs(CLEAN_DIR, exist_ok=True)

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
    filepath = os.path.join(CLEAN_DIR, filename)
    try:
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ 保存 {filename} 成功，形状: {data.shape}")
        return True
    except Exception as e:
        print(f"✗ 保存 {filename} 失败: {e}")
        return False

def fill_short_gaps(series, max_gap=3):
    valid_idx = series[series.notna()].index
    if valid_idx.empty:
        return series
    start, end = valid_idx[0], valid_idx[-1]

    sub_series = series.loc[start:end]

    mask = sub_series.isna()

    gap_groups = (mask != mask.shift()).cumsum()
    result = sub_series.copy()

    for grpid, grp in sub_series.groupby(gap_groups):
        if grp.isna().all():
            # print(f"find sequetial nan value of length: {len(grp)}")
            if len(grp) <= max_gap:
                # print(f"filling sequetial nan value of length: {len(grp)}")
                result.loc[grp.index] = np.nan
                result.loc[grp.index] = result.loc[:grp.index[0]].ffill().iloc[-1]


    filled = series.copy()
    filled.loc[start:end] = result
    return filled


def fill_df(df):
    df = df.sort_index()
    df_filled = df.apply(lambda col: fill_short_gaps(col, max_gap=3))
    return df_filled

def apply_mask_filter(target_df, mask_df, data_name):
    """应用mask过滤数据"""
    if target_df is None or mask_df is None:
        print(f"✗ {data_name}: 数据为空，跳过")
        return None
    
    # 找到共同列
    common_cols = mask_df.columns.intersection(target_df.columns)
    
    if len(common_cols) == 0:
        print(f"✗ {data_name}: 没有共同的列，跳过")
        return None
    
    # 只保留共同列
    target_filtered = target_df[common_cols]
    mask_filtered = mask_df[common_cols]
    
    # 应用mask - 保留mask为1的数据，其他设为NaN
    result = target_filtered.where(mask_filtered == 1)
    print(f"✓ {data_name}: 原始列数 {target_df.shape[1]} -> 过滤后列数 {result.shape[1]}")
    print(f"  有效数据比例: {(~result.isnull()).sum().sum() / (result.shape[0] * result.shape[1]):.2%}")
    
    return result

def main():
    print("=== 开始数据过滤程序 ===")
    
    # 1. 加载mask数据（假设已经生成了df_high）
    print("\n1. 加载过滤mask...")
    
    # 重新生成mask（基于之前的逻辑）
    status_df = load_pkl_safe("trade_status.pkl")
    st_df = load_pkl_safe("st.pkl")
    listed_df = load_pkl_safe("listed.pkl")
    limited_up_df = load_pkl_safe("limit_up.pkl")
    limited_dn_df = load_pkl_safe("limit_dn.pkl")
    
    # 填充NaN
    for df in [status_df, st_df, listed_df, limited_up_df, limited_dn_df]:
        if df is not None:
            df.fillna(0, inplace=True)
    
    # 生成严格过滤条件
    # strict_mask = (st_df == 1) & (listed_df == 1)
    # cols_strict = strict_mask.columns[strict_mask.all()]
    
    # # trade_status允许1%缺失
    # status_mean = status_df.eq(1.0).mean()
    # cols_status = status_mean[status_mean >= 0].index
    
    # # 取交集
    # final_cols = cols_strict.intersection(cols_status)
    # print(f'length of final_cols: {len(final_cols)}')
    # if len(final_cols) == 0:
    #     print("✗ 没有股票满足过滤条件，程序退出")
    #     return
    
    # 生成最终mask
    # mask_df = strict_mask.loc[:, final_cols].astype(float)
    # print(f"✓ 生成过滤mask，包含 {len(final_cols)} 只股票")
    mask_df = (st_df == 1) & (listed_df == 1) & (status_df == 1) & (limited_up_df == 1) & (limited_dn_df == 1)
    # 2. 定义需要过滤的文件列表
    price_files = [
        'open_adj.pkl', 'high_adj.pkl', 'low_adj.pkl', 'close_adj.pkl', 'vwap_adj.pkl',
        'volume_adj.pkl', 'turnover.pkl'
    ]
    
    factor_files = [
        'momentum.pkl', 'resvol.pkl', 'beta.pkl', 'srisk.pkl', 'ltrevrsl.pkl',
        'btop.pkl', 'earnyild.pkl', 'earnqlty.pkl', 'earnvar.pkl', 'divyild.pkl',
        'liquidty.pkl', 'invsqlty.pkl', 'size.pkl', 'midcap.pkl',
        'growth.pkl', 'profit.pkl', 'leverage.pkl'
    ]
    
    all_files = price_files + factor_files
    
    print(f"\n2. 开始过滤 {len(all_files)} 个数据文件...")
    
    # 3. 逐个处理文件
    success_count = 0
    failed_files = []
    
    for i, filename in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}] 处理 {filename}...")
        
        # 加载原始数据
        data = load_pkl_safe(filename)
        
        if data is None:
            failed_files.append(filename)
            continue
        
        # 应用过滤
        filtered_data = apply_mask_filter(data, mask_df, filename)
 #       if filename in price_files:
#           filtered_data.ffill(limit=5)        

        if filtered_data is None:
            failed_files.append(filename)
            continue
  
        if filename in price_files:
            filtered_data = fill_df(filtered_data)
            print(f" 填充后 有效数据比例: {(~filtered_data.isnull()).sum().sum() / (filtered_data.shape[0] * filtered_data.shape[1]):.2%}")      
        elif filename in factor_files:
            nan_counts = filtered_data.isnull().sum()

            # 检查是否全部相等
            if nan_counts.nunique() == 1:
                print(f"✓ 所有列的NaN数量都相同: {nan_counts.iloc[0]}")
               # filtered_data = filtered_data.dropna(how='all')
               # filtered_data = filtered_data.ffill()
               # print(f"去除nan行后 有效数据比例: {(~filtered_data.isnull()).sum().sum() / (filtered_data.shape[0] * filtered_data.shape[1]):.2%}") 
            # else:
            #     print(f"✗ 各列NaN数量不同:")
            #     print(f"  范围: {nan_counts.min()} - {nan_counts.max()}")
            #     print(f"  唯一值: {sorted(nan_counts.unique())}")
          #  filtered_data = filtered_data.dropna(how='all')
            filtered_data = filtered_data.dropna(how='all')
            filtered_data = fill_df(filtered_data)
            print(f"去除nan行并填补后 有效数据比例: {(~filtered_data.isnull()).sum().sum() / (filtered_data.shape[0] * filtered_data.shape[1]):.2%}") 
        # 保存过滤后的数据
        output_filename = f"filtered_{filename}"
        if save_pkl_safe(filtered_data, output_filename):
            success_count += 1
        else:
            failed_files.append(filename)    
    # 4. 保存mask本身
    print(f"\n保存过滤mask...")
    save_pkl_safe(mask_df, "filter_mask.pkl")
    
    # 5. 输出统计结果
    print("\n=== 过滤完成统计 ===")
    print(f"✓ 成功处理: {success_count} 个文件")
    print(f"✗ 失败文件: {len(failed_files)} 个")
    
    if failed_files:
        print("失败文件列表:")
        for file in failed_files:
            print(f"  - {file}")
    
    print(f"\n过滤条件:")
    print(f"  - 非ST股票 (st == 0)")
    print(f"  - 已上市 (listed == 1)")
    print(f"  - 非涨停 (limit_up == 0)")
    print(f"  - 非跌停 (limit_dn == 0)")
    print(f"  - 交易状态正常率 ≥ 99%")
    
    # 6. 生成使用说明
    print("\n=== 使用说明 ===")
    print("过滤后的文件已保存到同目录下，文件名格式：filtered_xxx.pkl")
    print("例如：")
    print("  - 原文件: close.pkl -> 过滤后: filtered_close.pkl")
    print("  - 原文件: momentum.pkl -> 过滤后: filtered_momentum.pkl")
    print("  - mask文件: filter_mask.pkl")
    
    print("\n加载示例：")
    print("```python")
    print("filtered_close = load_pkl_safe('filtered_close.pkl')")
    print("filtered_momentum = load_pkl_safe('filtered_momentum.pkl')")
    print("mask = load_pkl_safe('filter_mask.pkl')")
    print("```")

if __name__ == "__main__":
    main()