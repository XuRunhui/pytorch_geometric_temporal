import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import pickle
import os
import gzip
from datetime import datetime

class StockDataset(Dataset):
    """股票数据集 - 支持序列级别的价格标准化"""
    def __init__(self, 
                 features: torch.Tensor,  # [T, F, N] 
                 targets: torch.Tensor,   # [T, N, H]
                 feature_names: List[str],
                 sequence_length: int = 20,
                 prediction_horizon: int = 7,
                 normalize_features: bool = True,
                 # Metadata for tracking
                 date_index: Optional[List] = None,
                 stock_names: Optional[List[str]] = None,
                 return_metadata: bool = False,
                 # 鲁棒性参数
                 outlier_clip_threshold: float = 5.0,
                 noise_level: float = 1e-6,
                 use_fallback_normalization: bool = True):
        """
        Args:
            features: 特征数据 [时间, 特征数, 股票数]
            targets: 目标数据 [时间, 股票数, 预测期数]
            feature_names: 特征名称列表
            sequence_length: 序列长度
            prediction_horizon: 预测期数
            normalize_features: 是否对特征进行序列级标准化
            date_index: 日期索引列表
            stock_names: 股票名称/代码列表
            return_metadata: 是否在__getitem__中返回元数据
            outlier_clip_threshold: 异常值裁剪阈值
            noise_level: 正则化噪声水平
            use_fallback_normalization: 是否使用回退标准化
        """
        self.features = features
        self.targets = targets
        self.feature_names = feature_names
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalize_features = normalize_features
        
        # Metadata for tracking
        self.date_index = date_index
        self.stock_names = stock_names
        self.return_metadata = return_metadata
        
        # 鲁棒性参数
        self.outlier_clip_threshold = outlier_clip_threshold
        self.noise_level = noise_level
        self.use_fallback_normalization = use_fallback_normalization
        
        # 定义价格特征和成交量特征的索引
        self.price_feature_indices = []
        self.volume_feature_indices = []
        self.close_feature_index = None
        
        for i, name in enumerate(feature_names):
            if any(price_name in name for price_name in ['open_adj', 'high_adj', 'low_adj', 'close_adj', 'vwap_adj']):
                self.price_feature_indices.append(i)
                if 'close_adj' in name:
                    self.close_feature_index = i
            elif any(vol_name in name for vol_name in ['volume_adj', 'turnover']):
                self.volume_feature_indices.append(i)
        
        # 计算有效样本数量
        self.valid_indices = self._get_valid_indices()
        
    def _get_valid_indices(self):
        """获取有效的样本索引"""
        max_time = self.features.shape[0]
        # 确保有足够的历史数据和未来数据
        valid_indices = []
        for i in range(self.sequence_length, max_time - self.prediction_horizon + 1):
            valid_indices.append(i)
        return valid_indices
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        返回只包含完整序列数据的股票
        Returns:
            如果return_metadata=False:
                features: [sequence_length, feature_dim, n_valid_stocks]
                targets: [prediction_horizon, n_valid_stocks]
            如果return_metadata=True:
                features: [sequence_length, feature_dim, n_valid_stocks]
                targets: [prediction_horizon, n_valid_stocks]
                metadata: dict包含日期、股票代码等信息
        """
        current_idx = self.valid_indices[idx]
        
        # 获取历史特征序列
        start_idx = current_idx - self.sequence_length
        end_idx = current_idx
        features = self.features[start_idx:end_idx]  # [L, F, N]
        
        # 获取未来 T 期的收益率（先获取，后面会根据有效股票进行筛选）
        end_target_idx = current_idx + self.prediction_horizon
        targets = self.targets[current_idx:end_target_idx]  # [T, N]
        
        # 找出在这个序列期间没有NaN值的股票
        valid_stock_mask = self._get_valid_stocks_for_sequence(features)
        
        if valid_stock_mask.sum() == 0:
            # 如果没有完全有效的股票，选择NaN最少的股票
            nan_counts = torch.isnan(features).sum(dim=(0, 1))  # [N] - 每只股票的NaN数量
            min_nan_count = nan_counts.min()
            valid_stock_mask = (nan_counts == min_nan_count)
            print(f"警告: 时间步{current_idx}没有完全有效的股票，选择NaN最少的股票(NaN数量={min_nan_count})")
        
        # 只保留有效股票的数据
        features = features[:, :, valid_stock_mask]  # [L, F, N_valid]
        targets = targets[:, valid_stock_mask]       # [T, N_valid]
        
        # 对targets填充NaN为0（收益率缺失视为0收益）
        targets = torch.where(torch.isnan(targets), torch.zeros_like(targets), targets)
        
        # 如果需要标准化特征，应用序列级别的标准化
        if self.normalize_features:
            features = self._normalize_sequence_features(features)
        
        # 准备元数据（如果需要）
        if self.return_metadata:
            metadata = {
                "current_idx": current_idx,
                "valid_stock_mask": valid_stock_mask,
                "n_valid_stocks": valid_stock_mask.sum().item()
            }
            
            # 添加日期信息
            if self.date_index is not None and current_idx < len(self.date_index):
                metadata["current_date"] = self.date_index[current_idx]
                # 序列的日期范围
                if start_idx >= 0:
                    metadata["sequence_start_date"] = self.date_index[start_idx]
                    metadata["sequence_end_date"] = self.date_index[end_idx-1]
                # 预测目标的日期范围
                if end_target_idx <= len(self.date_index):
                    metadata["target_start_date"] = self.date_index[current_idx]
                    if end_target_idx - 1 < len(self.date_index):
                        metadata["target_end_date"] = self.date_index[end_target_idx - 1]
            
            # 添加股票代码信息
            if self.stock_names is not None:
                valid_stock_names = [self.stock_names[i] for i in range(len(self.stock_names)) if valid_stock_mask[i]]
                metadata["valid_stock_names"] = valid_stock_names
                metadata["all_stock_names"] = self.stock_names
            
            return features, targets, metadata
        
        return features, targets
    
    def _get_valid_stocks_for_sequence(self, features: torch.Tensor) -> torch.Tensor:
        """
        找出在整个序列期间都没有NaN值的股票
        
        Args:
            features: [L, F, N] 特征序列
            
        Returns:
            valid_mask: [N] 布尔张量，True表示该股票在整个序列期间都有效
        """
        # 检查每只股票在整个序列期间是否有NaN值
        # any(dim=(0,1)) 表示在时间维度和特征维度上是否有任何NaN值
        has_nan = torch.isnan(features).any(dim=(0, 1))  # [N]
        valid_mask = ~has_nan  # 取反，True表示没有NaN值
        
        return valid_mask
    
    def _normalize_sequence_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        改进的序列特征标准化 - 处理包含NaN的数据
        - 价格特征：使用序列中位数或加权平均作为标准化基准（忽略NaN）
        - 成交量特征：使用序列均值作为标准化基准（忽略NaN）
        - 因子特征：不标准化（已预处理）
        
        Args:
            features: [L, F, N] 序列特征（可能包含NaN）
            
        Returns:
            normalized_features: [L, F, N] 标准化后的序列特征
        """
        L, F, N = features.shape
        normalized_features = features.clone()
        
        # 对每只股票分别进行标准化
        for stock_idx in range(N):
            stock_features = features[:, :, stock_idx]  # [L, F]
            
            # 价格特征标准化：使用序列中位数或加权平均（减少异常值影响，忽略NaN）
            if self.close_feature_index is not None and len(self.price_feature_indices) > 0:
                close_prices = stock_features[:, self.close_feature_index]  # 整个序列的收盘价
                
                # 移除NaN值后计算统计量
                valid_close_prices = close_prices[~torch.isnan(close_prices)]
                
                if len(valid_close_prices) > 0:
                    # 使用最近有效价格的均值作为标准化基准
                    recent_days = min(5, len(valid_close_prices))
                    base_price = valid_close_prices[-recent_days:].mean()
                    
                    # 备选：如果均值无效，使用中位数
                    if torch.isnan(base_price) or torch.isinf(base_price) or base_price == 0:
                        base_price = valid_close_prices.median()
                    
                    # 最终备选：使用最后一个有效价格
                    if torch.isnan(base_price) or torch.isinf(base_price) or base_price == 0:
                        base_price = valid_close_prices[-1]
                    
                    # 应用标准化（只对非NaN值进行标准化）
                    if base_price != 0 and not torch.isnan(base_price) and not torch.isinf(base_price):
                        for price_idx in self.price_feature_indices:
                            price_series = stock_features[:, price_idx]
                            # 只标准化非NaN值
                            valid_mask = ~torch.isnan(price_series)
                            normalized_features[valid_mask, price_idx, stock_idx] = price_series[valid_mask] / base_price
            
            # 成交量特征标准化：使用序列均值（更稳定，忽略NaN）
            for vol_idx in self.volume_feature_indices:
                volume_series = stock_features[:, vol_idx]  # 整个序列的成交量
                
                # 移除NaN值后计算统计量
                valid_volume = volume_series[~torch.isnan(volume_series)]
                
                if len(valid_volume) > 0:
                    # 使用序列均值作为标准化基准
                    base_volume = valid_volume.mean()
                    
                    # 备选：如果均值无效，使用中位数
                    if torch.isnan(base_volume) or torch.isinf(base_volume) or base_volume == 0:
                        base_volume = valid_volume.median()
                    
                    # 最终备选：使用最后一个有效成交量
                    if torch.isnan(base_volume) or torch.isinf(base_volume) or base_volume == 0:
                        base_volume = valid_volume[-1]
                    
                    # 应用标准化（只对非NaN值进行标准化）
                    if base_volume != 0 and not torch.isnan(base_volume) and not torch.isinf(base_volume):
                        valid_mask = ~torch.isnan(volume_series)
                        normalized_features[valid_mask, vol_idx, stock_idx] = volume_series[valid_mask] / base_volume
        
        return normalized_features

class StockDataModule(pl.LightningDataModule):
    """股票数据模块"""
    
    def __init__(self,
                 data_dir: str = '/home/xu/clean_data_unaligned',
                 use_factors: bool = True,
                 sequence_length: int = 20,
                 prediction_horizons: List[int] = [1, 5, 10, 20],
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 normalize_features: bool = True,
                 normalize_targets: bool = True,
                 # 新增鲁棒性控制参数
                 outlier_clip_threshold: float = 5.0,      # 异常值裁剪阈值
                 noise_level: float = 1e-6,                # 噪声水平
                 use_fallback_normalization: bool = True,  # 是否使用回退标准化
                 cross_section_window_size: int = 20,      # 截面标准化滚动窗口大小
                 cross_section_decay_factor: float = 0.99, # 截面标准化衰减因子
                 min_std_threshold: float = 0.01,          # 最小标准差阈值
                 debug: bool = False):
        """
        Args:
            data_dir: 数据目录
            use_factors: 是否使用因子数据
            sequence_length: 序列长度
            prediction_horizons: 预测期数列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            batch_size: 批次大小
            num_workers: 数据加载进程数
            normalize_features: 是否对特征进行标准化
            normalize_targets: 是否对目标进行标准化
            outlier_clip_threshold: 异常值裁剪阈值（标准差倍数）
            noise_level: 添加的正则化噪声水平
            use_fallback_normalization: 当序列标准化失败时是否使用回退方法
            cross_section_window_size: 截面标准化的滚动窗口大小
            cross_section_decay_factor: 截面标准化的衰减因子
            min_std_threshold: 最小标准差阈值（避免除零）
            debug: 是否开启调试模式
        """
        super().__init__()
        self.data_dir = data_dir
        self.use_factors = use_factors
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize_features = normalize_features
        self.normalize_targets = normalize_targets
        self.debug = debug  # 添加调试选项
        
        # 鲁棒性控制参数
        self.outlier_clip_threshold = outlier_clip_threshold
        self.noise_level = noise_level
        self.use_fallback_normalization = use_fallback_normalization
        self.cross_section_window_size = cross_section_window_size
        self.cross_section_decay_factor = cross_section_decay_factor
        self.min_std_threshold = min_std_threshold
        
        # 数据文件定义 - 只使用复权调整的价格数据
        price_files = [
            'open_adj.pkl', 'high_adj.pkl', 'low_adj.pkl', 'close_adj.pkl', 'vwap_adj.pkl',
            'volume_adj.pkl', 'turnover.pkl'
        ]
        self.price_files = [f"filtered_{filename}" for filename in price_files]
        
        # 定义价格特征和成交量特征
        self.price_features = ['filtered_open_adj', 'filtered_high_adj', 'filtered_low_adj', 
                              'filtered_close_adj', 'filtered_vwap_adj']
        self.volume_features = ['filtered_volume_adj', 'filtered_turnover']
        
        factor_files = [
            'momentum.pkl', 'resvol.pkl', 'beta.pkl', 'srisk.pkl', 'ltrevrsl.pkl',
            'btop.pkl', 'earnyild.pkl', 'earnqlty.pkl', 'earnvar.pkl', 'divyild.pkl',
            'liquidty.pkl', 'invsqlty.pkl', 'size.pkl', 'midcap.pkl',
            'growth.pkl', 'profit.pkl', 'leverage.pkl'
        ]
        self.factor_files = [f"filtered_{filename}" for filename in factor_files]
        # 收益率文件
        self.return_files = [f'returns_{h}d.pkl' for h in prediction_horizons]
        
        # 数据容器
        self.features = None
        self.targets = None
        self.feature_names = None
        self.stock_names = None
        self.date_index = None
        
        # 标准化统计信息
        self.feature_stats = None  # 特征的均值和标准差
        self.target_stats = None   # 目标的均值和标准差
        
        # 数据集
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def _debug_print(self, *args, **kwargs):
        """调试打印方法 - 只在debug模式下打印"""
        if self.debug:
            print(*args, **kwargs)
    
    def _print(self, *args, **kwargs):
        """普通打印方法 - 总是打印"""
        print(*args, **kwargs)
    
    def load_pkl_safe(self, filename: str) -> Optional[pd.DataFrame]:
        """安全加载pkl文件"""
        filepath = os.path.join(self.data_dir, filename)
        try:
            with gzip.open(filepath, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"加载 {filename} 失败: {e}")
            return None
    
    def prepare_data(self):
        """准备数据（下载、预处理等）"""
        self._print("=== 准备股票数据 ===")
        
        # 检查必要文件是否存在
        required_files = self.price_files + self.return_files
        if self.use_factors:
            required_files += self.factor_files
        
        missing_files = []
        for file in required_files:
            filepath = os.path.join(self.data_dir, file)
            if not os.path.exists(filepath):
                missing_files.append(file)
        
        if missing_files:
            self._print(f"缺少文件: {missing_files}")
            raise FileNotFoundError(f"缺少必要的数据文件: {missing_files}")
        
        self._print("✓ 所有必要文件都存在")
    
    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        if stage == 'fit' or stage is None:
            self._load_and_process_data()
            self._split_data()
        
        if stage == 'test' or stage is None:
            if self.features is None:
                self._load_and_process_data()
                self._split_data()
    
    def _load_and_process_data(self):
        """加载并处理数据 - 新的标准化策略"""
        self._print("=== 加载数据 ===")
        
        # 1. 加载价格数据
        self._print("1. 加载价格数据...")
        price_data = {}
        for file in self.price_files:
            data = self.load_pkl_safe(file)
            if data is not None:
                price_data[file.replace('.pkl', '')] = data
                self._print(f"  ✓ {file}: {data.shape}")
            else:
                self._print(f"  ✗ {file}: 加载失败")
        
        # 2. 加载因子数据（如果使用）
        factor_data = {}
        if self.use_factors:
            self._print("2. 加载因子数据...")
            for file in self.factor_files:
                data = self.load_pkl_safe(file)
                if data is not None:
                    factor_data[file.replace('.pkl', '')] = data
                    self._print(f"  ✓ {file}: {data.shape}")
                else:
                    self._print(f"  ✗ {file}: 加载失败")
        
        # 3. 加载收益率数据
        self._print("3. 加载收益率数据...")
        return_data = {}
        for file in self.return_files:
            data = self.load_pkl_safe(file)
            if data is not None:
                return_data[file.replace('.pkl', '')] = data
                self._print(f"  ✓ {file}: {data.shape}")
            else:
                self._print(f"  ✗ {file}: 加载失败")
        
        # 4. 数据对齐和组织（不进行预标准化）
        self._print("4. 数据对齐和组织...")
        self._align_and_organize_data(price_data, factor_data, return_data)
    
    def _align_and_organize_data(self, price_data: Dict, factor_data: Dict, return_data: Dict):
        """对齐数据并组织，不进行预标准化"""
        
        # === 1. 数据对齐 ===
        self._debug_print("=== 1. 数据对齐 ===")
        
        # 找到公共的股票和时间
        all_data = {**price_data, **factor_data, **return_data}
        
        # 获取公共股票
        common_stocks = None
        for name, data in all_data.items():
            if data is not None and not data.empty:
                if common_stocks is None:
                    common_stocks = set(data.columns)
                else:
                    common_stocks = common_stocks.intersection(set(data.columns))
        
        if common_stocks is None or len(common_stocks) == 0:
            raise ValueError("没有找到公共股票")
        
        common_stocks = sorted(list(common_stocks))
        self._print(f"  公共股票数量: {len(common_stocks)}")
        
        # 获取公共时间范围
        if self.use_factors and factor_data:
            # 如果使用因子数据，以因子数据的时间范围为准
            common_dates = None
            for name, data in factor_data.items():
                if data is not None and not data.empty:
                    if common_dates is None:
                        common_dates = set(data.index)
                    else:
                        common_dates = common_dates.intersection(set(data.index))
            
            if common_dates is None:
                raise ValueError("因子数据没有公共时间")
            
            # 检查其他数据源的覆盖率
            all_available_dates = set(common_dates)
            for name, data in {**price_data, **return_data}.items():
                if data is not None:
                    available_dates = set(data.index)
                    all_available_dates = all_available_dates.intersection(available_dates)
            
            common_dates = sorted(list(all_available_dates))
        else:
            # 如果不使用因子数据，以价格数据的时间范围为准
            common_dates = None
            for name, data in {**price_data, **return_data}.items():
                if data is not None and not data.empty:
                    if common_dates is None:
                        common_dates = set(data.index)
                    else:
                        common_dates = common_dates.intersection(set(data.index))
            
            common_dates = sorted(list(common_dates))
        
        self._debug_print(f"  最终时间范围: {min(common_dates)} 到 {max(common_dates)} (共{len(common_dates)}天)")
        
        # === 2. 时间划分 ===
        self._debug_print("=== 2. 时间划分 ===")
        
        total_time = len(common_dates)
        train_size = int(total_time * self.train_ratio)
        val_size = int(total_time * self.val_ratio)
        
        train_end = train_size
        val_end = train_size + val_size
        
        train_dates = common_dates[:train_end]
        val_dates = common_dates[train_end:val_end]
        test_dates = common_dates[val_end:]
        
        self._print(f"总时间步数: {total_time}")
        self._debug_print(f"训练集: {train_dates[0]} 到 {train_dates[-1]} ({len(train_dates)} 天)")
        self._debug_print(f"验证集: {val_dates[0]} 到 {val_dates[-1]} ({len(val_dates)} 天)")
        self._debug_print(f"测试集: {test_dates[0]} 到 {test_dates[-1]} ({len(test_dates)} 天)")
        
        # === 3. 组织特征数据（不标准化）===
        self._debug_print("=== 3. 组织特征数据 ===")
        
        feature_list = []
        feature_names = []
        
        # 处理价格特征（不填充，保留NaN值）
        for name, data in price_data.items():
            if data is not None:
                self._debug_print(f"  处理价格特征: {name}")
                # 对齐数据
                aligned_data = data.reindex(index=common_dates, columns=common_stocks)
                # 不填充，保留原始NaN值
                feature_list.append(aligned_data.values)  # [T, N]
                feature_names.append(name)
        
        # 处理因子特征（如果使用，可以选择性填充或不填充）
        if self.use_factors:
            for name, data in factor_data.items():
                if data is not None:
                    self._debug_print(f"  处理因子特征: {name}")
                    # 对齐数据
                    aligned_data = data.reindex(index=common_dates, columns=common_stocks)
                    # 因子数据可以选择填充（因为通常已经处理过）
                    # 这里可以根据需要选择是否填充
                    filled_data = aligned_data
                    
                    feature_list.append(filled_data.values)  # [T, N]
                    feature_names.append(name)
        
        # === 4. 处理目标数据（截面标准化，保留NaN值）===
        self._debug_print("=== 4. 处理目标数据 ===")
        
        target_list = []
        for horizon in self.prediction_horizons:
            return_key = f'returns_{horizon}d'
            if return_key in return_data and return_data[return_key] is not None:
                self._debug_print(f"  处理目标: {return_key}")
                # 对齐数据
                aligned_data = return_data[return_key].reindex(index=common_dates, columns=common_stocks)
                
                # 不填充NaN值，保留原始NaN
                raw_data = aligned_data
                
                # 截面标准化：每天对所有股票做标准化（忽略NaN值）
                if self.normalize_targets:
                    normalized_data = self._cross_section_normalize(
                        raw_data, train_dates,
                        window_size=self.cross_section_window_size,
                        decay_factor=self.cross_section_decay_factor,
                        min_std=self.min_std_threshold
                    )
                else:
                    normalized_data = raw_data
                
                target_list.append(normalized_data.values)  # [T, N]
        
        # === 5. 构建最终数据 ===
        self._debug_print("=== 5. 构建最终数据 ===")
        
        # 检查是否有有效数据
        if len(feature_list) == 0:
            raise ValueError("没有有效的特征数据")
        if len(target_list) == 0:
            raise ValueError("没有有效的目标数据")
        
        # Stack features: [T, F, N]
        self.features = torch.tensor(np.stack(feature_list, axis=1), dtype=torch.float32)
        # Stack targets: [T, N, H]
        self.targets = torch.tensor(np.stack(target_list, axis=2), dtype=torch.float32)
        
        self.feature_names = feature_names
        self.stock_names = common_stocks
        self.date_index = common_dates
        
        self._print(f"  特征矩阵形状: {self.features.shape}")
        self._print(f"  目标矩阵形状: {self.targets.shape}")
        self._debug_print(f"  特征名称: {self.feature_names}")
        self._debug_print(f"  预测期数: {self.prediction_horizons}")
        
        # 数据质量检查
        self._debug_print("=== 6. 数据质量检查 ===")
        feature_nan_count = torch.isnan(self.features).sum().item()
        target_nan_count = torch.isnan(self.targets).sum().item()
        self._debug_print(f"  特征数据NaN数量: {feature_nan_count}")
        self._debug_print(f"  目标数据NaN数量: {target_nan_count}")
        
        if feature_nan_count > 0 or target_nan_count > 0:
            self._print("  警告: 数据中仍有NaN值，可能影响训练")

    def _cross_section_normalize(self, df: pd.DataFrame, train_dates: List, 
                               window_size: int = None, decay_factor: float = None, 
                               min_std: float = None) -> pd.DataFrame:
        """
        改进的截面标准化：每天对所有股票做标准化 - 减少过拟合风险
        使用滚动窗口统计量来提高稳定性
        注意：忽略NaN值进行统计计算
        
        Args:
            df: 要标准化的DataFrame [时间, 股票] (包含NaN值)
            train_dates: 训练集日期列表
            window_size: 滚动窗口大小，默认使用self.cross_section_window_size
            decay_factor: 衰减因子，默认使用self.cross_section_decay_factor  
            min_std: 最小标准差，默认使用self.min_std_threshold
            
        Returns:
            标准化后的DataFrame（NaN值保持为NaN）
        """
        # 使用类参数作为默认值
        if window_size is None:
            window_size = getattr(self, 'cross_section_window_size', 20)
        if decay_factor is None:
            decay_factor = getattr(self, 'cross_section_decay_factor', 0.99)
        if min_std is None:
            min_std = getattr(self, 'min_std_threshold', 0.01)
            
        self._debug_print(f"    改进的截面标准化收益率数据（忽略NaN值）")
        
        # 保留原始数据，不填充NaN
        raw_df = df.copy()
        
        # 2. 计算训练集的滚动截面统计量（忽略NaN值）
        train_data = raw_df.loc[train_dates]
        
        # 使用滚动窗口计算统计量（窗口大小为20天）
        window_size = min(window_size, len(train_dates) // 4)  # 至少用1/4的训练数据作为窗口
        
        # 每天计算所有股票的均值和标准差（忽略NaN值）
        daily_mean = train_data.mean(axis=1, skipna=True)  # 每天的均值，忽略NaN
        daily_std = train_data.std(axis=1, skipna=True)    # 每天的标准差，忽略NaN
        
        # 使用滚动窗口平滑统计量
        daily_mean_smooth = daily_mean.rolling(window=window_size, min_periods=1).mean()
        daily_std_smooth = daily_std.rolling(window=window_size, min_periods=1).mean()
        
        # 避免除零 - 使用更保守的最小标准差
        daily_std_smooth = daily_std_smooth.clip(lower=min_std)
        daily_std_smooth = daily_std_smooth.fillna(min_std)
        
        self._debug_print(f"      训练集日均收益率统计: 均值范围=[{daily_mean_smooth.min():.6f}, {daily_mean_smooth.max():.6f}]")
        self._debug_print(f"      训练集日收益率波动统计: 标准差范围=[{daily_std_smooth.min():.6f}, {daily_std_smooth.max():.6f}]")
        
        # 3. 对整个数据集应用截面标准化 - 改进版本，避免数据泄露
        # 先处理训练集
        train_normalized = raw_df.loc[train_dates].sub(daily_mean_smooth, axis=0).div(daily_std_smooth, axis=0)
        
        # 创建结果DataFrame
        normalized_df = raw_df.copy()
        normalized_df.loc[train_dates] = train_normalized
        
        # 对验证集和测试集逐日处理（避免未来信息泄露）
        train_end_date = train_dates[-1]
        
        # 保存最后的训练集统计量作为初始值
        last_train_mean = daily_mean_smooth.iloc[-1]
        last_train_std = daily_std_smooth.iloc[-1]
        
        # 创建滚动统计量更新器
        rolling_mean = last_train_mean
        rolling_std = last_train_std
        
        # 处理训练集之后的每一天
        for i, date in enumerate(raw_df.index):
            if date > train_end_date:
                # 获取当前日期的原始数据
                current_data = raw_df.loc[date]
                
                # 计算当前日期的实际统计量（只使用当天数据，忽略NaN）
                current_mean = current_data.mean(skipna=True)
                current_std = max(current_data.std(skipna=True), min_std)
                
                # 如果当前统计量无效（全为NaN），使用历史统计量
                if pd.isna(current_mean) or pd.isna(current_std):
                    current_mean = rolling_mean
                    current_std = rolling_std
                
                # 指数衰减更新滚动统计量
                days_after = i - len(train_dates) + 1
                decay = decay_factor ** min(days_after, 10)  # 限制衰减深度
                
                rolling_mean = decay * rolling_mean + (1 - decay) * current_mean
                rolling_std = decay * rolling_std + (1 - decay) * current_std
                
                # 应用标准化（NaN值保持为NaN）
                normalized_df.loc[date] = (current_data - rolling_mean) / rolling_std
                
                # 调试信息：记录关键统计量的变化
                if i % 50 == 0:  # 每50天打印一次
                    self._debug_print(f"        日期 {date}: rolling_mean={rolling_mean:.6f}, rolling_std={rolling_std:.6f}, decay={decay:.4f}")
        
        self._debug_print(f"    验证集标准化完成，最终统计量: mean={rolling_mean:.6f}, std={rolling_std:.6f}")
        
        # 限制极端值（避免异常值影响），但保持NaN值为NaN
        normalized_df = normalized_df.clip(-5, 5)  # 限制在[-5, 5]范围内，NaN保持为NaN
        
        # 4. 验证标准化效果
        train_normalized = normalized_df.loc[train_dates]
        train_daily_mean = train_normalized.mean(axis=1, skipna=True).mean(skipna=True)  # 忽略NaN计算均值
        train_daily_std = train_normalized.std(axis=1, skipna=True).mean(skipna=True)   # 忽略NaN计算标准差
        
        self._debug_print(f"      标准化后训练集统计: 日均值={train_daily_mean:.6f}, 日标准差={train_daily_std:.6f}")
        
        # 检查NaN值
        nan_count = normalized_df.isna().sum().sum()
        original_nan_count = raw_df.isna().sum().sum()
        self._debug_print(f"      原始NaN数量: {original_nan_count}, 标准化后NaN数量: {nan_count}")
        
        return normalized_df

    def _split_data(self):
        """按时间划分数据集"""
        self._print("=== 划分数据集 ===")
        
        total_time = self.features.shape[0]
        train_size = int(total_time * self.train_ratio)
        val_size = int(total_time * self.val_ratio)
        
        train_end = train_size
        val_end = train_size + val_size
        
        self._print(f"总时间步数: {total_time}")
        self._print(f"训练集: 0 - {train_end} ({train_end} 步)")
        self._print(f"验证集: {train_end} - {val_end} ({val_end - train_end} 步)")
        self._print(f"测试集: {val_end} - {total_time} ({total_time - val_end} 步)")
        
        # 创建数据集，传递feature_names和normalize_features参数
        self.train_dataset = StockDataset(
            features=self.features[:train_end],
            targets=self.targets[:train_end],
            feature_names=self.feature_names,
            sequence_length=self.sequence_length,
            normalize_features=self.normalize_features,
            date_index=self.date_index[:train_end] if self.date_index else None,
            stock_names=self.stock_names,
            return_metadata=False,  # Training doesn't need metadata
            outlier_clip_threshold=self.outlier_clip_threshold,
            noise_level=self.noise_level,
            use_fallback_normalization=self.use_fallback_normalization
        )
        
        self.val_dataset = StockDataset(
            features=self.features[train_end:val_end],
            targets=self.targets[train_end:val_end],
            feature_names=self.feature_names,
            sequence_length=self.sequence_length,
            normalize_features=self.normalize_features,
            date_index=self.date_index[train_end:val_end] if self.date_index else None,
            stock_names=self.stock_names,
            return_metadata=False,  # Validation doesn't need metadata
            outlier_clip_threshold=self.outlier_clip_threshold,
            noise_level=self.noise_level,
            use_fallback_normalization=self.use_fallback_normalization
        )
        
        self.test_dataset = StockDataset(
            features=self.features[val_end:],
            targets=self.targets[val_end:],
            feature_names=self.feature_names,
            sequence_length=self.sequence_length,
            normalize_features=self.normalize_features,
            date_index=self.date_index[val_end:] if self.date_index else None,
            stock_names=self.stock_names,
            return_metadata=True,  # Test dataset needs metadata for prediction tracking
            outlier_clip_threshold=self.outlier_clip_threshold,
            noise_level=self.noise_level,
            use_fallback_normalization=self.use_fallback_normalization
        )
        
        self._print(f"训练样本数: {len(self.train_dataset)}")
        self._print(f"验证样本数: {len(self.val_dataset)}")
        self._print(f"测试样本数: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_feature_dim(self):
        """获取特征维度"""
        return len(self.feature_names) if self.feature_names else 0
    
    def get_stock_num(self):
        """获取股票数量"""
        return len(self.stock_names) if self.stock_names else 0
    
    def get_prediction_horizons(self):
        """获取预测期数"""
        return self.prediction_horizons
    
    def get_date_range(self):
        """获取数据的日期范围"""
        if self.date_index:
            return self.date_index[0], self.date_index[-1]
        return None, None
    
    def get_stock_names(self):
        """获取股票名称列表"""
        return self.stock_names if self.stock_names else []
    
    def create_prediction_dataset(self, return_metadata: bool = True):
        """创建用于预测的测试数据集（包含元数据）"""
        if self.test_dataset is None:
            raise ValueError("Test dataset not initialized. Call setup('test') first.")
        
        # Create a new test dataset with metadata enabled
        val_end = int(self.features.shape[0] * (self.train_ratio + self.val_ratio))
        
        prediction_dataset = StockDataset(
            features=self.features[val_end:],
            targets=self.targets[val_end:],
            feature_names=self.feature_names,
            sequence_length=self.sequence_length,
            normalize_features=self.normalize_features,
            date_index=self.date_index[val_end:] if self.date_index else None,
            stock_names=self.stock_names,
            return_metadata=return_metadata,
            outlier_clip_threshold=self.outlier_clip_threshold,
            noise_level=self.noise_level,
            use_fallback_normalization=self.use_fallback_normalization
        )
        
        return prediction_dataset
    
    def denormalize_targets(self, normalized_targets: torch.Tensor) -> torch.Tensor:
        """
        反标准化目标数据
        注意：由于采用截面标准化，反标准化需要对应的日期信息
        这里提供简化版本，实际使用时可能需要更复杂的逻辑
        
        Args:
            normalized_targets: 标准化的目标数据 [..., H]
            
        Returns:
            原始尺度的目标数据（截面标准化情况下较复杂）
        """
        # 截面标准化的反标准化较为复杂，需要知道具体的日期
        # 这里返回原值，实际应用中可能需要更精细的处理
        return normalized_targets

    def get_normalization_stats(self):
        """获取标准化统计信息"""
        return {
            'normalize_features': self.normalize_features,
            'normalize_targets': self.normalize_targets,
            'price_features': self.price_features,
            'volume_features': self.volume_features
        }
    
    def get_data_quality_stats(self):
        """获取数据质量统计信息，特别是NaN分布"""
        if self.features is None:
            return None
        
        T, F, N = self.features.shape
        
        # 计算每个时间步、特征、股票的NaN统计
        nan_by_time = torch.isnan(self.features).sum(dim=(1, 2))  # [T] - 每个时间步的NaN数量
        nan_by_feature = torch.isnan(self.features).sum(dim=(0, 2))  # [F] - 每个特征的NaN数量  
        nan_by_stock = torch.isnan(self.features).sum(dim=(0, 1))  # [N] - 每只股票的NaN数量
        
        # 计算完整序列的股票比例（针对不同序列长度）
        sequence_lengths = [10, 20, 30, 60]  # 测试不同序列长度
        complete_sequence_stats = {}
        
        for seq_len in sequence_lengths:
            if seq_len > T:
                continue
                
            complete_stocks_counts = []
            for start_t in range(T - seq_len + 1):
                end_t = start_t + seq_len
                seq_features = self.features[start_t:end_t]  # [seq_len, F, N]
                
                # 找出完整的股票
                has_nan = torch.isnan(seq_features).any(dim=(0, 1))  # [N]
                complete_stocks = (~has_nan).sum().item()
                complete_stocks_counts.append(complete_stocks)
            
            complete_sequence_stats[seq_len] = {
                'mean_complete_stocks': np.mean(complete_stocks_counts),
                'min_complete_stocks': min(complete_stocks_counts),
                'max_complete_stocks': max(complete_stocks_counts),
                'std_complete_stocks': np.std(complete_stocks_counts)
            }
        
        return {
            'total_shape': (T, F, N),
            'total_nan_count': torch.isnan(self.features).sum().item(),
            'nan_percentage': torch.isnan(self.features).float().mean().item() * 100,
            'nan_by_time_stats': {
                'mean': nan_by_time.float().mean().item(),
                'min': nan_by_time.min().item(),
                'max': nan_by_time.max().item(),
                'std': nan_by_time.float().std().item()
            },
            'nan_by_feature_stats': {
                'mean': nan_by_feature.float().mean().item(),
                'feature_names': self.feature_names,
                'nan_counts': nan_by_feature.tolist()
            },
            'nan_by_stock_stats': {
                'mean': nan_by_stock.float().mean().item(),
                'min': nan_by_stock.min().item(),
                'max': nan_by_stock.max().item(),
                'stocks_with_no_nan': (nan_by_stock == 0).sum().item()
            },
            'complete_sequence_stats': complete_sequence_stats
        }

# 使用示例
def main():
    """使用示例 - 新的数据处理策略：features不填充，动态选择有效股票"""
    
    # 示例1：只使用价格数据，features不填充，动态选择有效股票
    print("=== 示例1：价格数据，features不填充，动态选择有效股票 ===")
    price_datamodule = StockDataModule(
        use_factors=False,
        sequence_length=20,
        prediction_horizons=[1, 5, 10],
        batch_size=1,
        normalize_features=True,  # 启用序列级价格标准化
        normalize_targets=True,   # 启用截面标准化
        debug=True  # 开启调试输出
    )
    
    price_datamodule.prepare_data()
    price_datamodule.setup()
    
    # 获取一个批次的数据
    train_loader = price_datamodule.train_dataloader()
    batch = next(iter(train_loader))
    features, targets = batch
    
    print(f"批次特征形状: {features.shape}")  # [B, L, F, N_valid] - N_valid可能每个样本不同
    print(f"批次目标形状: {targets.shape}")   # [B, H, N_valid]
    print(f"特征维度: {price_datamodule.get_feature_dim()}")
    print(f"原始股票数量: {price_datamodule.get_stock_num()}")
    
    # 检查不同样本的有效股票数量
    print(f"\n有效股票数量变化:")
    for i in range(min(5, features.shape[0])):  # 检查前5个样本
        valid_stocks = features[i].shape[2]  # 第i个样本的有效股票数
        print(f"  样本{i}: {valid_stocks}只有效股票")
    
    # 检查特征中是否还有NaN值
    nan_count = torch.isnan(features).sum().item()
    print(f"特征中NaN数量: {nan_count}")
    
    # 检查目标中是否还有NaN值（应该为0，因为已填充）
    target_nan_count = torch.isnan(targets).sum().item()
    print(f"目标中NaN数量: {target_nan_count}")
    
    # 示例2：使用因子数据和价格数据
    print("\n=== 示例2：因子+价格数据，features不填充策略 ===")
    factor_datamodule = StockDataModule(
        use_factors=True,
        sequence_length=30,  # 增加序列长度测试
        prediction_horizons=[1, 5, 10],
        batch_size=1,  # 减少batch_size因为每个样本股票数可能不同
        normalize_features=True,  # 序列级价格标准化，因子不标准化
        normalize_targets=True,   # 截面标准化
        debug=False  # 关闭调试输出
    )
    
    factor_datamodule.prepare_data()
    factor_datamodule.setup()
    
    # 获取一个批次的数据
    train_loader = factor_datamodule.train_dataloader()
    for _ in range(10):
        batch = next(iter(train_loader))
        features, targets = batch
        
        print(f"批次特征形状: {features.shape}")  # [B, L, F, N_valid]
        print(f"批次目标形状: {targets.shape}")   # [B, H, N_valid]
        print(f"特征维度: {factor_datamodule.get_feature_dim()}")
        print(f"原始股票数量: {factor_datamodule.get_stock_num()}")
        
        # 分析有效股票数量分布
        valid_stock_counts = []
        for i in range(features.shape[0]):
            valid_stocks = features[i].shape[2]
            valid_stock_counts.append(valid_stocks)
        
        print(f"\n有效股票数量统计:")
        print(f"  平均: {np.mean(valid_stock_counts):.1f}")
        print(f"  最小: {min(valid_stock_counts)}")
        print(f"  最大: {max(valid_stock_counts)}")
        print(f"  标准差: {np.std(valid_stock_counts):.1f}")
        
        # 检查特征数据质量
        print(f"\n数据质量检查:")
        all_nan_count = 0
        for i in range(features.shape[0]):
            sample_nan = torch.isnan(features[i]).sum().item()
            all_nan_count += sample_nan
        
        print(f"  所有样本特征中NaN总数: {all_nan_count}")
        print(f"  目标中NaN总数: {torch.isnan(targets).sum().item()}")
        
        # 测试序列标准化效果
        print(f"\n序列标准化效果检查:")
        sample_idx = 0  # 检查第一个样本
        sample_features = features[sample_idx]  # [L, F, N_valid]
    
    # 找到close价格特征的索引
    dataset = factor_datamodule.train_dataset
    if dataset.close_feature_index is not None:
        close_idx = dataset.close_feature_index
        print(f"  Close价格特征索引: {close_idx}")
        
        # 检查最后一天的close价格是否接近1.0（相对标准化）
        last_day_close = sample_features[-1, close_idx, :]  # 最后一天，close特征，所有有效股票
        if len(last_day_close) > 0:
            print(f"  标准化后最后一天close价格统计: 均值={last_day_close.mean():.4f}, 标准差={last_day_close.std():.4f}")
    
    print("\n=== 新数据处理策略测试完成 ===")
    print("主要改进:")
    print("1. Features不进行数据填充，保留原始NaN值")
    print("2. 动态选择30天序列中完全有效的股票")
    print("3. Targets用0填充，表示无收益")
    print("4. 每个batch的股票数量可能不同，更真实反映数据质量")

if __name__ == "__main__":
    main()
