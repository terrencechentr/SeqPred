import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import numpy as np

class SpetDataset(Dataset):
    """
    数据集类 - 负责数据加载、特征提取、平滑和标准化
    
    新增功能:
    1. apply_smoothing: 是否在数据集中进行平滑
    2. smooth_window_size: 平滑窗口大小
    3. smooth_target_features: 需要平滑的特征索引列表
    
    Bug修复:
    1. 修复了__getitem__返回的标签维度错误
    2. 修复了__len__的计算
    3. 添加了数据范围的灵活配置
    4. 修复了Volume特征的处理方式
    """
    def __init__(
        self, 
        csv_file, 
        pred_length, 
        is_train=True, 
        noise_level=0.1,
        train_start=3000,
        train_end=4000,
        test_start=5000,
        test_end=6000,
        apply_smoothing=False,
        smooth_window_size=50,
        smooth_target_features=None,
        # 数据增强参数
        aug_random_scale=True,
        aug_scale_range=(0.9, 1.1),
        aug_feature_dropout=False,
        aug_feature_dropout_prob=0.1,
        aug_time_warp=False,
        aug_time_warp_sigma=0.2,
        aug_mixup=False,
        aug_mixup_alpha=0.2,
    ):
        self.csv_file = csv_file
        self.is_train = is_train
        self.noise_level = noise_level
        self.pred_length = pred_length
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.apply_smoothing = apply_smoothing
        self.smooth_window_size = smooth_window_size
        self.smooth_target_features = smooth_target_features if smooth_target_features is not None else [0]
        
        # 数据增强参数
        self.aug_random_scale = aug_random_scale and is_train
        self.aug_scale_range = aug_scale_range
        self.aug_feature_dropout = aug_feature_dropout and is_train
        self.aug_feature_dropout_prob = aug_feature_dropout_prob
        self.aug_time_warp = aug_time_warp and is_train
        self.aug_time_warp_sigma = aug_time_warp_sigma
        self.aug_mixup = aug_mixup and is_train
        self.aug_mixup_alpha = aug_mixup_alpha
        
        self.raw_data = pd.read_csv(csv_file)
        train_data, test_data = self.process_data(self.raw_data)

        if self.is_train:
            self.data = train_data
        else:
            self.data = test_data

    def __len__(self):
        """
        返回数据集大小
        Bug修复: 之前是 len(self.data) - self.pred_length - 1，多减了1
        正确的计算: 如果数据有1000个点，pred_length=256，
        那么可以创建的样本数是 1000 - 256 = 744 个
        """
        return len(self.data) - self.pred_length

    def __getitem__(self, idx):
        """
        返回一个训练样本
        
        Args:
            idx: 样本索引
            
        Returns:
            input_values: (pred_length, 5) - 输入特征序列
            labels: (pred_length, 1) - 目标标签（Close价格的收益率）
            
        Bug修复:
        1. 之前获取 pred_length+1 个数据点，现在只获取 pred_length 个
        2. 之前返回 data[:, 0:1]，维度错误，现在返回 data[:, :, 0:1] 或正确切片
        3. labels 应该是完整序列的第一个特征（Close收益率）
        """
        # 获取 pred_length 个连续数据点
        data = self.data[idx:idx+self.pred_length]

        # 训练时应用数据增强
        if self.is_train:
            # 1. 添加噪声
            data = data + torch.randn_like(data) * self.noise_level
            
            # 2. 随机缩放（Random Scaling）
            if self.aug_random_scale:
                scale = random.uniform(self.aug_scale_range[0], self.aug_scale_range[1])
                data = data * scale
            
            # 3. 特征dropout（Feature Dropout）
            if self.aug_feature_dropout:
                dropout_mask = torch.rand(data.shape[1]) > self.aug_feature_dropout_prob
                data = data * dropout_mask.unsqueeze(0).float()
            
            # 4. 时间扭曲（Time Warping）
            if self.aug_time_warp:
                data = self._apply_time_warp(data)
            
            # 5. Mixup增强
            if self.aug_mixup and random.random() < 0.5:
                # 随机选择另一个样本进行混合
                other_idx = random.randint(0, len(self) - 1)
                other_data = self.data[other_idx:other_idx+self.pred_length]
                lam = np.random.beta(self.aug_mixup_alpha, self.aug_mixup_alpha)
                data = lam * data + (1 - lam) * other_data

        # 输入是所有5个特征，标签是Close收益率（第一列）
        # data shape: (pred_length, 5)
        # labels shape: (pred_length, 1)
        labels = data[:, 0:1]  # 取第一列，保持维度
        
        return data, labels

    def _apply_time_warp(self, data):
        """
        应用时间扭曲增强
        
        Args:
            data: torch.Tensor, shape (seq_length, n_features)
            
        Returns:
            warped_data: torch.Tensor, shape (seq_length, n_features)
        """
        seq_length = data.shape[0]
        # 生成扭曲路径
        warp = torch.cumsum(torch.randn(seq_length) * self.aug_time_warp_sigma, dim=0)
        warp = warp - warp[0]  # 从0开始
        warp = warp / warp[-1] * (seq_length - 1)  # 归一化到序列长度
        
        # 使用线性插值进行扭曲
        warped_data = torch.zeros_like(data)
        for i in range(seq_length):
            pos = warp[i].item()
            if pos < 0:
                pos = 0
            if pos >= seq_length - 1:
                pos = seq_length - 2
            
            idx_low = int(pos)
            idx_high = idx_low + 1
            alpha = pos - idx_low
            
            warped_data[i] = (1 - alpha) * data[idx_low] + alpha * data[idx_high]
        
        return warped_data
    
    def apply_moving_average_smoothing(self, data):
        """
        对数据应用滑动平均平滑
        
        Args:
            data: torch.Tensor, shape (n_samples, n_features)
            
        Returns:
            smoothed_data: torch.Tensor, shape (n_samples, n_features)
        """
        if not self.apply_smoothing:
            return data
        
        smoothed_data = data.clone()
        
        for feat_idx in self.smooth_target_features:
            feature = data[:, feat_idx].numpy()
            
            # 使用pandas的rolling方法进行平滑
            feature_series = pd.Series(feature)
            smoothed_feature = feature_series.rolling(
                window=self.smooth_window_size, 
                min_periods=1,  # 开始部分使用可用的数据
                center=False    # 不居中，只使用历史数据
            ).mean()
            
            smoothed_data[:, feat_idx] = torch.tensor(smoothed_feature.values, dtype=torch.float32)
        
        return smoothed_data

    def process_data(self, data):
        """
        处理数据：提取特征、平滑和标准化
        
        流程:
        1. 计算收益率
        2. 应用平滑（如果启用）
        3. 标准化
        
        Bug修复:
        1. 移除了硬编码的数据范围，使用参数控制
        2. 确保所有特征使用一致的pct_change处理
        """
        
        # 计算所有特征的变化率
        # 注意: pct_change() 的第一个值是NaN，我们用[1:]跳过
        close_returns = data['Close'].pct_change().values[1:]
        open_returns = data['Open'].pct_change().values[1:]
        high_returns = data['High'].pct_change().values[1:]
        low_returns = data['Low'].pct_change().values[1:]
        volume_returns = data['Volume'].pct_change().values[1:]
        
        # 组合所有特征: (n_samples, 5)
        all_data = torch.tensor(
            np.stack([
                close_returns,
                open_returns,
                high_returns,
                low_returns,
                volume_returns,
            ], axis=1),
            dtype=torch.float32
        )
        
        # 提取训练集和测试集
        train_data = all_data[self.train_start:self.train_end]
        test_data = all_data[self.test_start:self.test_end]
        
        # 保存原始数据用于分析（仅Close收益率，平滑前）
        self.raw_train_data = close_returns[self.train_start:self.train_end].tolist()
        self.raw_test_data = close_returns[self.test_start:self.test_end].tolist()
        
        # 保存原始价格数据（用于恢复绝对价格）
        # +1是因为pct_change跳过了第一个值
        self.original_close_prices_train = data['Close'].values[self.train_start+1:self.train_end+1]
        self.original_close_prices_test = data['Close'].values[self.test_start+1:self.test_end+1]
        
        # 应用平滑（如果启用）
        if self.apply_smoothing:
            print(f"Applying smoothing with window size {self.smooth_window_size} "
                  f"to features {self.smooth_target_features}")
            train_data = self.apply_moving_average_smoothing(train_data)
            test_data = self.apply_moving_average_smoothing(test_data)

        # 使用训练集的统计信息进行标准化
        # mean和std的shape: (5,)，对应5个特征
        self.mean = train_data.mean(dim=0)
        self.std = train_data.std(dim=0)
        
        # 防止除零错误
        self.std = torch.where(self.std > 1e-8, self.std, torch.ones_like(self.std))
        
        # 标准化训练集和测试集
        train_data = (train_data - self.mean) / self.std
        test_data = (test_data - self.mean) / self.std

        return train_data, test_data

