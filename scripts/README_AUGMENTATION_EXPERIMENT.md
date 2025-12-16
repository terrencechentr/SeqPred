# 数据增强方法影响实验

本实验脚本用于系统性地探究不同数据增强方法对模型性能的影响。

## 📋 实验设计

实验包含以下9个配置：

1. **baseline**: 基线（无增强）
2. **noise_only**: 仅噪声增强
3. **random_scale**: 随机缩放
4. **feature_dropout**: 特征Dropout
5. **time_warp**: 时间扭曲
6. **mixup**: Mixup增强
7. **scale_dropout**: 缩放+Dropout组合
8. **scale_warp**: 缩放+时间扭曲组合
9. **all_aug**: 全部增强方法

## 🚀 使用方法

### 方法1：使用Shell脚本（推荐）

```bash
cd /SeqPred
bash scripts/run_augmentation_experiment.sh
```

### 方法2：直接运行Python脚本

```bash
cd /SeqPred
python scripts/experiment_augmentation.py
```

## ⚙️ 实验配置

### 训练参数
- **模型类型**: csept_smooth
- **训练区间**: 0-4000
- **测试区间**: 4000-5000
- **训练步数**: 1000步
- **批次大小**: 32
- **平滑窗口**: 20（数据集级别）

### 评估参数
- **初始长度**: 200
- **预测区间数**: 3
- **自回归噪声**: 0.01

## 📊 输出结果

实验完成后，会在 `experiments/augmentation_experiment_YYYYMMDD_HHMMSS/` 目录下生成：

### 数据文件
- **results.csv**: 所有实验结果的CSV格式汇总
- **results.json**: 所有实验结果的JSON格式汇总

### 可视化文件
- **augmentation_comparison.png**: 多指标对比柱状图
  - 最佳测试Loss对比
  - R² Score对比
  - 方向准确率对比
  - MAE对比

- **augmentation_heatmap.png**: 指标热力图
  - 展示各增强方法在不同指标上的表现
  - 颜色越绿表示性能越好

### 报告文件
- **experiment_report.md**: 详细的实验报告
  - 实验配置
  - 结果汇总表格
  - 关键发现总结

## 📈 评估指标

实验会收集以下指标：

### 训练指标
- 最佳测试Loss
- 最终训练Loss
- 最终测试Loss
- 最佳Epoch

### 评估指标
- **误差指标**: MSE, RMSE, MAE, Median AE, Max Error, MAPE
- **相关性指标**: R² Score, Correlation
- **方向准确率**: 总体、上涨、下跌
- **分布统计**: 预测值和真实值的均值、标准差

## ⏱️ 预计运行时间

- 每个实验约2-3分钟（1000步训练 + 评估）
- 总共9个实验，预计需要20-30分钟

## 🔧 自定义实验

如需修改实验配置，编辑 `scripts/experiment_augmentation.py`：

```python
# 修改训练参数
TRAIN_PARAMS = {
    "max_steps": 2000,  # 增加训练步数
    "batch_size": 64,   # 修改批次大小
    # ...
}

# 添加新的实验配置
EXPERIMENTS.append({
    "name": "custom_exp",
    "description": "自定义实验",
    "args": ["--noise_level", "0.02", "--aug_random_scale"]
})
```

## 📝 结果解读

### 最佳配置选择

1. **如果关注测试Loss**: 选择 `best_test_loss` 最小的实验
2. **如果关注预测准确性**: 选择 `R² Score` 最大的实验
3. **如果关注方向预测**: 选择 `Direction Accuracy` 最高的实验
4. **如果关注误差**: 选择 `MAE` 或 `RMSE` 最小的实验

### 增强方法效果分析

- **单个增强方法**: 对比 baseline 和各个单独增强方法，找出最有效的单个方法
- **组合增强方法**: 对比组合方法的效果，看是否有协同效应
- **全部增强**: 对比是否所有方法一起使用效果最好

## 🐛 故障排除

### 问题1：内存不足
**解决方案**: 减少 `max_steps` 或 `batch_size`

### 问题2：CUDA内存不足
**解决方案**: 在训练脚本中添加 `--device cpu` 或减少批次大小

### 问题3：实验中断
**解决方案**: 脚本会保存已完成实验的结果，可以手动继续未完成的实验

## 📧 注意事项

1. 确保有足够的磁盘空间（每个实验约50-100MB）
2. 建议在GPU环境下运行以加快训练速度
3. 实验过程中会占用GPU资源，请确保没有其他任务在使用GPU

