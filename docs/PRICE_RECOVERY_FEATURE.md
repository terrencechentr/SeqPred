# 绝对价格恢复功能说明

## 🎯 新增功能

自回归预测现在会自动将标准化的收益率恢复为**绝对价格**进行绘图，使结果更加直观易懂。

## 📊 功能特点

### 1. 双重指标计算

评估时会同时计算两套指标：

#### 收益率层面（用于模型性能评估）
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- 方向准确率

#### 价格层面（更直观）
- Price MAE: 价格的平均绝对误差（单位：价格）
- Price MAPE: 价格的平均绝对百分比误差（单位：%）

### 2. 绝对价格可视化

自回归预测图现在显示：
- **Y轴**: 绝对价格（而非标准化收益率）
- **蓝色实线**: 真实价格曲线
- **红色虚线**: 预测价格曲线
- **灰色填充**: 预测误差区域
- **标题**: 包含收益率和价格两套指标

### 3. 数据保存

预测结果会保存为 `.npz` 文件，包含：
- `predictions`: 标准化收益率预测
- `ground_truth`: 标准化收益率真实值
- `pred_prices`: **绝对价格预测** ⭐ 新增
- `truth_prices`: **绝对价格真实值** ⭐ 新增
- `start_positions`: 起始位置

## 🔧 技术实现

### 价格恢复流程

1. **反标准化**
   ```python
   original_returns = normalized_returns * std + mean
   ```

2. **累积计算价格**
   ```python
   price_t = price_{t-1} * (1 + return_t)
   ```

3. **起始价格**
   - 从原始数据中获取对应位置的真实价格
   - 确保预测和真实值使用相同的起始点

### 数据集改进

`SpetDataset` 现在会保存：
- `original_close_prices_train`: 训练集的原始Close价格
- `original_close_prices_test`: 测试集的原始Close价格
- `mean`, `std`: 标准化参数（用于恢复）

## 📈 使用示例

### 基础使用（自动启用价格恢复）

```bash
python test.py \
    --model_path experiments/your_model/best_model \
    --model_type csept_smooth \
    --plot \
    --autoregressive_steps 100 \
    --num_intervals 3
```

### 输出示例

```
开始多区间自回归预测 (3 个区间, 每个 100 步)...
  区间 1/3: 起始位置 0
    收益率 - MAE: 0.4626, RMSE: 0.5833
    价格 - MAE: 48.86, MAPE: 1.47%
  区间 2/3: 起始位置 321
    收益率 - MAE: 0.6145, RMSE: 0.8436
    价格 - MAE: 392.00, MAPE: 12.37%
  区间 3/3: 起始位置 643
    收益率 - MAE: 0.4265, RMSE: 0.5349
    价格 - MAE: 245.61, MAPE: 7.77%
```

### 读取保存的价格数据

```python
import numpy as np

# 加载数据
data = np.load('experiments/xxx/evaluation/autoregressive_predictions_multi.npz', 
               allow_pickle=True)

# 获取价格数据
pred_prices = data['pred_prices']      # 预测价格 (n_intervals, n_steps+1)
truth_prices = data['truth_prices']    # 真实价格 (n_intervals, n_steps+1)

# 第一个区间的价格
interval_1_pred = pred_prices[0]
interval_1_truth = truth_prices[0]

print(f"预测价格范围: {interval_1_pred.min():.2f} - {interval_1_pred.max():.2f}")
print(f"真实价格范围: {interval_1_truth.min():.2f} - {interval_1_truth.max():.2f}")
```

## 🔍 指标解读

### 收益率指标
- **MAE/RMSE**: 衡量模型在收益率预测上的准确性
- 用于评估模型的核心性能

### 价格指标
- **Price MAE**: 平均价格误差（例如：48.86 表示平均偏差约49元）
- **Price MAPE**: 平均百分比误差（例如：1.47% 表示平均偏差1.47%）
- 更直观，便于理解实际应用价值

### 指标关系
- 收益率指标好 ≠ 价格指标好（因为误差会累积）
- 价格MAPE通常比收益率误差更大（累积效应）
- 两者结合评估更全面

## 📊 可视化改进

### 之前（标准化收益率）
- Y轴范围: -2 到 2（无实际意义）
- 难以理解预测的实际价值
- 不便于与实际应用对接

### 现在（绝对价格）
- Y轴范围: 实际价格（例如：3200 - 3600）
- 直观显示价格走势
- 便于评估实际应用价值
- 标题同时显示两套指标

## 💡 最佳实践

### 1. 评估模型性能
```bash
# 关注收益率指标（MAE, RMSE）
# 这些指标反映模型的核心预测能力
```

### 2. 评估实际应用价值
```bash
# 关注价格指标（Price MAE, Price MAPE）
# 这些指标反映实际交易中的误差
```

### 3. 多区间对比
```bash
# 使用多个区间评估模型稳定性
python test.py \
    --model_path experiments/your_model/best_model \
    --model_type csept_smooth \
    --plot \
    --autoregressive_steps 100 \
    --num_intervals 5  # 更多区间
```

### 4. 长期预测
```bash
# 评估误差累积效应
python test.py \
    --model_path experiments/your_model/best_model \
    --model_type csept_smooth \
    --plot \
    --autoregressive_steps 200  # 更长预测
    --num_intervals 2
```

## 🎓 注意事项

### 1. 误差累积
- 自回归预测会累积误差
- 预测步数越多，价格误差越大
- 这是正常现象，不代表模型性能差

### 2. 起始价格
- 每个区间使用真实的起始价格
- 确保预测和真实值在相同基准上对比

### 3. 数据对齐
- 价格序列比收益率序列多1个元素（包含起始价格）
- `pred_prices[0]` 是起始价格
- `pred_prices[1:]` 是预测的价格序列

### 4. MAPE解读
- MAPE对小价格敏感
- 如果价格接近0，MAPE可能很大
- 建议结合MAE一起看

## 📝 更新日志

**版本**: 2.1
**日期**: 2025-12-15

**新增**:
- ✅ 绝对价格恢复功能
- ✅ 价格层面的MAE和MAPE指标
- ✅ 价格数据保存
- ✅ 改进的可视化（使用绝对价格）

**改进**:
- ✅ 更直观的预测结果展示
- ✅ 双重指标评估（收益率+价格）
- ✅ 完整的数据保存（收益率+价格）

## 🔗 相关文档

- 评估功能改进: `EVALUATION_IMPROVEMENTS.md`
- 快速开始: `QUICKSTART.md`
- 完整文档: `README.md`

