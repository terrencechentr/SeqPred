# 评估功能改进说明

## 🎯 改进内容

### 1. 新增评估指标

#### 误差指标
- **MSE** (Mean Squared Error): 均方误差
- **RMSE** (Root Mean Squared Error): 均方根误差
- **MAE** (Mean Absolute Error): 平均绝对误差
- **Median AE**: 中位数绝对误差
- **Max Error**: 最大误差
- **MAPE** (Mean Absolute Percentage Error): 平均绝对百分比误差

#### 相关性指标
- **R² Score**: 决定系数，衡量模型拟合优度
- **Correlation**: 皮尔逊相关系数

#### 方向准确率
- **Direction Accuracy (总体)**: 整体方向预测准确率
- **Up Direction Accuracy (上涨)**: 上涨方向预测准确率
- **Down Direction Accuracy (下跌)**: 下跌方向预测准确率

#### 分布统计
- **Pred Mean/Std**: 预测值的均值和标准差
- **Label Mean/Std**: 真实值的均值和标准差

### 2. 改进的可视化

#### 预测对比图（4个子图）
1. **时序对比图**: 真实值 vs 预测值，带误差区域
2. **散点图**: 预测值 vs 真实值，包含拟合线和R²
3. **误差分布直方图**: 显示预测误差的分布
4. **累积误差曲线**: 显示误差随时间的累积

#### 多区间自回归预测图
- 支持在测试集的多个位置进行自回归预测
- 每个区间显示：
  - 真实值曲线（蓝色实线）
  - 预测值曲线（红色虚线）
  - 误差区域（灰色填充）
  - 该区间的MAE、RMSE和方向准确率

### 3. 新增功能

#### 多区间自回归预测
- 参数: `--num_intervals` (默认3个区间)
- 在测试集的不同位置进行预测
- 对比不同区间的预测效果
- 评估模型在不同数据分布下的表现

#### 指标保存
- 自动保存评估指标到 `metrics.txt`
- 便于后续分析和对比

## 📊 使用方法

### 基础评估（包含新指标和改进可视化）

```bash
python test.py \
    --model_path experiments/your_model/best_model \
    --model_type csept_smooth \
    --plot
```

### 多区间自回归预测

```bash
python test.py \
    --model_path experiments/your_model/best_model \
    --model_type csept_smooth \
    --plot \
    --autoregressive_steps 100 \
    --num_intervals 5
```

参数说明：
- `--autoregressive_steps`: 每个区间预测的步数
- `--num_intervals`: 在测试集中选择的区间数量

## 📁 输出文件

评估完成后，会在 `evaluation/` 目录下生成：

```
evaluation/
├── metrics.txt                          # 详细的评估指标
├── predictions.png                      # 4合1预测对比图
├── autoregressive_multi_intervals.png   # 多区间自回归预测对比
└── autoregressive_predictions_multi.npz # 预测数据（NumPy格式）
```

## 🔍 指标解读

### 误差指标
- **MSE/RMSE**: 越小越好，对大误差敏感
- **MAE**: 越小越好，对所有误差一视同仁
- **Median AE**: 中位数误差，对异常值不敏感
- **MAPE**: 百分比误差，注意可能因除以接近0的值而很大

### 相关性指标
- **R² Score**: 
  - 1.0 = 完美拟合
  - 0.0 = 等同于预测均值
  - < 0 = 比预测均值还差
- **Correlation**:
  - 1.0 = 完全正相关
  - 0.0 = 无相关
  - -1.0 = 完全负相关

### 方向准确率
- **> 50%**: 比随机猜测好
- **60-70%**: 较好的方向预测
- **> 70%**: 优秀的方向预测

### 分布统计
- 比较预测值和真实值的分布差异
- Std差异大说明模型可能过于保守或激进

## 💡 使用建议

### 1. 全面评估
```bash
# 完整评估：基础指标 + 3个区间自回归预测
python test.py \
    --model_path experiments/your_model/best_model \
    --model_type csept_smooth \
    --plot \
    --autoregressive_steps 100 \
    --num_intervals 3
```

### 2. 详细分析
```bash
# 更多区间，更短步数
python test.py \
    --model_path experiments/your_model/best_model \
    --model_type csept_smooth \
    --plot \
    --autoregressive_steps 50 \
    --num_intervals 5
```

### 3. 长期预测
```bash
# 少量区间，更长步数
python test.py \
    --model_path experiments/your_model/best_model \
    --model_type csept_smooth \
    --plot \
    --autoregressive_steps 200 \
    --num_intervals 2
```

## 📈 示例输出

### 控制台输出
```
评估指标:

=== 误差指标 ===
  MSE: 0.315978
  RMSE: 0.562119
  MAE: 0.411978
  Median AE: 0.312995
  Max Error: 4.607116
  MAPE: 208.76%

=== 相关性指标 ===
  R² Score: 0.019710
  Correlation: 0.182203

=== 方向准确率 ===
  Direction Accuracy (总体): 52.76%
  Up Direction Accuracy (上涨): 63.73%
  Down Direction Accuracy (下跌): 43.08%

开始多区间自回归预测 (3 个区间, 每个 100 步)...
  区间 1/3: 起始位置 0
    MAE: 0.4626, RMSE: 0.5833
  区间 2/3: 起始位置 321
    MAE: 0.6145, RMSE: 0.8436
  区间 3/3: 起始位置 643
    MAE: 0.4265, RMSE: 0.5349
```

## 🎓 最佳实践

1. **先看整体指标**: 了解模型的基本性能
2. **查看可视化**: 直观理解预测质量
3. **分析多区间**: 评估模型的稳定性
4. **对比不同模型**: 使用相同的评估设置

## 🔧 技术细节

### 新增依赖
- `scipy`: 用于统计计算（相关系数、线性回归）
- `pandas`: 用于数据处理

### 改进的函数
- `calculate_metrics()`: 扩展到15+个指标
- `plot_predictions()`: 从单图改为4合1多子图
- `autoregressive_predict()`: 支持多区间预测
- 新增 `get_ground_truth_sequence()`: 获取真实值序列
- 新增 `plot_autoregressive_with_ground_truth()`: 绘制多区间对比

## 📝 更新日志

**版本**: 2.0
**日期**: 2025-12-15

**新增**:
- ✅ 15+个评估指标
- ✅ 4合1预测对比图
- ✅ 多区间自回归预测
- ✅ 真实值对比
- ✅ 指标自动保存

**改进**:
- ✅ 更详细的误差分析
- ✅ 更直观的可视化
- ✅ 更全面的性能评估

