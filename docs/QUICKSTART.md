# 快速开始指南

## 安装

```bash
# 克隆仓库
git clone <repository_url>
cd SeqPred

# 安装依赖
pip install -r requirements.txt

# 或者安装为包
pip install -e .
```

## 基本使用

### 训练模型

```bash
# CSEPT模型 - 基础版本
python train.py \
    --model_type csept \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.0001

# CSEPT Smooth模型 - 带平滑机制
python train.py \
    --model_type csept_smooth \
    --epochs 50 \
    --smooth_window_size 50 \
    --smooth_learnable

# Sept模型 - 基于Qwen2
python train.py \
    --model_type sept \
    --epochs 50 \
    --batch_size 16
```

### 测试模型

```bash
# 基础测试
python test.py \
    --model_path experiments/csept_training_xxx/best_model \
    --model_type csept \
    --plot

# 自回归预测
python test.py \
    --model_path experiments/csept_training_xxx/best_model \
    --model_type csept \
    --autoregressive_steps 100 \
    --plot
```

## 自定义配置

### 修改模型架构

```bash
python train.py \
    --model_type csept \
    --hidden_size 256 \
    --num_layers 6 \
    --num_heads 8 \
    --intermediate_size 1024
```

### 修改训练策略

```bash
python train.py \
    --model_type csept \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --early_stopping_patience 15 \
    --loss_type huber
```

### 使用自定义数据

```bash
python train.py \
    --model_type csept \
    --data_path /path/to/your/data.csv \
    --seq_length 512
```

## 继续训练

```bash
# 从已有模型继续训练
python train.py \
    --model_type csept \
    --pretrained_model_path experiments/csept_training_xxx/best_model \
    --epochs 50
```

## 输出说明

训练完成后，会在 `experiments/` 目录下生成：
- `best_model/` - 最佳模型权重和配置
- 训练日志

测试完成后，会在模型目录下的 `evaluation/` 目录生成：
- `predictions.png` - 预测结果可视化
- `autoregressive_predictions.npy` - 自回归预测结果
- `autoregressive_plot.png` - 自回归预测可视化

## 常见问题

### Q: 如何选择合适的模型？
- **CSEPT**: 适合基础时间序列预测，速度快
- **CSEPT Smooth**: 适合有噪声的数据，预测更平滑
- **Sept**: 适合离散化预测任务

### Q: 如何调整超参数？
1. 先使用默认参数训练baseline
2. 根据训练曲线调整学习率
3. 根据过拟合情况调整模型大小
4. 使用早停避免过拟合

### Q: 训练太慢怎么办？
- 减小 `batch_size`
- 减小 `hidden_size` 和 `num_layers`
- 使用GPU训练
- 减少 `seq_length`

## 下一步

- 查看 `README.md` 了解详细文档
- 查看 `scripts/` 目录的分析工具
- 查看 `notebooks/` 目录的示例笔记本

