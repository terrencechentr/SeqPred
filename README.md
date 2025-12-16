# SeqPred - 序列预测框架

一个基于Transformer的序列预测框架，支持多种模型架构，提供统一的训练、预测和评估接口。

## ✨ 特性

- 🎯 **统一接口**：所有模型使用相同的训练和评估接口
- 🚀 **多种模型**：支持 CSEPT、CSEPT Smooth、Sept 三种模型架构
- 📊 **数据增强**：内置多种数据增强方法（随机缩放、特征dropout、时间扭曲、Mixup）
- 📈 **灵活训练**：支持按epoch或按步数训练，带实时进度条
- 🔍 **完整评估**：提供多种评估指标和可视化
- 🎨 **自回归预测**：支持多区间自回归预测，可配置噪声

## 📁 项目结构

```
SeqPred/
├── src/seqpred/              # 核心源代码
│   ├── models/               # 模型定义
│   │   ├── csept/           # CSEPT模型
│   │   ├── csept_smooth/    # CSEPT Smooth模型
│   │   └── sept/            # Sept模型
│   ├── data/                # 数据处理
│   │   └── spetDataset.py   # 数据集类（支持数据增强）
│   ├── train/               # 训练模块
│   │   └── trainer.py       # 统一训练器
│   ├── eval/                # 评估模块
│   │   └── evaluator.py     # 统一评估器
│   └── cli.py               # 统一命令行接口
├── scripts/                  # 分析工具和脚本
├── experiments/              # 实验结果（自动生成）
├── notebooks/               # Jupyter笔记本
├── train.py                 # 训练入口脚本
├── test.py                  # 测试入口脚本
└── pyproject.toml           # 项目配置
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 使用pip安装
pip install -r requirements.txt

# 或安装为可编辑包
pip install -e .
```

### 2. 训练模型

#### 基础训练

```bash
# 训练CSEPT模型
python train.py --model_type csept --epochs 50 --batch_size 32

# 训练CSEPT Smooth模型（带平滑窗口）
python train.py --model_type csept_smooth --epochs 50 --smooth_window_size 50

# 训练Sept模型
python train.py --model_type sept --epochs 50
```

#### 按步数训练（带进度条）

```bash
# 训练1000步（自动显示进度条）
python train.py --model_type csept_smooth --max_steps 1000 --batch_size 32
```

#### 使用数据增强训练

```bash
python train.py \
    --model_type csept_smooth \
    --max_steps 1000 \
    --train_start 0 --train_end 4000 \
    --test_start 4000 --test_end 5000 \
    --dataset_apply_smoothing \
    --dataset_smooth_window_size 20 \
    --noise_level 0.01 \
    --aug_random_scale \
    --aug_feature_dropout \
    --aug_time_warp \
    --aug_mixup \
    --output_dir experiments/my_training
```

#### 使用统一CLI（安装后）

```bash
seqpred train --model_type csept_smooth --epochs 50 --batch_size 32
```

### 3. 评估和预测

#### 基础评估

```bash
python test.py \
    --model_path experiments/csept_smooth_xxx/best_model \
    --model_type csept_smooth \
    --plot
```

#### 自回归预测

```bash
python test.py \
    --model_path experiments/csept_smooth_xxx/best_model \
    --model_type csept_smooth \
    --train_start 0 --train_end 4000 \
    --test_start 4000 --test_end 5000 \
    --initial_length 200 \
    --num_intervals 3 \
    --autoregressive_noise_std 0.01 \
    --plot
```

#### 使用统一CLI

```bash
seqpred test \
    --model_path experiments/csept_smooth_xxx/best_model \
    --model_type csept_smooth \
    --plot
```

## 📊 支持的模型

### CSEPT (Causal Self-attention Encoder for Price Prediction with Transformer)
- 基础的因果自注意力模型
- 适用于标准时间序列预测
- 支持多种损失函数（MSE、MAE、Smooth L1、Huber、Log Cosh）

### CSEPT Smooth
- 带平滑机制的CSEPT模型
- 支持可学习的平滑权重
- 支持多种恢复模式（identity、learnable、inverse）
- 适用于噪声数据、需要平滑预测的场景

### Sept (Sequence Prediction Transformer)
- 基于Qwen2的序列预测模型
- 使用分类方法进行预测
- 适用于离散化预测任务

## ⚙️ 训练参数详解

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_type` | str | 必需 | 模型类型 (csept/csept_smooth/sept) |
| `--data_path` | str | None | 数据文件路径（默认: src/data/data.csv） |
| `--seq_length` | int | 256 | 序列长度 |
| `--epochs` | int | 50 | 训练轮数 |
| `--max_steps` | int | None | 最大训练步数（如果指定，将覆盖epochs） |
| `--batch_size` | int | 32 | 批次大小 |
| `--learning_rate` | float | 0.0001 | 学习率 |
| `--early_stopping_patience` | int | 10 | 早停耐心值 |
| `--output_dir` | str | None | 输出目录（默认自动生成） |

### 数据区间参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--train_start` | int | 3000 | 训练集起始索引 |
| `--train_end` | int | 4000 | 训练集结束索引 |
| `--test_start` | int | 5000 | 测试集起始索引 |
| `--test_end` | int | 6000 | 测试集结束索引 |

### 模型架构参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--hidden_size` | int | 128 | 隐藏层维度 |
| `--num_layers` | int | 4 | Transformer层数 |
| `--num_heads` | int | 4 | 注意力头数 |
| `--intermediate_size` | int | 512 | FFN中间层维度 |
| `--loss_type` | str | mse | 损失函数类型 (mse/mae/smooth_l1/huber/log_cosh) |

### CSEPT Smooth特有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--smooth_window_size` | int | 50 | 平滑窗口大小 |
| `--smooth_learnable` | flag | False | 使用可学习的平滑权重 |
| `--unsmooth_mode` | str | identity | 恢复模式 (identity/learnable/inverse) |

### 数据增强参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--noise_level` | float | 0.001 | 训练噪声水平 |
| `--dataset_apply_smoothing` | flag | False | 是否在数据集阶段进行滑动平均平滑 |
| `--dataset_smooth_window_size` | int | 50 | 数据集平滑窗口大小 |
| `--dataset_smooth_target_features` | list | [0] | 需要平滑的特征索引列表 |
| `--aug_random_scale` | flag | False | 启用随机缩放增强 |
| `--aug_scale_range` | list | [0.9, 1.1] | 随机缩放范围 [min, max] |
| `--aug_feature_dropout` | flag | False | 启用特征dropout增强 |
| `--aug_feature_dropout_prob` | float | 0.1 | 特征dropout概率 |
| `--aug_time_warp` | flag | False | 启用时间扭曲增强 |
| `--aug_time_warp_sigma` | float | 0.2 | 时间扭曲的sigma参数 |
| `--aug_mixup` | flag | False | 启用Mixup增强 |
| `--aug_mixup_alpha` | float | 0.2 | Mixup的alpha参数 |

## 🔍 评估参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_path` | str | 必需 | 模型路径（使用save_pretrained保存的） |
| `--model_type` | str | 必需 | 模型类型 |
| `--data_path` | str | None | 数据文件路径 |
| `--train_start` | int | 3000 | 训练集起始索引（用于标准化统计） |
| `--train_end` | int | 4000 | 训练集结束索引 |
| `--test_start` | int | 4000 | 测试集起始索引 |
| `--test_end` | int | 5000 | 测试集结束索引 |
| `--batch_size` | int | 32 | 批次大小 |
| `--initial_length` | int | None | 自回归初始上下文长度 |
| `--autoregressive_steps` | int | 0 | 自回归预测步数 |
| `--num_intervals` | int | 3 | 自回归预测的区间数量 |
| `--autoregressive_noise_std` | float | 0.0 | 自回归预测时的噪声标准差 |
| `--plot` | flag | False | 是否绘制预测结果 |
| `--output_dir` | str | None | 输出目录（默认在模型目录下创建evaluation） |

## 📈 评估指标

评估脚本会计算以下指标：

### 误差指标
- **MSE** (均方误差)
- **RMSE** (均方根误差)
- **MAE** (平均绝对误差)
- **Median AE** (中位数绝对误差)
- **Max Error** (最大误差)
- **MAPE** (平均绝对百分比误差)

### 相关性指标
- **R² Score** (决定系数)
- **Correlation** (相关系数)

### 方向准确率
- **Direction Accuracy (总体)**：预测方向正确的比例
- **Up Direction Accuracy (上涨)**：上涨方向预测准确率
- **Down Direction Accuracy (下跌)**：下跌方向预测准确率

### 分布统计
- 预测值和真实值的均值、标准差

## 💾 数据格式

数据文件应为CSV格式，包含以下列：

| 列名 | 说明 |
|------|------|
| `Open` | 开盘价 |
| `High` | 最高价 |
| `Low` | 最低价 |
| `Close` | 收盘价 |
| `Volume` | 成交量 |

数据集会自动计算收益率并进行标准化处理。

## 🎯 使用示例

### 示例1：完整训练和评估流程

```bash
# 1. 训练模型（0-4000区间，使用数据增强）
python train.py \
    --model_type csept_smooth \
    --train_start 0 --train_end 4000 \
    --test_start 4000 --test_end 5000 \
    --max_steps 1000 \
    --dataset_apply_smoothing \
    --dataset_smooth_window_size 20 \
    --noise_level 0.01 \
    --aug_random_scale \
    --aug_feature_dropout \
    --aug_time_warp \
    --aug_mixup

# 2. 评估模型（在4000-5000区间）
python test.py \
    --model_path experiments/csept_smooth_xxx/best_model \
    --model_type csept_smooth \
    --train_start 0 --train_end 4000 \
    --test_start 4000 --test_end 5000 \
    --initial_length 200 \
    --num_intervals 3 \
    --autoregressive_noise_std 0.01 \
    --plot
```

### 示例2：使用脚本快速训练和评估

```bash
# 使用提供的脚本
bash scripts/train_eval_csept_smooth_3000_5000.sh
```

## 🔧 开发指南

### 添加新模型

1. 在 `src/seqpred/models/` 下创建新模型目录
2. 定义模型配置类（继承自`PretrainedConfig`）
3. 定义模型类（继承自`PreTrainedModel`）
4. 在 `src/seqpred/models/__init__.py` 中注册模型

```python
from transformers import AutoConfig, AutoModel

# 注册配置
AutoConfig.register("your_model", YourModelConfig)

# 注册模型
AutoModel.register(YourModelConfig, YourModel)
```

### 模型保存和加载

本框架使用HuggingFace的统一接口：

```python
from transformers import AutoModel, AutoConfig

# 加载模型
model = AutoModel.from_pretrained("path/to/model")

# 保存模型
model.save_pretrained("path/to/save")
```

## 📝 输出文件说明

### 训练输出

训练完成后，会在输出目录生成以下文件：

- `best_model/`：最佳模型（使用HuggingFace格式）
- `config.json`：模型配置
- `training_curves.png`：训练和测试损失曲线
- `training_history.txt`：详细的训练历史

### 评估输出

评估完成后，会在评估目录生成以下文件：

- `metrics.txt`：详细的评估指标
- `predictions.png`：预测结果可视化（如果启用`--plot`）
- `autoregressive_predictions_multi.npz`：自回归预测结果（numpy格式）
- `autoregressive_multi_intervals.png`：多区间自回归预测可视化（如果启用`--plot`）

## 🐛 常见问题

### Q: 如何查看训练进度？
A: 使用`--max_steps`参数训练时会自动显示进度条，显示当前步数、loss和最佳loss。

### Q: 如何选择合适的训练区间？
A: 建议训练集和测试集不重叠，且测试集在训练集之后。例如：训练集3000-4000，测试集4000-5000。

### Q: 数据增强会影响性能吗？
A: 数据增强通常能提升模型的泛化能力，但可能会增加训练时间。建议先不使用增强训练基线，再逐步添加增强方法。

### Q: 自回归预测的噪声参数如何设置？
A: 建议与训练时的噪声水平保持一致（`--noise_level`），例如训练时使用0.01，预测时也使用0.01。

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📧 联系方式

如有问题或建议，请通过Issue联系我们。
