 项目结构说明

## 目录结构

```
SeqPred/
├── src/                          # 核心源代码
│   ├── models/                   # 模型定义
│   │   ├── __init__.py          # 模型注册
│   │   ├── csept/               # CSEPT模型
│   │   ├── csept_smooth/        # CSEPT Smooth模型
│   │   ├── csept_smooth_improved.py  # 改进版本
│   │   └── sept/                # Sept模型
│   ├── data/                    # 数据处理
│   │   ├── __init__.py
│   │   ├── spetDataset.py       # 数据集类
│   │   └── data.csv             # 数据文件
│   ├── train/                   # 训练模块
│   │   ├── __init__.py
│   │   └── trainer.py           # 统一训练器
│   ├── eval/                    # 评估模块
│   │   ├── __init__.py
│   │   └── evaluator.py         # 统一评估器
│   ├── utils/                   # 工具函数
│   │   ├── processor.py         # 数据处理
│   │   └── data_collator.py     # 数据整理
│   └── tutils/                  # 技术工具
│       ├── io.py
│       ├── matrix.py
│       ├── memory.py
│       ├── plot.py
│       ├── random.py
│       └── timing.py
├── scripts/                     # 分析脚本
│   ├── analyze_train_test_comparison.py  # 训练测试对比分析
│   ├── autoregressive_prediction.py      # 自回归预测工具
│   └── plot_direction_improvement.py     # 方向改进可视化
├── experiments/                 # 实验结果（自动生成）
├── notebooks/                   # Jupyter笔记本
├── wandb/                       # Weights & Biases日志
├── train.py                     # 训练入口
├── test.py                      # 测试入口
├── setup.py                     # 安装脚本
├── requirements.txt             # 依赖包
├── README.md                    # 详细文档
├── QUICKSTART.md               # 快速开始
├── STRUCTURE.md                # 本文件
└── .gitignore                  # Git忽略配置

```

## 模块说明

### 1. src/models/ - 模型模块

所有模型都遵循HuggingFace的统一接口：
- 配置类继承 `PretrainedConfig`
- 模型类继承 `PreTrainedModel`
- 通过 `AutoConfig` 和 `AutoModel` 注册

**支持的模型：**

#### CSEPT (Causal Self-attention Encoder for Price Prediction)
- 文件: `src/models/csept/`
- 特点: 基础因果注意力模型
- 适用: 标准时间序列预测

#### CSEPT Smooth
- 文件: `src/models/csept_smooth/`
- 特点: 带平滑和恢复机制
- 适用: 噪声数据、需要平滑预测

#### Sept (Sequence Prediction Transformer)
- 文件: `src/models/sept/`
- 特点: 基于Qwen2的分类预测
- 适用: 离散化预测任务

### 2. src/data/ - 数据模块

#### SpetDataset
- 位置: `src/data/spetDataset.py`
- 功能: 时间序列数据加载和预处理
- 支持: 训练/测试集划分、噪声注入

### 3. src/train/ - 训练模块

#### trainer.py
- 统一训练接口
- 支持所有模型类型
- 功能：
  - 模型创建和配置
  - 训练循环
  - 早停机制
  - 模型保存

### 4. src/eval/ - 评估模块

#### evaluator.py
- 统一评估接口
- 支持所有模型类型
- 功能：
  - 模型加载
  - 批量评估
  - 自回归预测
  - 可视化

### 5. scripts/ - 分析工具

保留的核心分析脚本：
- `analyze_train_test_comparison.py`: 训练测试对比分析
- `autoregressive_prediction.py`: 自回归预测工具
- `plot_direction_improvement.py`: 方向预测改进可视化

## 统一接口设计

### 训练接口

```python
# 命令行
python train.py --model_type csept --epochs 50

# 或在代码中
from src.train import train_model
train_model(args, device)
```

### 测试接口

```python
# 命令行
python test.py --model_path path/to/model --model_type csept

# 或在代码中
from src.eval import evaluate_model
evaluate_model(model, test_loader, device, model_type)
```

### 模型接口

```python
from transformers import AutoModel, AutoConfig

# 创建新模型
config = AutoConfig.from_pretrained("path/to/config")
model = AutoModel.from_config(config)

# 加载已有模型
model = AutoModel.from_pretrained("path/to/model")

# 保存模型
model.save_pretrained("path/to/save")
```

## 工作流程

### 训练流程

1. 用户调用 `train.py`
2. 导入 `src/train/trainer.py`
3. 创建/加载模型配置
4. 加载数据集
5. 训练循环
6. 保存最佳模型到 `experiments/`

### 测试流程

1. 用户调用 `test.py`
2. 导入 `src/eval/evaluator.py`
3. 从 `experiments/` 加载模型
4. 加载测试数据
5. 评估和预测
6. 生成可视化结果

## 扩展指南

### 添加新模型

1. 在 `src/models/` 创建新目录
2. 实现配置类和模型类
3. 在 `src/models/__init__.py` 注册
4. 在 `trainer.py` 添加配置创建逻辑
5. 测试训练和评估

### 添加新功能

1. 工具函数 → `src/utils/` 或 `src/tutils/`
2. 数据处理 → `src/data/`
3. 分析脚本 → `scripts/`
4. 示例代码 → `notebooks/`

## 最佳实践

1. **模型开发**: 使用统一接口，遵循HuggingFace标准
2. **代码组织**: 核心代码在 `src/`，脚本在 `scripts/`
3. **实验管理**: 结果自动保存到 `experiments/`
4. **版本控制**: 实验目录不提交，模型通过路径引用

## 依赖关系

```
train.py → src/train/trainer.py → src/models/
                                 → src/data/
                                 
test.py → src/eval/evaluator.py → src/models/
                                 → src/data/

src/models/ → transformers (AutoModel, AutoConfig)
src/data/ → torch.utils.data
```

## 配置管理

所有配置通过命令行参数传递：
- 模型配置 → `--model_type`, `--hidden_size` 等
- 训练配置 → `--epochs`, `--batch_size` 等
- 数据配置 → `--data_path`, `--seq_length` 等

配置会随模型保存到 `experiments/` 目录。

