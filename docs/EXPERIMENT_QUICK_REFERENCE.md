# 实验快速参考

## 🎯 核心结论

**最佳平滑窗口大小**: **50**

## 📊 性能对比

| 指标 | 窗口10 | 窗口20 | 窗口30 | 窗口50 | 最优 |
|-----|--------|--------|--------|--------|------|
| Test Loss | 0.3339 | 0.3465 | 0.3443 | **0.3332** | ✅ 50 |
| MAE | 0.4202 | 0.4195 | 0.4305 | **0.4082** | ✅ 50 |
| 方向准确率 | 54.44% | 49.76% | 54.24% | **55.16%** | ✅ 50 |
| R² Score | 0.0122 | -0.0265 | -0.0490 | **0.0458** | ✅ 50 |
| 相关系数 | 0.1428 | 0.1392 | 0.0580 | **0.2241** | ✅ 50 |

## 🚀 推荐配置

```bash
python train.py \
    --model_type csept_smooth \
    --smooth_window_size 50 \
    --smooth_learnable \
    --train_start 3000 \
    --train_end 4000 \
    --test_start 5000 \
    --test_end 6000 \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --early_stopping_patience 30
```

## 🔧 后续实验

### 扩大窗口搜索

```bash
cd /SeqPred/scripts
python experiment_smooth_window.py --window_sizes 50 70 100 120 150
```

### 精细搜索

```bash
python experiment_smooth_window.py --window_sizes 40 45 50 55 60 65
```

### 解决过拟合

```bash
# 减小模型
python train.py --smooth_window_size 50 --hidden_size 64 --num_layers 2

# 增加正则化
python train.py --smooth_window_size 50 --dropout 0.3 --weight_decay 0.01

# 增加数据
python train.py --smooth_window_size 50 --train_start 2000 --train_end 5000
```

## 📁 实验结果位置

```
experiments/smooth_window_exp_20251215_150640/
├── comparison_plots.png          # 对比图（6个子图）
├── analysis_report.txt            # 分析报告
├── DETAILED_ANALYSIS.md           # 详细分析
├── results_summary.csv            # CSV结果
├── results.json                   # JSON结果
└── window_XX/                     # 各窗口完整结果
```

## 📖 详细文档

- [实验完整说明](SMOOTH_WINDOW_EXPERIMENT.md)
- [详细分析报告](experiments/smooth_window_exp_20251215_150640/DETAILED_ANALYSIS.md)

## ⚠️ 重要发现

1. **窗口50全面最优** - 所有指标都是最佳
2. **窗口20不可用** - 方向准确率低于随机
3. **存在过拟合** - 训练/测试Loss差距16-21倍
4. **早停过早** - 所有模型第1个epoch就最佳

## 💡 实用技巧

### 查看结果

```bash
# 查看分析报告
cat experiments/smooth_window_exp_20251215_150640/analysis_report.txt

# 用Python分析
python3 << EOF
import pandas as pd
df = pd.read_csv('experiments/smooth_window_exp_20251215_150640/results_summary.csv')
print(df.sort_values('best_test_loss'))
EOF
```

### 对比实验

```bash
# 运行多次验证稳定性
for i in {1..3}; do
    python experiment_smooth_window.py --window_sizes 40 50 60
done
```

---

**最后更新**: 2025-12-15  
**实验耗时**: 约2分钟  
**训练模型数**: 4个

