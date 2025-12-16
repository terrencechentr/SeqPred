#!/usr/bin/env python3
"""
数据增强方法影响实验脚本
运行多个实验，对比不同增强方法的效果，并生成对比报告
"""

import os
import sys
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 实验配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")

# 实验配置
EXPERIMENTS = [
    {
        "name": "baseline",
        "description": "基线（无增强）",
        "args": []
    },
    {
        "name": "noise_only",
        "description": "仅噪声增强",
        "args": ["--noise_level", "0.01"]
    },
    {
        "name": "random_scale",
        "description": "随机缩放",
        "args": ["--noise_level", "0.01", "--aug_random_scale"]
    },
    {
        "name": "feature_dropout",
        "description": "特征Dropout",
        "args": ["--noise_level", "0.01", "--aug_feature_dropout"]
    },
    {
        "name": "time_warp",
        "description": "时间扭曲",
        "args": ["--noise_level", "0.01", "--aug_time_warp"]
    },
    {
        "name": "mixup",
        "description": "Mixup",
        "args": ["--noise_level", "0.01", "--aug_mixup"]
    },
    {
        "name": "scale_dropout",
        "description": "缩放+Dropout",
        "args": ["--noise_level", "0.01", "--aug_random_scale", "--aug_feature_dropout"]
    },
    {
        "name": "scale_warp",
        "description": "缩放+时间扭曲",
        "args": ["--noise_level", "0.01", "--aug_random_scale", "--aug_time_warp"]
    },
    {
        "name": "all_aug",
        "description": "全部增强",
        "args": ["--noise_level", "0.01", "--aug_random_scale", "--aug_feature_dropout", 
                 "--aug_time_warp", "--aug_mixup"]
    },
]

# 训练参数
TRAIN_PARAMS = {
    "model_type": "csept_smooth",
    "train_start": 0,
    "train_end": 4000,
    "test_start": 4000,
    "test_end": 5000,
    "max_steps": 1000,
    "batch_size": 32,
    "smooth_window_size": 50,
    "dataset_apply_smoothing": True,
    "dataset_smooth_window_size": 20,
}

# 评估参数
EVAL_PARAMS = {
    "initial_length": 200,
    "num_intervals": 3,
    "autoregressive_noise_std": 0.01,
    "plot": True,
}


def run_training(exp_name, exp_args):
    """运行训练"""
    print(f"\n{'='*80}")
    print(f"训练实验: {exp_name}")
    print(f"{'='*80}")
    
    output_dir = os.path.join(EXPERIMENTS_DIR, f"aug_exp_{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "train.py"),
        "--model_type", TRAIN_PARAMS["model_type"],
        "--train_start", str(TRAIN_PARAMS["train_start"]),
        "--train_end", str(TRAIN_PARAMS["train_end"]),
        "--test_start", str(TRAIN_PARAMS["test_start"]),
        "--test_end", str(TRAIN_PARAMS["test_end"]),
        "--max_steps", str(TRAIN_PARAMS["max_steps"]),
        "--batch_size", str(TRAIN_PARAMS["batch_size"]),
        "--smooth_window_size", str(TRAIN_PARAMS["smooth_window_size"]),
        "--dataset_apply_smoothing",
        "--dataset_smooth_window_size", str(TRAIN_PARAMS["dataset_smooth_window_size"]),
        "--data_path", DATA_PATH,
        "--output_dir", output_dir,
    ] + exp_args
    
    print(f"执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ 训练失败: {result.stderr}")
        return None
    
    print(f"✓ 训练完成: {output_dir}")
    return output_dir


def run_evaluation(model_path, exp_name):
    """运行评估"""
    print(f"\n评估实验: {exp_name}")
    
    eval_output_dir = os.path.join(model_path, "evaluation")
    
    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "test.py"),
        "--model_path", os.path.join(model_path, "best_model"),
        "--model_type", TRAIN_PARAMS["model_type"],
        "--data_path", DATA_PATH,
        "--train_start", str(TRAIN_PARAMS["train_start"]),
        "--train_end", str(TRAIN_PARAMS["train_end"]),
        "--test_start", str(TRAIN_PARAMS["test_start"]),
        "--test_end", str(TRAIN_PARAMS["test_end"]),
        "--initial_length", str(EVAL_PARAMS["initial_length"]),
        "--num_intervals", str(EVAL_PARAMS["num_intervals"]),
        "--autoregressive_noise_std", str(EVAL_PARAMS["autoregressive_noise_std"]),
        "--plot",
        "--output_dir", eval_output_dir,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ 评估失败: {result.stderr}")
        return None
    
    print(f"✓ 评估完成: {eval_output_dir}")
    return eval_output_dir


def parse_training_results(output_dir):
    """解析训练结果"""
    history_file = os.path.join(output_dir, "training_history.txt")
    if not os.path.exists(history_file):
        return None
    
    results = {}
    with open(history_file, 'r') as f:
        content = f.read()
        
        # 提取关键信息
        for line in content.split('\n'):
            if '最佳Epoch' in line:
                results['best_epoch'] = int(line.split(':')[1].strip())
            elif '最佳测试Loss' in line:
                results['best_test_loss'] = float(line.split(':')[1].strip())
            elif '最终训练Loss' in line:
                results['final_train_loss'] = float(line.split(':')[1].strip())
            elif '最终测试Loss' in line:
                results['final_test_loss'] = float(line.split(':')[1].strip())
    
    return results


def parse_evaluation_results(eval_output_dir):
    """解析评估结果"""
    metrics_file = os.path.join(eval_output_dir, "metrics.txt")
    if not os.path.exists(metrics_file):
        return None
    
    results = {}
    with open(metrics_file, 'r') as f:
        content = f.read()
        
        current_section = None
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if '===' in line:
                current_section = line.strip('= ')
            elif ':' in line and current_section:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # 尝试转换为数值
                try:
                    if '%' in value:
                        value = float(value.replace('%', ''))
                    else:
                        value = float(value)
                    results[key] = value
                except ValueError:
                    pass
    
    return results


def collect_all_results(exp_results):
    """收集所有实验结果"""
    all_results = []
    
    for exp in exp_results:
        if exp['output_dir'] is None:
            continue
        
        result = {
            'experiment': exp['name'],
            'description': exp['description'],
        }
        
        # 训练结果
        train_results = parse_training_results(exp['output_dir'])
        if train_results:
            result.update(train_results)
        
        # 评估结果
        if exp['eval_output_dir']:
            eval_results = parse_evaluation_results(exp['eval_output_dir'])
            if eval_results:
                result.update(eval_results)
        
        all_results.append(result)
    
    return pd.DataFrame(all_results)


def plot_comparison(df, output_dir):
    """绘制对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置样式
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 1. 训练损失对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 最佳测试Loss
    ax = axes[0, 0]
    df_sorted = df.sort_values('best_test_loss', ascending=True)
    ax.barh(range(len(df_sorted)), df_sorted['best_test_loss'], color='steelblue')
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['description'], fontsize=9)
    ax.set_xlabel('最佳测试Loss', fontsize=11, fontweight='bold')
    ax.set_title('最佳测试Loss对比', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # R² Score对比
    ax = axes[0, 1]
    if 'R² Score' in df.columns:
        df_sorted = df.sort_values('R² Score', ascending=False)
        ax.barh(range(len(df_sorted)), df_sorted['R² Score'], color='forestgreen')
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['description'], fontsize=9)
        ax.set_xlabel('R² Score', fontsize=11, fontweight='bold')
        ax.set_title('R² Score对比', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    # 方向准确率对比
    ax = axes[1, 0]
    if 'Direction Accuracy (总体)' in df.columns:
        df_sorted = df.sort_values('Direction Accuracy (总体)', ascending=False)
        ax.barh(range(len(df_sorted)), df_sorted['Direction Accuracy (总体)'], color='coral')
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['description'], fontsize=9)
        ax.set_xlabel('方向准确率 (%)', fontsize=11, fontweight='bold')
        ax.set_title('方向准确率对比', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    # MAE对比
    ax = axes[1, 1]
    if 'MAE' in df.columns:
        df_sorted = df.sort_values('MAE', ascending=True)
        ax.barh(range(len(df_sorted)), df_sorted['MAE'], color='mediumpurple')
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['description'], fontsize=9)
        ax.set_xlabel('MAE', fontsize=11, fontweight='bold')
        ax.set_title('MAE对比', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'augmentation_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 对比图保存到: {os.path.join(output_dir, 'augmentation_comparison.png')}")
    plt.close()
    
    # 2. 详细指标热力图
    if len(df) > 0:
        metrics_cols = ['best_test_loss', 'R² Score', 'MAE', 'RMSE', 'Direction Accuracy (总体)']
        metrics_cols = [col for col in metrics_cols if col in df.columns]
        
        if len(metrics_cols) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 归一化数据用于热力图
            df_heatmap = df[['description'] + metrics_cols].set_index('description')
            
            # 对于需要反向的指标（loss越小越好），取负值
            df_normalized = df_heatmap.copy()
            for col in ['best_test_loss', 'MAE', 'RMSE']:
                if col in df_normalized.columns:
                    df_normalized[col] = -df_normalized[col]  # 反向，使得越大越好
            
            # 标准化到0-1
            df_normalized = (df_normalized - df_normalized.min()) / (df_normalized.max() - df_normalized.min() + 1e-10)
            
            sns.heatmap(df_normalized.T, annot=df_heatmap.T, fmt='.4f', cmap='RdYlGn', 
                       cbar_kws={'label': '归一化分数（越高越好）'}, ax=ax, 
                       linewidths=0.5, linecolor='gray')
            ax.set_title('数据增强方法效果热力图', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('实验方法', fontsize=12, fontweight='bold')
            ax.set_ylabel('评估指标', fontsize=12, fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'augmentation_heatmap.png'), dpi=300, bbox_inches='tight')
            print(f"✓ 热力图保存到: {os.path.join(output_dir, 'augmentation_heatmap.png')}")
            plt.close()


def generate_report(df, output_dir):
    """生成实验报告"""
    report_file = os.path.join(output_dir, "experiment_report.md")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 数据增强方法影响实验报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 实验配置\n\n")
        f.write(f"- 训练区间: {TRAIN_PARAMS['train_start']}-{TRAIN_PARAMS['train_end']}\n")
        f.write(f"- 测试区间: {TRAIN_PARAMS['test_start']}-{TRAIN_PARAMS['test_end']}\n")
        f.write(f"- 训练步数: {TRAIN_PARAMS['max_steps']}\n")
        f.write(f"- 批次大小: {TRAIN_PARAMS['batch_size']}\n\n")
        
        f.write("## 实验结果汇总\n\n")
        # 尝试使用markdown格式，如果失败则使用CSV格式
        try:
            f.write(df.to_markdown(index=False))
        except:
            f.write(df.to_string(index=False))
        f.write("\n\n")
        
        f.write("## 关键发现\n\n")
        
        if 'best_test_loss' in df.columns:
            best_loss_exp = df.loc[df['best_test_loss'].idxmin()]
            f.write(f"- **最佳测试Loss**: {best_loss_exp['description']} ({best_loss_exp['best_test_loss']:.6f})\n")
        
        if 'R² Score' in df.columns:
            best_r2_exp = df.loc[df['R² Score'].idxmax()]
            f.write(f"- **最佳R² Score**: {best_r2_exp['description']} ({best_r2_exp['R² Score']:.6f})\n")
        
        if 'Direction Accuracy (总体)' in df.columns:
            best_acc_exp = df.loc[df['Direction Accuracy (总体)'].idxmax()]
            f.write(f"- **最佳方向准确率**: {best_acc_exp['description']} ({best_acc_exp['Direction Accuracy (总体)']:.2f}%)\n")
        
        if 'MAE' in df.columns:
            best_mae_exp = df.loc[df['MAE'].idxmin()]
            f.write(f"- **最佳MAE**: {best_mae_exp['description']} ({best_mae_exp['MAE']:.6f})\n")
    
    print(f"✓ 实验报告保存到: {report_file}")


def main():
    """主函数"""
    print("="*80)
    print("数据增强方法影响实验")
    print("="*80)
    
    # 创建实验输出目录
    exp_output_dir = os.path.join(EXPERIMENTS_DIR, f"augmentation_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(exp_output_dir, exist_ok=True)
    
    exp_results = []
    
    # 运行所有实验
    for exp in EXPERIMENTS:
        print(f"\n开始实验: {exp['name']} - {exp['description']}")
        
        # 训练
        output_dir = run_training(exp['name'], exp['args'])
        if output_dir is None:
            print(f"⚠️ 实验 {exp['name']} 训练失败，跳过")
            exp_results.append({
                'name': exp['name'],
                'description': exp['description'],
                'output_dir': None,
                'eval_output_dir': None,
            })
            continue
        
        # 评估
        eval_output_dir = run_evaluation(output_dir, exp['name'])
        
        exp_results.append({
            'name': exp['name'],
            'description': exp['description'],
            'output_dir': output_dir,
            'eval_output_dir': eval_output_dir,
        })
    
    # 收集结果
    print("\n" + "="*80)
    print("收集实验结果...")
    print("="*80)
    
    df = collect_all_results(exp_results)
    
    # 保存结果到CSV
    csv_file = os.path.join(exp_output_dir, "results.csv")
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"✓ 结果保存到: {csv_file}")
    
    # 保存结果到JSON
    json_file = os.path.join(exp_output_dir, "results.json")
    df.to_json(json_file, orient='records', indent=2, force_ascii=False)
    print(f"✓ 结果保存到: {json_file}")
    
    # 绘制对比图
    print("\n生成对比图...")
    plot_comparison(df, exp_output_dir)
    
    # 生成报告
    print("\n生成实验报告...")
    generate_report(df, exp_output_dir)
    
    print("\n" + "="*80)
    print("实验完成！")
    print(f"结果目录: {exp_output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

