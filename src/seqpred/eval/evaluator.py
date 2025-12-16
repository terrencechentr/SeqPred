#!/usr/bin/env python3
"""
统一评估脚本 - 使用AutoModel.from_pretrained加载模型
支持所有模型类型: csept, csept_smooth, sept
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader
from transformers import AutoModel

# 导入自定义模型配置和类（触发注册）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import *  # noqa: F401,F403

# 导入数据集
from data.spetDataset import SpetDataset


def evaluate_model(model, test_loader, device, model_type):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for input_values, labels in test_loader:
            input_values, labels = input_values.to(device), labels.to(device)
            
            # 统一接口
            if model_type == "sept":
                output = model(input_ids=input_values.long(), labels=labels)
            else:
                output = model(input_values=input_values, labels=labels)
            
            loss = output.loss if hasattr(output, 'loss') else output['loss']
            total_loss += loss.item()
            
            # 获取预测结果
            if hasattr(output, 'preds'):
                preds = output.preds
            elif hasattr(output, 'logits'):
                preds = output.logits
            else:
                preds = output['preds']
            
            all_predictions.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return avg_loss, all_predictions, all_labels


def autoregressive_predict(model, initial_input, predict_steps, device, model_type, initial_length, noise_std: float = 0.0):
    """
    自回归预测
    
    Args:
        model: 已加载模型
        initial_input: (1, seq_length, features) 的初始输入
        predict_steps: 需要预测的步数 (= seq_length - initial_length)
        device: 设备
        model_type: 模型类型
        initial_length: 初始上下文长度 (< seq_length)
        noise_std: 预测时添加的高斯噪声标准差，0为不加噪声
    """
    model.eval()
    predictions = []
    
    # 将未观测部分置零，避免未来信息泄漏
    current_input = initial_input.clone()
    if initial_length < current_input.shape[1]:
        current_input[:, initial_length:, :] = 0.0
    
    with torch.no_grad():
        for _ in range(predict_steps):
            # 预测下一步
            if model_type == "sept":
                output = model(input_ids=current_input.long())
            else:
                output = model(input_values=current_input)
            
            # 获取预测结果
            if hasattr(output, 'preds'):
                pred = output.preds[:, -1:, :]  # (batch, 1, features)
            elif hasattr(output, 'logits'):
                pred = output.logits[:, -1:, :]
            else:
                pred = output['preds'][:, -1:, :]
            
            # 添加噪声以增强鲁棒性
            if noise_std > 0:
                pred = pred + torch.randn_like(pred) * noise_std
            
            # 只保存第一个特征（Close价格）用于可视化
            predictions.append(pred[:, :, 0:1].cpu().numpy())
            
            # 更新输入（滑动窗口，长度保持为seq_length）
            new_features = pred.repeat(1, 1, current_input.shape[2])  # (batch, 1, 5)
            current_input = torch.cat([current_input[:, 1:, :], new_features], dim=1)
    
    return np.concatenate(predictions, axis=1)


def get_ground_truth_sequence(test_dataset, start_idx, length):
    """获取指定位置的真实值序列（标准化的收益率）"""
    ground_truth = []
    for i in range(length):
        if start_idx + i < len(test_dataset):
            _, labels = test_dataset[start_idx + i]
            # 取第一个时间步的第一个特征（Close价格）
            ground_truth.append(labels[0, 0].item())
        else:
            break
    return np.array(ground_truth)


def convert_returns_to_prices(normalized_returns, initial_price, mean, std):
    """
    将标准化的收益率序列转换回绝对价格
    
    Args:
        normalized_returns: 标准化的收益率序列 (n_steps,)
        initial_price: 起始价格
        mean: 训练集收益率的均值
        std: 训练集收益率的标准差
    
    Returns:
        prices: 绝对价格序列 (n_steps+1,), 包含初始价格
    """
    # 1. 反标准化：从标准化的收益率恢复到原始收益率
    original_returns = normalized_returns * std + mean
    
    # 2. 从收益率恢复价格
    # return = (price_t - price_{t-1}) / price_{t-1}
    # => price_t = price_{t-1} * (1 + return)
    prices = [initial_price]
    current_price = initial_price
    
    for ret in original_returns:
        current_price = current_price * (1 + ret)
        prices.append(current_price)
    
    return np.array(prices)


def get_ground_truth_prices(test_dataset, start_idx, length):
    """获取指定位置的真实价格序列"""
    # 获取起始价格
    if hasattr(test_dataset, 'original_close_prices_test'):
        prices = test_dataset.original_close_prices_test
        if start_idx < len(prices):
            # 返回从start_idx开始的length+1个价格（包含起始价格）
            end_idx = min(start_idx + length + 1, len(prices))
            return prices[start_idx:end_idx]
    return None


def plot_autoregressive_with_ground_truth(predictions_list, ground_truth_list, labels_list, output_path, 
                                         initial_length, pred_prices_list=None, truth_prices_list=None):
    """
    绘制多个区间的自回归预测与真实值对比
    
    Args:
        predictions_list: 标准化收益率预测全序列列表（长度=seq_length）
        ground_truth_list: 标准化收益率真实值全序列列表（长度=seq_length）
        labels_list: 标签列表
        output_path: 输出路径
        pred_prices_list: 绝对价格预测列表（用于绘图，可选）
        truth_prices_list: 绝对价格真实值列表（用于绘图，可选）
        initial_length: 初始上下文长度
    """
    n_intervals = len(predictions_list)
    
    # 创建子图
    fig, axes = plt.subplots(n_intervals, 1, figsize=(15, 5 * n_intervals))
    if n_intervals == 1:
        axes = [axes]
    
    # 判断是否使用绝对价格绘图
    use_absolute_price = (pred_prices_list is not None and truth_prices_list is not None)
    
    for idx, (pred, truth, label) in enumerate(zip(predictions_list, ground_truth_list, labels_list)):
        ax = axes[idx]
        
        # 选择绘图数据（绝对价格或标准化收益率）
        if use_absolute_price:
            pred_plot = pred_prices_list[idx]
            truth_plot = truth_prices_list[idx]
            ylabel = 'Price (Absolute)'
            # 价格序列比收益率序列多1个元素（包含初始价格）
            steps_plot = min(len(pred_plot), len(truth_plot))
            x_plot = np.arange(steps_plot)
        else:
            pred_plot = pred
            truth_plot = truth
            ylabel = 'Normalized Returns'
            steps_plot = min(len(pred), len(truth))
            x_plot = np.arange(steps_plot)
        
        # 绘制真实值和预测值
        ax.plot(x_plot, truth_plot[:steps_plot], label='Ground Truth', 
               linewidth=2, alpha=0.8, color='blue')
        ax.plot(x_plot, pred_plot[:steps_plot], label='Prediction', 
               linewidth=2, alpha=0.8, color='red', linestyle='--')
        
        # 标注初始上下文分界
        ax.axvline(x=initial_length, color='green', linestyle='--', alpha=0.7, label='Initial Context')
        
        # 计算该区间的误差（使用标准化收益率）
        steps = min(len(pred), len(truth))
        pred_part = pred[initial_length:steps]
        truth_part = truth[initial_length:steps]
        mae = np.mean(np.abs(pred_part - truth_part))
        rmse = np.sqrt(np.mean((pred_part - truth_part) ** 2))
        
        # 计算方向准确率
        pred_dir = (pred_part[1:] > pred_part[:-1]).astype(float)
        truth_dir = (truth_part[1:] > truth_part[:-1]).astype(float)
        dir_acc = np.mean(pred_dir == truth_dir) * 100
        
        # 如果使用绝对价格，也计算价格的MAPE
        if use_absolute_price and steps_plot > 1:
            price_pred_part = pred_plot[initial_length:steps_plot]
            price_truth_part = truth_plot[initial_length:steps_plot]
            price_mape = np.mean(np.abs((price_truth_part - price_pred_part) / 
                                       (price_truth_part + 1e-10))) * 100
            title_text = (f'{label}\n'
                         f'MAE: {mae:.4f} | RMSE: {rmse:.4f} | Dir Acc: {dir_acc:.2f}% | '
                         f'Price MAPE: {price_mape:.2f}%')
        else:
            title_text = f'{label}\nMAE: {mae:.4f} | RMSE: {rmse:.4f} | Dir Acc: {dir_acc:.2f}%'
        
        ax.set_xlabel('Time Steps', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title_text, fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 添加误差区域
        ax.fill_between(x_plot, pred_plot[:steps_plot], truth_plot[:steps_plot], 
                       alpha=0.2, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 多区间自回归预测图保存到: {output_path}")
    plt.close()


def plot_predictions(predictions, labels, output_path):
    """绘制预测结果 - 改进版，多子图展示"""
    fig = plt.figure(figsize=(18, 12))
    
    # 取第一个样本的第一个特征
    pred_sample = predictions[0, :, 0]
    label_sample = labels[0, :, 0]
    
    # 1. 时序对比图
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(label_sample, label='True Values', linewidth=2, alpha=0.8, color='blue')
    ax1.plot(pred_sample, label='Predictions', linewidth=2, alpha=0.8, color='red', linestyle='--')
    ax1.fill_between(range(len(pred_sample)), pred_sample, label_sample, alpha=0.2, color='gray')
    ax1.set_xlabel('Time Steps', fontsize=11)
    ax1.set_ylabel('Normalized Returns', fontsize=11)
    ax1.set_title('Predictions vs True Values (Time Series)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 散点图（预测值 vs 真实值）
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(label_sample, pred_sample, alpha=0.5, s=20)
    
    # 添加对角线（完美预测线）
    min_val = min(label_sample.min(), pred_sample.min())
    max_val = max(label_sample.max(), pred_sample.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # 计算R²
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(label_sample, pred_sample)
    ax2.plot(label_sample, slope * label_sample + intercept, 'g-', linewidth=2, 
            label=f'Fit: y={slope:.3f}x+{intercept:.3f}')
    
    ax2.set_xlabel('True Values', fontsize=11)
    ax2.set_ylabel('Predictions', fontsize=11)
    ax2.set_title(f'Scatter Plot (R²={r_value**2:.4f})', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. 误差分布直方图
    ax3 = plt.subplot(2, 2, 3)
    errors = pred_sample - label_sample
    ax3.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax3.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, 
               label=f'Mean Error: {np.mean(errors):.4f}')
    ax3.set_xlabel('Prediction Error', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 累积误差曲线
    ax4 = plt.subplot(2, 2, 4)
    cumulative_error = np.cumsum(np.abs(errors))
    ax4.plot(cumulative_error, linewidth=2, color='purple')
    ax4.set_xlabel('Time Steps', fontsize=11)
    ax4.set_ylabel('Cumulative Absolute Error', fontsize=11)
    ax4.set_title('Cumulative Absolute Error Over Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.fill_between(range(len(cumulative_error)), cumulative_error, alpha=0.3, color='purple')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 预测对比图保存到: {output_path}")
    plt.close()


def calculate_metrics(predictions, labels):
    """计算评估指标"""
    # 展平为一维
    pred_flat = predictions.flatten()
    label_flat = labels.flatten()
    
    # 基础误差指标
    mse = np.mean((pred_flat - label_flat) ** 2)
    mae = np.mean(np.abs(pred_flat - label_flat))
    rmse = np.sqrt(mse)
    
    # 中位数绝对误差
    median_ae = np.median(np.abs(pred_flat - label_flat))
    
    # 最大误差
    max_error = np.max(np.abs(pred_flat - label_flat))
    
    # R² 决定系数
    ss_res = np.sum((label_flat - pred_flat) ** 2)
    ss_tot = np.sum((label_flat - np.mean(label_flat)) ** 2)
    r2_score = 1 - (ss_res / (ss_tot + 1e-10))
    
    # MAPE (平均绝对百分比误差) - 避免除以0
    non_zero_mask = np.abs(label_flat) > 1e-10
    mape = np.mean(np.abs((label_flat[non_zero_mask] - pred_flat[non_zero_mask]) / 
                          (label_flat[non_zero_mask] + 1e-10))) * 100
    
    # 相关系数
    correlation = np.corrcoef(pred_flat, label_flat)[0, 1]
    
    # 方向准确率（总体）
    pred_direction = (pred_flat > 0).astype(float)
    label_direction = (label_flat > 0).astype(float)
    direction_accuracy = np.mean(pred_direction == label_direction) * 100
    
    # 上涨和下跌的准确率
    up_mask = label_flat > 0
    down_mask = label_flat < 0
    up_accuracy = np.mean(pred_direction[up_mask] == label_direction[up_mask]) * 100 if np.sum(up_mask) > 0 else 0
    down_accuracy = np.mean(pred_direction[down_mask] == label_direction[down_mask]) * 100 if np.sum(down_mask) > 0 else 0
    
    # 预测分布统计
    pred_mean = np.mean(pred_flat)
    pred_std = np.std(pred_flat)
    label_mean = np.mean(label_flat)
    label_std = np.std(label_flat)
    
    return {
        '=== 误差指标 ===': None,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Median AE': median_ae,
        'Max Error': max_error,
        'MAPE': mape,
        
        '=== 相关性指标 ===': None,
        'R² Score': r2_score,
        'Correlation': correlation,
        
        '=== 方向准确率 ===': None,
        'Direction Accuracy (总体)': direction_accuracy,
        'Up Direction Accuracy (上涨)': up_accuracy,
        'Down Direction Accuracy (下跌)': down_accuracy,
        
        '=== 分布统计 ===': None,
        'Pred Mean': pred_mean,
        'Pred Std': pred_std,
        'Label Mean': label_mean,
        'Label Std': label_std,
    }
def build_eval_parser(add_help: bool = False) -> argparse.ArgumentParser:
    """
    构建评估/预测参数解析器，供脚本与统一CLI复用
    """
    parser = argparse.ArgumentParser(
        description='统一评估脚本 - 使用AutoModel.from_pretrained',
        add_help=add_help,
    )

    # 模型参数
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='模型路径（使用save_pretrained保存的）',
    )
    parser.add_argument(
        '--model_type',
        type=str,
        required=True,
        choices=['csept', 'csept_smooth', 'sept'],
        help='模型类型',
    )

    # 数据参数
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='数据文件路径（默认: src/data/data.csv）',
    )
    parser.add_argument(
        '--train_start',
        type=int,
        default=3000,
        help='训练集起始索引（用于标准化统计）',
    )
    parser.add_argument(
        '--train_end',
        type=int,
        default=4000,
        help='训练集结束索引（用于标准化统计）',
    )
    parser.add_argument(
        '--test_start',
        type=int,
        default=5000,
        help='测试集起始索引',
    )
    parser.add_argument(
        '--test_end',
        type=int,
        default=6000,
        help='测试集结束索引',
    )
    parser.add_argument(
        '--seq_length',
        type=int,
        default=256,
        help='序列长度',
    )
    parser.add_argument(
        '--dataset_apply_smoothing',
        action='store_true',
        help='是否在数据集阶段进行滑动平均平滑',
    )
    parser.add_argument(
        '--dataset_smooth_window_size',
        type=int,
        default=50,
        help='数据集平滑窗口大小',
    )
    parser.add_argument(
        '--dataset_smooth_target_features',
        type=int,
        nargs='+',
        default=[0],
        help='需要平滑的特征索引列表（默认仅平滑Close，索引0）',
    )

    # 评估参数
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='批次大小',
    )
    parser.add_argument(
        '--autoregressive_steps',
        type=int,
        default=0,
        help='自回归预测步数（若指定initial_length，将被覆盖为 seq_length - initial_length）',
    )
    parser.add_argument(
        '--autoregressive_noise_std',
        type=float,
        default=0.0,
        help='自回归预测时在每一步加入的高斯噪声标准差（0为不加噪声）',
    )
    parser.add_argument(
        '--initial_length',
        type=int,
        default=None,
        help='自回归初始上下文长度，需小于seq_length',
    )
    parser.add_argument(
        '--num_intervals',
        type=int,
        default=3,
        help='自回归预测的区间数量（在测试集不同位置进行预测）',
    )

    # 输出参数
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='是否绘制预测结果',
    )

    return parser


def run_eval(args):
    """
    根据解析好的参数执行评估与预测，便于CLI与脚本复用
    """
    if args.data_path is None:
        args.data_path = os.path.join(os.path.dirname(__file__), "..", "data", "data.csv")

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.model_path), "evaluation")
    os.makedirs(args.output_dir, exist_ok=True)

    # 校验initial_length并计算自回归步数
    if args.initial_length is not None:
        if args.initial_length >= args.seq_length:
            raise ValueError("initial_length 必须小于 seq_length")
        args.autoregressive_steps = args.seq_length - args.initial_length
    else:
        # 如果未提供，则使用默认的predict steps，但仍需要保证>0
        if args.autoregressive_steps <= 0:
            args.initial_length = max(1, args.seq_length // 2)
            args.autoregressive_steps = args.seq_length - args.initial_length
        else:
            # 反推初始长度（尽量靠近一半）
            args.initial_length = args.seq_length - args.autoregressive_steps

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    print("=" * 80)
    print("统一评估脚本 - 使用AutoModel.from_pretrained")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  - 模型路径: {args.model_path}")
    print(f"  - 模型类型: {args.model_type}")
    print(f"  - 输出目录: {args.output_dir}")
    print()

    # 从预训练模型加载
    print("✓ 加载模型...")
    model = AutoModel.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True).to(device)
    print(f"✓ 模型加载成功")
    print(f"✓ 模型配置: {model.config.model_type}")
    print()

    # 准备数据
    test_dataset = SpetDataset(
        args.data_path,
        args.seq_length,
        is_train=False,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        apply_smoothing=args.dataset_apply_smoothing,
        smooth_window_size=args.dataset_smooth_window_size,
        smooth_target_features=args.dataset_smooth_target_features,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"✓ 测试集: {len(test_dataset)} 样本")
    print()

    # 评估模型
    print("开始评估...")
    avg_loss, predictions, labels = evaluate_model(model, test_loader, device, args.model_type)

    print(f"✓ 平均损失: {avg_loss:.6f}")
    print()

    # 计算指标
    metrics = calculate_metrics(predictions, labels)
    print("评估指标:")
    for key, value in metrics.items():
        if value is None:  # 分组标题
            print(f"\n{key}")
        elif 'Accuracy' in key or 'MAPE' in key:
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value:.6f}")
    print()

    # 保存指标到文件
    metrics_file = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("评估指标\n")
        f.write("=" * 50 + "\n\n")
        for key, value in metrics.items():
            if value is None:
                f.write(f"\n{key}\n")
            elif 'Accuracy' in key or 'MAPE' in key:
                f.write(f"  {key}: {value:.2f}%\n")
            else:
                f.write(f"  {key}: {value:.6f}\n")
    print(f"✓ 评估指标保存到: {metrics_file}")

    # 绘制预测结果
    if args.plot:
        plot_path = os.path.join(args.output_dir, 'predictions.png')
        plot_predictions(predictions, labels, plot_path)

    # 自回归预测（多区间）
    if args.autoregressive_steps > 0:
        print(f"\n开始多区间自回归预测 ({args.num_intervals} 个区间, 初始长度 {args.initial_length}, "
              f"预测步数 {args.autoregressive_steps}, 总seq长度 {args.seq_length})...")

        # 计算测试集中均匀分布的起始点
        test_len = len(test_dataset)
        interval_positions = np.linspace(
            0, max(0, test_len - args.autoregressive_steps - 1),
            args.num_intervals, dtype=int
        )

        predictions_list = []
        ground_truth_list = []
        full_predictions_list = []
        full_truth_list = []
        labels_list = []
        pred_prices_list = []
        truth_prices_list = []

        # 获取标准化参数（用于恢复价格）
        mean = test_dataset.mean[0].item()  # Close价格的均值
        std = test_dataset.std[0].item()    # Close价格的标准差

        for i, start_idx in enumerate(interval_positions):
            print(f"  区间 {i+1}/{args.num_intervals}: 起始位置 {start_idx}")

            # 获取初始输入
            initial_input, labels = test_dataset[start_idx]
            initial_input = initial_input.unsqueeze(0).to(device)  # (1, seq_length, features)
            truth_full = labels.squeeze().numpy()  # (seq_length,)

            # 自回归预测（标准化的收益率）
            ar_pred = autoregressive_predict(
                model, initial_input, args.autoregressive_steps, device, args.model_type, args.initial_length,
                noise_std=args.autoregressive_noise_std,
            )  # (1, steps, 1)

            # 组装完整预测序列：前initial_length使用真实值，后半部分使用预测
            pred_full = np.concatenate([truth_full[:args.initial_length], ar_pred[0, :, 0]])
            full_predictions_list.append(pred_full)
            full_truth_list.append(truth_full[:len(pred_full)])

            predictions_list.append(ar_pred[0, :, 0])  # 仅预测段
            ground_truth_list.append(truth_full[args.initial_length:args.initial_length + args.autoregressive_steps])
            labels_list.append(f"Interval {i+1} (Start: {start_idx})")

            # 恢复为绝对价格
            truth_prices = get_ground_truth_prices(test_dataset, start_idx, args.seq_length)
            if truth_prices is not None and len(truth_prices) > 0:
                initial_price = truth_prices[0]
                pred_prices_full = convert_returns_to_prices(pred_full, initial_price, mean, std)
                truth_prices_full = convert_returns_to_prices(truth_full[:len(pred_full)], initial_price, mean, std)
                pred_prices_list.append(pred_prices_full)
                truth_prices_list.append(truth_prices_full)

                # 计算价格层面的指标
                steps_price = min(len(pred_prices_full), len(truth_prices_full))
                price_mae = np.mean(np.abs(pred_prices_full[:steps_price] - truth_prices_full[:steps_price]))
                price_mape = np.mean(np.abs((truth_prices_full[1:steps_price] - pred_prices_full[1:steps_price]) /
                                           (truth_prices_full[1:steps_price] + 1e-10))) * 100
                print(f"    收益率 - MAE: {np.mean(np.abs(ar_pred[0, :, 0] - truth_full[args.initial_length:args.initial_length + args.autoregressive_steps])):.4f}, "
                      f"RMSE: {np.sqrt(np.mean((ar_pred[0, :, 0] - truth_full[args.initial_length:args.initial_length + args.autoregressive_steps]) ** 2)):.4f}")
                print(f"    价格 - MAE: {price_mae:.2f}, MAPE: {price_mape:.2f}%")
            else:
                # 如果没有价格数据，使用收益率指标
                steps = min(len(ar_pred[0, :, 0]), args.autoregressive_steps)
                mae = np.mean(np.abs(ar_pred[0, :steps, 0] - truth_full[args.initial_length:args.initial_length + steps]))
                rmse = np.sqrt(np.mean((ar_pred[0, :steps, 0] - truth_full[args.initial_length:args.initial_length + steps]) ** 2))
                print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        # 保存所有预测结果
        ar_output_path = os.path.join(args.output_dir, 'autoregressive_predictions_multi.npz')
        save_dict = {
            'predictions': np.array(predictions_list),
            'ground_truth': np.array(ground_truth_list),
            'predictions_full': np.array(full_predictions_list),
            'ground_truth_full': np.array(full_truth_list),
            'start_positions': interval_positions,
            'initial_length': args.initial_length,
            'predict_steps': args.autoregressive_steps,
        }
        if pred_prices_list:
            save_dict['pred_prices'] = np.array(pred_prices_list, dtype=object)
            save_dict['truth_prices'] = np.array(truth_prices_list, dtype=object)
        np.savez(ar_output_path, **save_dict)
        print(f"\n✓ 自回归预测结果保存到: {ar_output_path}")

        # 绘制多区间对比图（使用绝对价格）
        if args.plot:
            ar_plot_path = os.path.join(args.output_dir, 'autoregressive_multi_intervals.png')
            plot_autoregressive_with_ground_truth(
                full_predictions_list, full_truth_list, labels_list, ar_plot_path, initial_length=args.initial_length,
                pred_prices_list=pred_prices_list if pred_prices_list else None,
                truth_prices_list=truth_prices_list if truth_prices_list else None
            )

    print()
    print("评估完成！")
    return {
        "loss": avg_loss,
        "metrics": metrics,
        "predictions": predictions,
        "labels": labels,
    }


def main(argv=None):
    parser = build_eval_parser(add_help=True)
    args = parser.parse_args(argv)
    run_eval(args)


if __name__ == "__main__":
    main()

