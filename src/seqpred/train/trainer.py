#!/usr/bin/env python3
"""
统一训练脚本 - 使用AutoModel和from_pretrained接口
支持所有模型类型: csept, csept_smooth, sept
"""

import argparse
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel

# 导入自定义模型配置和类（触发注册）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import *  # noqa: F401,F403

# 导入数据集
from data.spetDataset import SpetDataset


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def plot_training_curves(train_losses, test_losses, best_epoch, output_path):
    """
    绘制训练和测试损失曲线
    
    Args:
        train_losses: 训练损失列表
        test_losses: 测试损失列表
        best_epoch: 最佳epoch
        output_path: 输出路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = np.arange(1, len(train_losses) + 1)
    
    # 子图1: 训练和测试损失曲线
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, label='Train Loss', linewidth=2, marker='o', 
            markersize=4, alpha=0.8, color='blue')
    ax1.plot(epochs, test_losses, label='Test Loss', linewidth=2, marker='s', 
            markersize=4, alpha=0.8, color='red')
    
    # 标记最佳epoch
    ax1.axvline(x=best_epoch+1, color='green', linestyle='--', linewidth=2, 
               label=f'Best Epoch ({best_epoch+1})')
    ax1.scatter([best_epoch+1], [test_losses[best_epoch]], color='green', 
               s=100, zorder=5, marker='*')
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Test Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(train_losses) + 1)
    
    # 子图2: 对数尺度（如果loss变化范围很大）
    ax2 = axes[1]
    ax2.semilogy(epochs, train_losses, label='Train Loss', linewidth=2, marker='o', 
                markersize=4, alpha=0.8, color='blue')
    ax2.semilogy(epochs, test_losses, label='Test Loss', linewidth=2, marker='s', 
                markersize=4, alpha=0.8, color='red')
    
    # 标记最佳epoch
    ax2.axvline(x=best_epoch+1, color='green', linestyle='--', linewidth=2,
               label=f'Best Epoch ({best_epoch+1})')
    ax2.scatter([best_epoch+1], [test_losses[best_epoch]], color='green', 
               s=100, zorder=5, marker='*')
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Test Loss Curves (Log Scale)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(0, len(train_losses) + 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练曲线保存到: {output_path}")
    plt.close()


def save_training_history(train_losses, test_losses, best_epoch, best_test_loss, output_path):
    """
    保存训练历史到文本文件
    
    Args:
        train_losses: 训练损失列表
        test_losses: 测试损失列表
        best_epoch: 最佳epoch
        best_test_loss: 最佳测试损失
        output_path: 输出路径
    """
    with open(output_path, 'w') as f:
        f.write("训练历史\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"最佳Epoch: {best_epoch + 1}\n")
        f.write(f"最佳测试Loss: {best_test_loss:.6f}\n")
        f.write(f"最终训练Loss: {train_losses[-1]:.6f}\n")
        f.write(f"最终测试Loss: {test_losses[-1]:.6f}\n\n")
        
        # 损失统计
        f.write("训练集损失统计:\n")
        f.write(f"  初始: {train_losses[0]:.6f}\n")
        f.write(f"  最终: {train_losses[-1]:.6f}\n")
        f.write(f"  最小: {min(train_losses):.6f} (Epoch {train_losses.index(min(train_losses))+1})\n")
        f.write(f"  平均: {np.mean(train_losses):.6f}\n\n")
        
        f.write("测试集损失统计:\n")
        f.write(f"  初始: {test_losses[0]:.6f}\n")
        f.write(f"  最终: {test_losses[-1]:.6f}\n")
        f.write(f"  最小: {min(test_losses):.6f} (Epoch {test_losses.index(min(test_losses))+1})\n")
        f.write(f"  平均: {np.mean(test_losses):.6f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Epoch-by-Epoch详情:\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Epoch':>6} | {'Train Loss':>12} | {'Test Loss':>12} | {'Note':>20}\n")
        f.write("-" * 80 + "\n")
        
        for i, (train_loss, test_loss) in enumerate(zip(train_losses, test_losses)):
            note = ""
            if i == best_epoch:
                note = "⭐ Best Model"
            elif i == 0:
                note = "Initial"
            elif i == len(train_losses) - 1:
                note = "Final"
            
            f.write(f"{i+1:6d} | {train_loss:12.6f} | {test_loss:12.6f} | {note:>20}\n")
    
    print(f"✓ 训练历史保存到: {output_path}")


def create_config(args):
    """根据model_type创建配置"""
    if args.model_type == "csept":
        from models import CseptConfig
        config = CseptConfig(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            num_key_value_heads=args.num_heads,
            intermediate_size=args.intermediate_size,
            loss_type=args.loss_type,
        )
    elif args.model_type == "csept_smooth":
        from models import CseptSmoothConfig
        config = CseptSmoothConfig(
            input_size=args.input_size,
            smooth_window_size=args.smooth_window_size,
            smooth_learnable=args.smooth_learnable,
            unsmooth_mode=args.unsmooth_mode,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            num_key_value_heads=args.num_heads,
            intermediate_size=args.intermediate_size,
            loss_type=args.loss_type,
        )
    elif args.model_type == "sept":
        from models import SeptConfig
        config = SeptConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            num_key_value_heads=args.num_heads,
            intermediate_size=args.intermediate_size,
            classifier_dropout=0.1,
        )
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    return config


def train_model(args, device):
    """训练模型"""
    print("=" * 80)
    print("统一训练脚本 - 使用AutoModel接口")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  - 模型类型: {args.model_type}")
    print(f"  - 输出目录: {args.output_dir}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Learning Rate: {args.learning_rate}")
    print(f"  - 早停耐心值: {args.early_stopping_patience}")
    print()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 准备数据
    train_dataset = SpetDataset(
        args.data_path, 
        args.seq_length, 
        is_train=True, 
        noise_level=args.noise_level,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        apply_smoothing=args.dataset_apply_smoothing,
        smooth_window_size=args.dataset_smooth_window_size,
        smooth_target_features=args.dataset_smooth_target_features,
    )
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
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"✓ 训练集: {len(train_dataset)} 样本")
    print(f"✓ 测试集: {len(test_dataset)} 样本")
    print()
    
    # 创建或加载模型
    if args.pretrained_model_path:
        print(f"✓ 从预训练模型加载: {args.pretrained_model_path}")
        # 使用AutoModel.from_pretrained加载
        model = AutoModel.from_pretrained(args.pretrained_model_path).to(device)
        config = model.config
    else:
        print(f"✓ 创建新模型")
        # 创建配置
        config = create_config(args)
        # 保存配置
        config.save_pretrained(args.output_dir)
        
        # 使用AutoModel.from_config创建
        model = AutoModel.from_config(config).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    early_stopper = EarlyStopper(patience=args.early_stopping_patience)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型参数量: {param_count:,}")
    print()
    
    # 训练
    print("开始训练...")
    print("-" * 80)
    
    best_test_loss = float('inf')
    best_epoch = 0
    train_losses = []
    test_losses = []
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for input_values, labels in train_loader:
            input_values, labels = input_values.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # 统一接口：所有模型都接受input_values和labels
            if args.model_type == "sept":
                # Sept模型使用不同的输入格式
                output = model(input_ids=input_values.long(), labels=labels)
            else:
                output = model(input_values=input_values, labels=labels)
            
            loss = output.loss if hasattr(output, 'loss') else output['loss']
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 测试阶段
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for input_values, labels in test_loader:
                input_values, labels = input_values.to(device), labels.to(device)
                
                if args.model_type == "sept":
                    output = model(input_ids=input_values.long(), labels=labels)
                else:
                    output = model(input_values=input_values, labels=labels)
                
                loss = output.loss if hasattr(output, 'loss') else output['loss']
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            
            # 使用save_pretrained统一保存
            model.save_pretrained(f'{args.output_dir}/best_model')
            print(f"✓ 保存最佳模型到: {args.output_dir}/best_model")
        
        # 每5个epoch打印一次
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Test Loss: {test_loss:.6f} | "
                  f"Best: {best_test_loss:.6f}")
        
        # 早停检查
        early_stopper(test_loss)
        if early_stopper.early_stop:
            print(f"\n早停触发于 Epoch {epoch+1}")
            break
    
    print("-" * 80)
    print(f"\n训练完成！")
    print(f"  - 最佳Epoch: {best_epoch+1}")
    print(f"  - 最佳测试Loss: {best_test_loss:.6f}")
    print(f"  - 最终训练Loss: {train_losses[-1]:.6f}")
    print(f"  - 最终测试Loss: {test_losses[-1]:.6f}")
    print(f"  - 模型保存路径: {args.output_dir}/best_model")
    print()
    
    # 绘制训练曲线
    print("生成训练可视化...")
    plot_path = os.path.join(args.output_dir, 'training_curves.png')
    plot_training_curves(train_losses, test_losses, best_epoch, plot_path)
    
    # 保存训练历史
    history_path = os.path.join(args.output_dir, 'training_history.txt')
    save_training_history(train_losses, test_losses, best_epoch, best_test_loss, history_path)
    print()
    
    return model, config


def build_train_parser(add_help: bool = False) -> argparse.ArgumentParser:
    """
    构建训练参数解析器，供脚本与统一CLI复用
    """
    parser = argparse.ArgumentParser(
        description="统一训练脚本 - 使用AutoModel",
        add_help=add_help,
    )

    # 模型参数
    parser.add_argument(
        '--model_type',
        type=str,
        required=True,
        choices=['csept', 'csept_smooth', 'sept'],
        help='模型类型',
    )
    parser.add_argument(
        '--pretrained_model_path',
        type=str,
        default=None,
        help='预训练模型路径（可选）',
    )

    # 数据参数
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='数据文件路径（默认: src/data/data.csv）',
    )
    parser.add_argument(
        '--seq_length',
        type=int,
        default=256,
        help='序列长度',
    )
    parser.add_argument(
        '--noise_level',
        type=float,
        default=0.001,
        help='训练噪声水平',
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

    # 数据区间参数
    parser.add_argument(
        '--train_start',
        type=int,
        default=3000,
        help='训练集起始索引',
    )
    parser.add_argument(
        '--train_end',
        type=int,
        default=4000,
        help='训练集结束索引',
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

    # 模型配置参数
    parser.add_argument(
        '--input_size',
        type=int,
        default=5,
        help='输入特征维度',
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=128,
        help='隐藏层维度',
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=4,
        help='Transformer层数',
    )
    parser.add_argument(
        '--num_heads',
        type=int,
        default=4,
        help='注意力头数',
    )
    parser.add_argument(
        '--intermediate_size',
        type=int,
        default=512,
        help='FFN中间层维度',
    )
    parser.add_argument(
        '--loss_type',
        type=str,
        default='mse',
        choices=['mse', 'mae', 'smooth_l1', 'huber', 'log_cosh'],
        help='损失函数类型',
    )

    # CSEPT Smooth特有参数
    parser.add_argument(
        '--smooth_window_size',
        type=int,
        default=50,
        help='平滑窗口大小',
    )
    parser.add_argument(
        '--smooth_learnable',
        action='store_true',
        help='是否使用可学习的平滑权重',
    )
    parser.add_argument(
        '--unsmooth_mode',
        type=str,
        default='identity',
        choices=['identity', 'learnable', 'inverse'],
        help='恢复模式',
    )

    # 训练参数
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='训练轮数',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='批次大小',
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='学习率',
    )
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=10,
        help='早停耐心值',
    )

    # 输出参数
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录',
    )

    return parser


def run_train(args):
    """
    根据解析好的参数执行训练，便于CLI与脚本复用
    """
    if args.data_path is None:
        args.data_path = os.path.join(os.path.dirname(__file__), "..", "data", "data.csv")

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
        args.output_dir = os.path.join(base_dir, "experiments", f"{args.model_type}_training_{timestamp}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    return train_model(args, device)


def main(argv=None):
    parser = build_train_parser(add_help=True)
    args = parser.parse_args(argv)
    run_train(args)


if __name__ == "__main__":
    main()

