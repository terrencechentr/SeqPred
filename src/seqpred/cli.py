#!/usr/bin/env python3
"""
统一命令行入口，集中管理训练与测试/预测接口
"""

import argparse
import os
import sys

# 支持直接运行脚本（无需提前安装）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from seqpred.train import build_train_parser, run_train
from seqpred.eval import build_eval_parser, run_eval


def build_cli_parser() -> argparse.ArgumentParser:
    """
    构建顶层CLI解析器，复用子模块的参数定义，确保所有模型一致的入口体验
    """
    parser = argparse.ArgumentParser(
        description="SeqPred 统一命令行接口（train/test）",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="子命令")

    # 训练子命令
    train_parent = build_train_parser(add_help=False)
    subparsers.add_parser(
        "train",
        parents=[train_parent],
        add_help=True,
        help="训练模型（支持 csept / csept_smooth / sept）",
    )

    # 测试/评估子命令
    eval_parent = build_eval_parser(add_help=False)
    subparsers.add_parser(
        "test",
        parents=[eval_parent],
        add_help=True,
        help="评估或自回归预测（支持 csept / csept_smooth / sept）",
    )

    return parser


def main(argv=None):
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        return run_train(args)
    if args.command == "test":
        return run_eval(args)

    raise ValueError(f"未知子命令: {args.command}")


if __name__ == "__main__":
    main()

