#!/usr/bin/env python3
"""
顶层测试/评估入口，兼容文档与脚本调用
"""
import os
import sys

# 支持源码直接运行（无需先安装）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from seqpred.eval.evaluator import main


if __name__ == "__main__":
    main()

