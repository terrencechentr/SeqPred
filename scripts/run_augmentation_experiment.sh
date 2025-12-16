#!/bin/bash

# 数据增强方法影响实验脚本
# 运行多个实验，对比不同增强方法的效果

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}数据增强方法影响实验${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到python3${NC}"
    exit 1
fi

# 检查数据文件
DATA_FILE="$PROJECT_DIR/data/data.csv"
if [ ! -f "$DATA_FILE" ]; then
    echo -e "${RED}错误: 数据文件不存在: $DATA_FILE${NC}"
    exit 1
fi

# 检查实验脚本
EXP_SCRIPT="$SCRIPT_DIR/experiment_augmentation.py"
if [ ! -f "$EXP_SCRIPT" ]; then
    echo -e "${RED}错误: 实验脚本不存在: $EXP_SCRIPT${NC}"
    exit 1
fi

# 运行实验
echo -e "${GREEN}开始运行实验...${NC}"
echo ""

cd "$PROJECT_DIR"
python3 "$EXP_SCRIPT"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}实验完成！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "实验结果保存在: experiments/augmentation_experiment_*/"
    echo ""
    echo "生成的文件："
    echo "  - results.csv: 实验结果汇总（CSV格式）"
    echo "  - results.json: 实验结果汇总（JSON格式）"
    echo "  - augmentation_comparison.png: 对比柱状图"
    echo "  - augmentation_heatmap.png: 热力图"
    echo "  - experiment_report.md: 实验报告"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}实验失败！${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

