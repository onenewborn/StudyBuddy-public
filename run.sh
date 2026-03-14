#!/bin/bash
# StudyBuddy 快捷运行脚本
# 使用 DeepTutor 的 venv（已有全部依赖）
# 用法：
#   ./run.sh data/test_data/run_test.py          # 完整测试
#   ./run.sh src/agents/homework/ocr_agent.py /tmp/hw.jpg math   # OCR 调试

VENV_PYTHON="/Users/mona/DeepTutor/venv/bin/python"
SCRIPT="$1"
shift

if [ -z "$SCRIPT" ]; then
    echo "用法: ./run.sh <脚本路径> [参数...]"
    echo "示例: ./run.sh data/test_data/run_test.py"
    exit 1
fi

cd "$(dirname "$0")"
exec "$VENV_PYTHON" "$SCRIPT" "$@"
