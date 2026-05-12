#!/usr/bin/env bash
# 依次执行 micro、ultra_short、short、balanced、detailed 的生成+评估+统计
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "===== 1/5 micro ====="
bash micro.sh

echo "===== 2/5 ultra_short ====="
bash ultra_short.sh

echo "===== 3/5 short ====="
bash short.sh

echo "===== 4/5 balanced ====="
bash balanced.sh

echo "===== 5/5 detailed ====="
bash detailed.sh

echo "===== 全部完成 ====="
