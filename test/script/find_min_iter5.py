#!/usr/bin/env python3
"""找出iter5阶段耗时最少的5个文档"""

import csv
from pathlib import Path

# 读取CSV文件
csv_file = Path("/root/project/srs/srs-gen2/output/sample10_iter5/耗时统计-总耗时.csv")

with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)

# 第一行是列名（阶段名称 + 文档名）
headers = rows[0]
doc_names = headers[1:]  # 去掉第一个"阶段名称"

# 找到iter5行
iter5_row = None
for row in rows:
    if row[0] == 'iter5':
        iter5_row = row
        break

if not iter5_row:
    print("未找到iter5阶段数据")
    exit(1)

# 提取耗时数据（跳过第一个"iter5"列名）
time_values = iter5_row[1:]

# 将文档名和耗时配对，并过滤掉空值
doc_times = []
for doc_name, time_str in zip(doc_names, time_values):
    if time_str and time_str.strip():  # 过滤空值
        try:
            time_value = float(time_str)
            doc_times.append((doc_name, time_value))
        except ValueError:
            continue

# 按耗时排序，找出最少的5个
doc_times_sorted = sorted(doc_times, key=lambda x: x[1])
top5 = doc_times_sorted[:5]

# 输出结果
print("iter5阶段耗时最少的5个文档：")
print("-" * 80)
for i, (doc_name, time_value) in enumerate(top5, 1):
    print(f"{i}. {doc_name:50s} 耗时: {time_value:.2f} 秒")

