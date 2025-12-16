#!/usr/bin/env python3
"""
使用最佳权重配置生成minimal_en_iter10的数据汇总Excel文件
"""
import sys
from pathlib import Path

# 导入generate_weighted_summary中的函数
sys.path.insert(0, str(Path(__file__).parent))
from generate_weighted_summary import (
    generate_summary_workbook_with_weights,
    CATEGORIES,
)

# 最佳权重配置（满足10/13个相邻对）
BEST_WEIGHTS = {
    "FUNCTIONAL": 0.4511,
    "BUSINESS_FLOW": 0.1191,
    "BOUNDARY": 0.0756,
    "EXCEPTION": 0.1421,
    "DATA_STATE": 0.0656,
    "CONSISTENCY_RULE": 0.1464,
}

def main():
    # 路径配置
    eval_root = Path("/root/project/srs/srs-gen2/eval_reports/minimal_en_iter10")
    doc_eval_root = eval_root / "documents"
    unit_eval_root = eval_root / "units"
    outputs_dir = Path("/root/project/srs/srs-gen2/output/minimal_en_iter10")
    
    # 生成数据汇总
    summary_path = eval_root / "数据汇总v2_最佳权重.xlsx"
    generate_summary_workbook_with_weights(
        summary_path,
        doc_eval_root,
        unit_eval_root,
        outputs_dir,
        BEST_WEIGHTS,
        "最佳权重配置（满足10/13个相邻对）",
    )
    
    print("\n" + "="*80)
    print("权重配置详情:")
    print("="*80)
    for cat in CATEGORIES:
        print(f"  {cat:20s}: {BEST_WEIGHTS.get(cat, 0.0):.4f}")
    print(f"  总和: {sum(BEST_WEIGHTS.values()):.4f}")
    print(f"\n文件已生成: {summary_path}")


if __name__ == "__main__":
    main()


