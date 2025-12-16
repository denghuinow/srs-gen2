#!/usr/bin/env python3
"""
为所有测试的权重配置分别生成数据汇总v2.xlsx文件
"""
import sys
from pathlib import Path

# 导入generate_weighted_summary中的函数
sys.path.insert(0, str(Path(__file__).parent))
from generate_weighted_summary import (
    generate_summary_workbook_with_weights,
    CATEGORIES,
)

# 所有测试的权重配置
WEIGHT_CONFIGS = [
    {
        "name": "当前权重",
        "filename": "数据汇总v2_当前权重.xlsx",
        "weights": {
            "FUNCTIONAL": 0.25,
            "BUSINESS_FLOW": 0.15,
            "BOUNDARY": 0.10,
            "EXCEPTION": 0.20,
            "DATA_STATE": 0.20,
            "CONSISTENCY_RULE": 0.10,
        }
    },
    {
        "name": "方案1_高FUNCTIONAL_低BOUNDARY",
        "filename": "数据汇总v2_方案1_高FUNCTIONAL_低BOUNDARY.xlsx",
        "weights": {
            "FUNCTIONAL": 0.60,
            "BUSINESS_FLOW": 0.10,
            "BOUNDARY": 0.02,
            "EXCEPTION": 0.12,
            "DATA_STATE": 0.10,
            "CONSISTENCY_RULE": 0.06,
        }
    },
    {
        "name": "方案2_平衡FUNCTIONAL_EXCEPTION_DATA_STATE",
        "filename": "数据汇总v2_方案2_平衡FUNCTIONAL_EXCEPTION_DATA_STATE.xlsx",
        "weights": {
            "FUNCTIONAL": 0.50,
            "BUSINESS_FLOW": 0.08,
            "BOUNDARY": 0.02,
            "EXCEPTION": 0.20,
            "DATA_STATE": 0.15,
            "CONSISTENCY_RULE": 0.05,
        }
    },
    {
        "name": "方案3_极高FUNCTIONAL",
        "filename": "数据汇总v2_方案3_极高FUNCTIONAL.xlsx",
        "weights": {
            "FUNCTIONAL": 0.85,
            "BUSINESS_FLOW": 0.05,
            "BOUNDARY": 0.01,
            "EXCEPTION": 0.05,
            "DATA_STATE": 0.03,
            "CONSISTENCY_RULE": 0.01,
        }
    },
    {
        "name": "方案4_FUNCTIONAL_EXCEPTION主导",
        "filename": "数据汇总v2_方案4_FUNCTIONAL_EXCEPTION主导.xlsx",
        "weights": {
            "FUNCTIONAL": 0.55,
            "BUSINESS_FLOW": 0.05,
            "BOUNDARY": 0.02,
            "EXCEPTION": 0.25,
            "DATA_STATE": 0.10,
            "CONSISTENCY_RULE": 0.03,
        }
    },
    {
        "name": "方案5_FUNCTIONAL_DATA_STATE主导",
        "filename": "数据汇总v2_方案5_FUNCTIONAL_DATA_STATE主导.xlsx",
        "weights": {
            "FUNCTIONAL": 0.55,
            "BUSINESS_FLOW": 0.05,
            "BOUNDARY": 0.02,
            "EXCEPTION": 0.10,
            "DATA_STATE": 0.25,
            "CONSISTENCY_RULE": 0.03,
        }
    },
    {
        "name": "最优配置_自动搜索",
        "filename": "数据汇总v2_最优配置_自动搜索.xlsx",
        "weights": {
            "FUNCTIONAL": 0.618,
            "BUSINESS_FLOW": 0.038,
            "BOUNDARY": 0.072,
            "EXCEPTION": 0.066,
            "DATA_STATE": 0.071,
            "CONSISTENCY_RULE": 0.136,
        }
    },
]


def main():
    """为所有权重配置生成Excel汇总文件"""
    
    # 路径配置
    eval_root = Path("/root/project/srs/srs-gen2/eval_reports/en_iter10")
    doc_eval_root = eval_root / "documents"
    unit_eval_root = eval_root / "units"
    outputs_dir = Path("/root/project/srs/srs-gen2/output/en_iter10")
    
    print("="*80)
    print("批量生成权重配置数据汇总Excel")
    print("="*80)
    print(f"\n将生成 {len(WEIGHT_CONFIGS)} 个Excel文件\n")
    
    generated_files = []
    
    for idx, config in enumerate(WEIGHT_CONFIGS, 1):
        print(f"[{idx}/{len(WEIGHT_CONFIGS)}] 生成: {config['name']}")
        print(f"  权重配置: {config['weights']}")
        
        summary_path = eval_root / config['filename']
        
        try:
            generate_summary_workbook_with_weights(
                summary_path,
                doc_eval_root,
                unit_eval_root,
                outputs_dir,
                config['weights'],
                config['name'],
            )
            generated_files.append(summary_path)
            print(f"  ✓ 已生成: {summary_path.name}\n")
        except Exception as e:
            print(f"  ✗ 生成失败: {e}\n")
            import traceback
            traceback.print_exc()
    
    # 总结
    print("="*80)
    print("生成完成")
    print("="*80)
    print(f"\n成功生成 {len(generated_files)} 个文件:")
    for file_path in generated_files:
        print(f"  - {file_path.name}")
    
    if len(generated_files) < len(WEIGHT_CONFIGS):
        print(f"\n警告: {len(WEIGHT_CONFIGS) - len(generated_files)} 个文件生成失败")


if __name__ == "__main__":
    main()



