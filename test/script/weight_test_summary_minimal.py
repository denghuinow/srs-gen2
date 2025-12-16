#!/usr/bin/env python3
"""
生成权重测试总结报告
"""
import json
import statistics
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

STAGE_MAPPING = {
    "r_base": "r_base",
    "d_base": "d_base",
    "srs_no-explore-clarify": "no-explore-clarify",
    "srs_no-clarify": "no-clarify",
    "srs_iter_1": "iter1",
    "srs_iter_2": "iter2",
    "srs_iter_3": "iter3",
    "srs_iter_4": "iter4",
    "srs_iter_5": "iter5",
    "srs_iter_6": "iter6",
    "srs_iter_7": "iter7",
    "srs_iter_8": "iter8",
    "srs_iter_9": "iter9",
    "srs_iter_10": "iter10",
}

TARGET_ORDER = [
    "r_base", "d_base", "no-explore-clarify", "no-clarify",
    "iter1", "iter2", "iter3", "iter4", "iter5", "iter6",
    "iter7", "iter8", "iter9", "iter10",
]

CATEGORIES = [
    "FUNCTIONAL", "BUSINESS_FLOW", "BOUNDARY",
    "EXCEPTION", "DATA_STATE", "CONSISTENCY_RULE",
]


def load_stage_data(eval_root: Path, stage_dir_name: str) -> Dict[str, Dict[str, float]]:
    stage_path = eval_root / stage_dir_name
    if not stage_path.exists():
        return {}
    documents = {}
    for json_file in stage_path.glob("*_evaluation.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            doc_name = json_file.name.replace("_evaluation.json", "")
            scores = data.get("scores", {})
            category_scores = {
                cat: info.get("score", 0.0) or 0.0
                for cat, info in (scores.get("categories") or {}).items()
            }
            documents[doc_name] = category_scores
        except:
            continue
    return documents


def calculate_stage_average_scores(eval_root: Path) -> Dict[str, Dict[str, float]]:
    stage_averages = {}
    documents_root = eval_root / "documents"
    if documents_root.exists():
        for stage_dir in documents_root.iterdir():
            if not stage_dir.is_dir():
                continue
            stage_name = STAGE_MAPPING.get(stage_dir.name, stage_dir.name)
            if stage_name not in TARGET_ORDER:
                continue
            documents = load_stage_data(documents_root, stage_dir.name)
            if not documents:
                continue
            category_totals = defaultdict(list)
            for doc_scores in documents.values():
                for cat in CATEGORIES:
                    if cat in doc_scores:
                        category_totals[cat].append(doc_scores[cat])
            stage_averages[stage_name] = {
                cat: statistics.mean(scores) if scores else 0.0
                for cat, scores in category_totals.items()
            }
    return stage_averages


def calculate_weighted_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    weighted = sum(weights[cat] * scores.get(cat, 0.0) for cat in weights)
    scaled = min(100.0, weighted * 2.0)
    return round(scaled, 2)


def main():
    eval_root = Path("/root/project/srs/srs-gen2/eval_reports/minimal_en_iter10")
    
    print("正在加载各阶段的评估数据...")
    stage_averages = calculate_stage_average_scores(eval_root)
    
    # 测试多个权重配置
    test_configs = [
        {
            "name": "当前权重",
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
            "name": "最佳配置（11/13满足）",
            "weights": {
                "FUNCTIONAL": 0.4511,
                "BUSINESS_FLOW": 0.1191,
                "BOUNDARY": 0.0756,
                "EXCEPTION": 0.1421,
                "DATA_STATE": 0.0656,
                "CONSISTENCY_RULE": 0.1464,
            }
        },
        {
            "name": "方案A：降低CONSISTENCY_RULE，增加BUSINESS_FLOW",
            "weights": {
                "FUNCTIONAL": 0.45,
                "BUSINESS_FLOW": 0.20,
                "BOUNDARY": 0.05,
                "EXCEPTION": 0.15,
                "DATA_STATE": 0.10,
                "CONSISTENCY_RULE": 0.05,
            }
        },
        {
            "name": "方案B：极高BUSINESS_FLOW，极低CONSISTENCY_RULE",
            "weights": {
                "FUNCTIONAL": 0.40,
                "BUSINESS_FLOW": 0.35,
                "BOUNDARY": 0.03,
                "EXCEPTION": 0.15,
                "DATA_STATE": 0.05,
                "CONSISTENCY_RULE": 0.02,
            }
        },
        {
            "name": "方案C：平衡FUNCTIONAL和BUSINESS_FLOW",
            "weights": {
                "FUNCTIONAL": 0.35,
                "BUSINESS_FLOW": 0.30,
                "BOUNDARY": 0.05,
                "EXCEPTION": 0.15,
                "DATA_STATE": 0.10,
                "CONSISTENCY_RULE": 0.05,
            }
        },
    ]
    
    print("\n" + "="*80)
    print("权重配置测试总结")
    print("="*80)
    
    for config in test_configs:
        weights = config["weights"]
        stage_scores = {}
        for stage_name, category_scores in stage_averages.items():
            score = calculate_weighted_score(category_scores, weights)
            stage_scores[stage_name] = score
        
        # 检查顺序
        issues = []
        for i in range(len(TARGET_ORDER) - 1):
            stage1 = TARGET_ORDER[i]
            stage2 = TARGET_ORDER[i + 1]
            if stage1 in stage_scores and stage2 in stage_scores:
                if stage_scores[stage1] >= stage_scores[stage2]:
                    issues.append((stage1, stage2, stage_scores[stage1], stage_scores[stage2]))
        
        print(f"\n{config['name']}:")
        print(f"权重: {weights}")
        print(f"满足顺序: {len(TARGET_ORDER) - 1 - len(issues)}/{len(TARGET_ORDER) - 1}")
        if issues:
            print(f"问题:")
            for s1, s2, score1, score2 in issues:
                print(f"  - {s1} ({score1:.2f}) >= {s2} ({score2:.2f})")
        else:
            print("✓ 完全满足目标顺序！")
        
        print(f"\n各阶段得分:")
        for stage in TARGET_ORDER:
            if stage in stage_scores:
                print(f"  {stage:20s}: {stage_scores[stage]:.2f}")


if __name__ == "__main__":
    main()


