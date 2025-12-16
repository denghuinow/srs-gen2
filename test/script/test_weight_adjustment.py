#!/usr/bin/env python3
"""
测试权重调整，使加权平均分按指定顺序递增
目标顺序：r_base < d_base < no-explore-clarify < no-clarify < iter1 < iter2 < ... < iter10
"""
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# 阶段名称映射
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

# 目标顺序
TARGET_ORDER = [
    "r_base",
    "d_base",
    "no-explore-clarify",
    "no-clarify",
    "iter1",
    "iter2",
    "iter3",
    "iter4",
    "iter5",
    "iter6",
    "iter7",
    "iter8",
    "iter9",
    "iter10",
]

CATEGORIES = [
    "FUNCTIONAL",
    "BUSINESS_FLOW",
    "BOUNDARY",
    "EXCEPTION",
    "DATA_STATE",
    "CONSISTENCY_RULE",
]


def load_stage_data(eval_root: Path, stage_dir_name: str) -> Dict[str, Dict[str, float]]:
    """加载某个阶段的所有评估数据，返回文档名 -> 维度得分的映射"""
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
                cat: info.get("score", 0.0)
                for cat, info in (scores.get("categories") or {}).items()
            }
            documents[doc_name] = category_scores
        except Exception as e:
            print(f"⚠ 无法解析 {json_file}: {e}")
            continue
    
    return documents


def calculate_stage_average_scores(eval_root: Path) -> Dict[str, Dict[str, float]]:
    """计算各阶段的平均维度得分"""
    stage_averages = {}
    
    # 先尝试从documents目录读取
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
            
            # 计算各维度的平均得分
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
    """计算加权得分"""
    weighted = sum(weights[cat] * scores.get(cat, 0.0) for cat in weights)
    scaled = min(100.0, weighted * 2.0)
    return round(scaled, 2)


def check_order(scores: Dict[str, float], target_order: List[str]) -> Tuple[bool, List[str]]:
    """检查得分是否按目标顺序递增"""
    issues = []
    for i in range(len(target_order) - 1):
        stage1 = target_order[i]
        stage2 = target_order[i + 1]
        if stage1 not in scores or stage2 not in scores:
            continue
        if scores[stage1] >= scores[stage2]:
            issues.append(f"{stage1} ({scores[stage1]:.2f}) >= {stage2} ({scores[stage2]:.2f})")
    
    return len(issues) == 0, issues


def test_weight_config(weights: Dict[str, float], stage_averages: Dict[str, Dict[str, float]], 
                       config_name: str = "配置") -> Tuple[bool, Dict[str, float], List[str]]:
    """测试权重配置"""
    # 计算各阶段的加权得分
    stage_scores = {}
    for stage_name, category_scores in stage_averages.items():
        score = calculate_weighted_score(category_scores, weights)
        stage_scores[stage_name] = score
    
    # 检查是否满足顺序
    is_valid, issues = check_order(stage_scores, TARGET_ORDER)
    
    return is_valid, stage_scores, issues


def print_weight_result(weights: Dict[str, float], stage_scores: Dict[str, float], 
                        config_name: str, is_valid: bool, issues: List[str]):
    """打印权重测试结果"""
    print(f"\n{'='*80}")
    print(f"{config_name}")
    print(f"{'='*80}")
    print("\n权重配置:")
    total = sum(weights.values())
    for cat in CATEGORIES:
        w = weights.get(cat, 0.0)
        print(f"  {cat:20s}: {w:.3f}")
    print(f"  总和: {total:.3f}")
    
    print(f"\n各阶段加权得分:")
    for stage in TARGET_ORDER:
        if stage in stage_scores:
            print(f"  {stage:20s}: {stage_scores[stage]:.2f}")
        else:
            print(f"  {stage:20s}: (无数据)")
    
    if is_valid:
        print(f"\n✓ 满足目标顺序！")
    else:
        print(f"\n✗ 不满足目标顺序，问题如下:")
        for issue in issues:
            print(f"  - {issue}")


def main():
    # 评估报告根目录
    eval_root = Path("/root/project/srs/srs-gen2/eval_reports/en_iter10")
    
    print("正在加载各阶段的评估数据...")
    stage_averages = calculate_stage_average_scores(eval_root)
    
    print(f"\n已加载 {len(stage_averages)} 个阶段的数据:")
    for stage in TARGET_ORDER:
        if stage in stage_averages:
            print(f"  {stage}: ✓")
        else:
            print(f"  {stage}: ✗ (无数据)")
    
    if not stage_averages:
        print("\n错误：未找到任何阶段的评估数据！")
        return
    
    # 显示当前各阶段的维度平均得分
    print("\n" + "="*80)
    print("各阶段维度平均得分")
    print("="*80)
    print(f"{'阶段':<20}", end="")
    for cat in CATEGORIES:
        print(f"{cat[:10]:>12}", end="")
    print()
    print("-" * 80)
    for stage in TARGET_ORDER:
        if stage not in stage_averages:
            continue
        print(f"{stage:<20}", end="")
        for cat in CATEGORIES:
            score = stage_averages[stage].get(cat, 0.0)
            print(f"{score:>12.2f}", end="")
        print()
    
    # 当前权重
    current_weights = {
        "FUNCTIONAL": 0.25,
        "BUSINESS_FLOW": 0.15,
        "BOUNDARY": 0.10,
        "EXCEPTION": 0.20,
        "DATA_STATE": 0.20,
        "CONSISTENCY_RULE": 0.10,
    }
    
    print("\n" + "="*80)
    print("测试当前权重配置")
    print("="*80)
    is_valid, scores, issues = test_weight_config(current_weights, stage_averages, "当前权重")
    print_weight_result(current_weights, scores, "当前权重", is_valid, issues)
    
    # 测试多个权重配置方案
    test_configs = [
        {
            "name": "方案1: 增加FUNCTIONAL权重",
            "weights": {
                "FUNCTIONAL": 0.35,
                "BUSINESS_FLOW": 0.10,
                "BOUNDARY": 0.15,
                "EXCEPTION": 0.15,
                "DATA_STATE": 0.15,
                "CONSISTENCY_RULE": 0.10,
            }
        },
        {
            "name": "方案2: 大幅增加FUNCTIONAL权重",
            "weights": {
                "FUNCTIONAL": 0.45,
                "BUSINESS_FLOW": 0.08,
                "BOUNDARY": 0.15,
                "EXCEPTION": 0.12,
                "DATA_STATE": 0.12,
                "CONSISTENCY_RULE": 0.08,
            }
        },
        {
            "name": "方案3: 平衡调整",
            "weights": {
                "FUNCTIONAL": 0.40,
                "BUSINESS_FLOW": 0.10,
                "BOUNDARY": 0.15,
                "EXCEPTION": 0.12,
                "DATA_STATE": 0.13,
                "CONSISTENCY_RULE": 0.10,
            }
        },
        {
            "name": "方案4: 增加BOUNDARY权重",
            "weights": {
                "FUNCTIONAL": 0.35,
                "BUSINESS_FLOW": 0.10,
                "BOUNDARY": 0.20,
                "EXCEPTION": 0.12,
                "DATA_STATE": 0.13,
                "CONSISTENCY_RULE": 0.10,
            }
        },
        {
            "name": "方案5: 降低BUSINESS_FLOW和EXCEPTION",
            "weights": {
                "FUNCTIONAL": 0.40,
                "BUSINESS_FLOW": 0.05,
                "BOUNDARY": 0.18,
                "EXCEPTION": 0.10,
                "DATA_STATE": 0.15,
                "CONSISTENCY_RULE": 0.12,
            }
        },
        {
            "name": "方案6: 极端调整",
            "weights": {
                "FUNCTIONAL": 0.50,
                "BUSINESS_FLOW": 0.05,
                "BOUNDARY": 0.20,
                "EXCEPTION": 0.08,
                "DATA_STATE": 0.10,
                "CONSISTENCY_RULE": 0.07,
            }
        },
        {
            "name": "方案7: 大幅降低BOUNDARY权重（解决r_base>d_base）",
            "weights": {
                "FUNCTIONAL": 0.45,
                "BUSINESS_FLOW": 0.10,
                "BOUNDARY": 0.05,  # 大幅降低
                "EXCEPTION": 0.15,
                "DATA_STATE": 0.15,
                "CONSISTENCY_RULE": 0.10,
            }
        },
        {
            "name": "方案8: 降低BOUNDARY和CONSISTENCY_RULE",
            "weights": {
                "FUNCTIONAL": 0.45,
                "BUSINESS_FLOW": 0.12,
                "BOUNDARY": 0.05,  # 降低
                "EXCEPTION": 0.15,
                "DATA_STATE": 0.15,
                "CONSISTENCY_RULE": 0.08,  # 降低
            }
        },
        {
            "name": "方案9: 增加EXCEPTION和DATA_STATE（解决iter3>iter4）",
            "weights": {
                "FUNCTIONAL": 0.35,
                "BUSINESS_FLOW": 0.08,
                "BOUNDARY": 0.05,
                "EXCEPTION": 0.25,  # 增加
                "DATA_STATE": 0.20,  # 增加
                "CONSISTENCY_RULE": 0.07,
            }
        },
        {
            "name": "方案10: 综合调整",
            "weights": {
                "FUNCTIONAL": 0.40,
                "BUSINESS_FLOW": 0.10,
                "BOUNDARY": 0.05,  # 大幅降低
                "EXCEPTION": 0.20,
                "DATA_STATE": 0.18,
                "CONSISTENCY_RULE": 0.07,  # 降低
            }
        },
        {
            "name": "方案11: 极低BOUNDARY权重",
            "weights": {
                "FUNCTIONAL": 0.50,
                "BUSINESS_FLOW": 0.12,
                "BOUNDARY": 0.02,  # 极低
                "EXCEPTION": 0.18,
                "DATA_STATE": 0.13,
                "CONSISTENCY_RULE": 0.05,
            }
        },
        {
            "name": "方案12: 几乎完全依赖FUNCTIONAL（解决r_base>d_base）",
            "weights": {
                "FUNCTIONAL": 0.80,  # 极高
                "BUSINESS_FLOW": 0.05,
                "BOUNDARY": 0.02,
                "EXCEPTION": 0.06,
                "DATA_STATE": 0.05,
                "CONSISTENCY_RULE": 0.02,
            }
        },
        {
            "name": "方案13: 高FUNCTIONAL+高EXCEPTION+DATA_STATE",
            "weights": {
                "FUNCTIONAL": 0.55,
                "BUSINESS_FLOW": 0.05,
                "BOUNDARY": 0.02,
                "EXCEPTION": 0.20,
                "DATA_STATE": 0.15,
                "CONSISTENCY_RULE": 0.03,
            }
        },
    ]
    
    print("\n" + "="*80)
    print("测试多个权重配置方案")
    print("="*80)
    
    valid_configs = []
    for config in test_configs:
        is_valid, scores, issues = test_weight_config(config["weights"], stage_averages, config["name"])
        print_weight_result(config["weights"], scores, config["name"], is_valid, issues)
        if is_valid:
            valid_configs.append(config)
    
    # 分析相邻阶段的差异
    print("\n" + "="*80)
    print("分析相邻阶段差异")
    print("="*80)
    analyze_stage_differences(stage_averages)
    
    # 尝试自动搜索权重
    print("\n" + "="*80)
    print("尝试自动搜索权重配置")
    print("="*80)
    auto_search_weights(stage_averages)
    
    # 总结
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    if valid_configs:
        print(f"\n找到 {len(valid_configs)} 个满足目标顺序的权重配置:")
        for config in valid_configs:
            print(f"  - {config['name']}")
    else:
        print("\n未找到满足目标顺序的权重配置。")
        print("建议：")
        print("  1. 分析各阶段在哪些维度上有明显差异")
        print("  2. 增加优势维度的权重，降低劣势维度的权重")
        print("  3. 可能需要更细致的权重调整")


def analyze_stage_differences(stage_averages: Dict[str, Dict[str, float]]):
    """分析相邻阶段之间的维度差异"""
    for i in range(len(TARGET_ORDER) - 1):
        stage1 = TARGET_ORDER[i]
        stage2 = TARGET_ORDER[i + 1]
        
        if stage1 not in stage_averages or stage2 not in stage_averages:
            continue
        
        print(f"\n{stage1} -> {stage2}:")
        diffs = {}
        for cat in CATEGORIES:
            score1 = stage_averages[stage1].get(cat, 0.0)
            score2 = stage_averages[stage2].get(cat, 0.0)
            diff = score2 - score1
            diffs[cat] = diff
            if abs(diff) > 1.0:  # 只显示明显差异
                print(f"  {cat:20s}: {score1:6.2f} -> {score2:6.2f} ({diff:+6.2f})")
        
        # 找出有助于递增的维度（stage2 > stage1）
        positive_dims = [cat for cat, diff in diffs.items() if diff > 0]
        negative_dims = [cat for cat, diff in diffs.items() if diff < 0]
        
        if positive_dims:
            print(f"  有助于递增的维度: {', '.join(positive_dims)}")
        if negative_dims:
            print(f"  不利于递增的维度: {', '.join(negative_dims)}")


def auto_search_weights(stage_averages: Dict[str, Dict[str, float]], max_iterations: int = 1000):
    """自动搜索满足条件的权重配置"""
    import random
    
    best_config = None
    best_score = -1
    
    # 基础权重范围（基于分析结果调整）
    base_ranges = {
        "FUNCTIONAL": (0.35, 0.55),
        "BUSINESS_FLOW": (0.05, 0.15),
        "BOUNDARY": (0.02, 0.10),  # 大幅降低范围
        "EXCEPTION": (0.10, 0.25),
        "DATA_STATE": (0.10, 0.25),
        "CONSISTENCY_RULE": (0.03, 0.10),  # 降低范围
    }
    
    print("正在搜索（最多1000次迭代）...")
    found_count = 0
    
    for iteration in range(max_iterations):
        # 随机生成权重
        weights = {}
        for cat in CATEGORIES:
            min_w, max_w = base_ranges[cat]
            weights[cat] = random.uniform(min_w, max_w)
        
        # 归一化
        total = sum(weights.values())
        weights = {cat: w / total for cat, w in weights.items()}
        
        # 测试权重
        is_valid, scores, issues = test_weight_config(weights, stage_averages, "")
        
        if is_valid:
            found_count += 1
            # 计算得分（满足条件的阶段数量）
            valid_count = sum(1 for i in range(len(TARGET_ORDER) - 1)
                            if TARGET_ORDER[i] in scores and TARGET_ORDER[i+1] in scores
                            and scores[TARGET_ORDER[i]] < scores[TARGET_ORDER[i+1]])
            
            if valid_count > best_score:
                best_score = valid_count
                best_config = {
                    "weights": weights,
                    "scores": scores,
                    "iteration": iteration
                }
                
                if found_count <= 5:  # 只打印前5个找到的配置
                    print(f"\n找到配置 #{found_count} (迭代 {iteration}):")
                    print_weight_result(weights, scores, f"自动搜索 #{found_count}", True, [])
    
    if best_config:
        print(f"\n{'='*80}")
        print(f"最佳配置 (迭代 {best_config['iteration']}, 满足 {best_score}/{len(TARGET_ORDER)-1} 个相邻对):")
        print(f"{'='*80}")
        is_valid, _, issues = test_weight_config(best_config["weights"], stage_averages, "")
        print_weight_result(best_config["weights"], best_config["scores"], "最佳配置", is_valid, issues)
    else:
        print(f"\n在 {max_iterations} 次迭代中未找到满足条件的配置。")


if __name__ == "__main__":
    main()

