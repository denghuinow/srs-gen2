#!/usr/bin/env python3
"""
权重优化测算脚本
目标：通过调整权重，使加权平均分按以下顺序严格递增：
r_base < d_base < no-explore-clarify < no-clarify < iter1 < iter2 < ... < iter10

本脚本只进行测算，不修改现有数据和代码
"""
import json
import random
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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

# 目标顺序（按用户要求）
TARGET_ORDER = [
    "r_base",           # 0
    "d_base",           # 1
    "no-explore-clarify",  # 2
    "no-clarify",       # 3
    "iter1",            # 4
    "iter2",            # 5
    "iter3",            # 6
    "iter4",            # 7
    "iter5",            # 8
    "iter6",            # 9
    "iter7",            # 10
    "iter8",            # 11
    "iter9",            # 12
    "iter10",           # 13
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


def check_order(scores: Dict[str, float], target_order: List[str]) -> Tuple[bool, List[str], int]:
    """检查得分是否按目标顺序递增，返回(是否满足, 问题列表, 满足的相邻对数量)"""
    issues = []
    valid_pairs = 0
    
    for i in range(len(target_order) - 1):
        stage1 = target_order[i]
        stage2 = target_order[i + 1]
        if stage1 not in scores or stage2 not in scores:
            continue
        if scores[stage1] >= scores[stage2]:
            issues.append(f"{stage1} ({scores[stage1]:.2f}) >= {stage2} ({scores[stage2]:.2f})")
        else:
            valid_pairs += 1
    
    return len(issues) == 0, issues, valid_pairs


def test_weight_config(
    weights: Dict[str, float], 
    stage_averages: Dict[str, Dict[str, float]]
) -> Tuple[bool, Dict[str, float], List[str], int]:
    """测试权重配置"""
    # 计算各阶段的加权得分
    stage_scores = {}
    for stage_name, category_scores in stage_averages.items():
        score = calculate_weighted_score(category_scores, weights)
        stage_scores[stage_name] = score
    
    # 检查是否满足顺序
    is_valid, issues, valid_pairs = check_order(stage_scores, TARGET_ORDER)
    
    return is_valid, stage_scores, issues, valid_pairs


def print_weight_result(
    weights: Dict[str, float], 
    stage_scores: Dict[str, float], 
    config_name: str, 
    is_valid: bool, 
    issues: List[str],
    valid_pairs: int,
    total_pairs: int
):
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
    for idx, stage in enumerate(TARGET_ORDER):
        if stage in stage_scores:
            score = stage_scores[stage]
            arrow = "→" if idx < len(TARGET_ORDER) - 1 else ""
            next_stage = TARGET_ORDER[idx + 1] if idx < len(TARGET_ORDER) - 1 else None
            if next_stage and next_stage in stage_scores:
                next_score = stage_scores[next_stage]
                status = "✓" if score < next_score else "✗"
                print(f"  [{idx:2d}] {stage:20s}: {score:6.2f} {status} {arrow} {next_stage} ({next_score:.2f})")
            else:
                print(f"  [{idx:2d}] {stage:20s}: {score:6.2f}")
        else:
            print(f"  [{idx:2d}] {stage:20s}: (无数据)")
    
    print(f"\n满足顺序: {valid_pairs}/{total_pairs} 个相邻对")
    if is_valid:
        print(f"✓ 完全满足目标顺序！")
    else:
        print(f"✗ 不满足目标顺序，问题如下:")
        for issue in issues:
            print(f"  - {issue}")


def generate_random_weights(
    ranges: Dict[str, Tuple[float, float]],
    normalize: bool = True
) -> Dict[str, float]:
    """根据指定范围生成随机权重"""
    weights = {}
    for cat in CATEGORIES:
        min_w, max_w = ranges.get(cat, (0.0, 1.0))
        weights[cat] = random.uniform(min_w, max_w)
    
    if normalize:
        total = sum(weights.values())
        weights = {cat: w / total for cat, w in weights.items()}
    
    return weights


def optimize_weights(
    stage_averages: Dict[str, Dict[str, float]],
    max_iterations: int = 10000,
    verbose: bool = True
) -> Optional[Dict[str, float]]:
    """使用随机搜索优化权重配置"""
    
    # 基于数据分析的权重范围
    # 根据之前测试，需要降低BOUNDARY权重，增加FUNCTIONAL权重
    base_ranges = {
        "FUNCTIONAL": (0.30, 0.70),
        "BUSINESS_FLOW": (0.03, 0.20),
        "BOUNDARY": (0.01, 0.15),  # 大幅降低
        "EXCEPTION": (0.05, 0.30),
        "DATA_STATE": (0.05, 0.30),
        "CONSISTENCY_RULE": (0.01, 0.15),  # 降低
    }
    
    best_config = None
    best_valid_pairs = -1
    found_configs = []
    
    if verbose:
        print(f"开始随机搜索（最多 {max_iterations} 次迭代）...")
    
    for iteration in range(max_iterations):
        # 随机生成权重
        weights = generate_random_weights(base_ranges)
        
        # 测试权重
        is_valid, scores, issues, valid_pairs = test_weight_config(weights, stage_averages)
        
        if valid_pairs > best_valid_pairs:
            best_valid_pairs = valid_pairs
            best_config = {
                "weights": weights,
                "scores": scores,
                "iteration": iteration,
                "is_valid": is_valid,
                "issues": issues,
                "valid_pairs": valid_pairs,
            }
        
        if is_valid:
            found_configs.append({
                "weights": weights,
                "scores": scores,
                "iteration": iteration,
            })
            if verbose and len(found_configs) <= 3:  # 只打印前3个找到的配置
                print(f"\n找到满足条件的配置 #{len(found_configs)} (迭代 {iteration}):")
                total_pairs = len(TARGET_ORDER) - 1
                print_weight_result(
                    weights, scores, f"自动搜索 #{len(found_configs)}", 
                    True, [], total_pairs, total_pairs
                )
    
    if verbose:
        print(f"\n搜索完成：")
        print(f"  - 总迭代次数: {max_iterations}")
        print(f"  - 找到满足条件的配置数: {len(found_configs)}")
        print(f"  - 最佳配置满足: {best_valid_pairs}/{len(TARGET_ORDER)-1} 个相邻对")
    
    if found_configs:
        return found_configs[0]["weights"]
    elif best_config:
        return best_config["weights"]
    else:
        return None


def analyze_dimension_contributions(
    stage_averages: Dict[str, Dict[str, float]]
):
    """分析各维度对顺序的贡献"""
    print("\n" + "="*80)
    print("维度贡献分析")
    print("="*80)
    
    # 分析每个相邻对的维度差异
    for i in range(len(TARGET_ORDER) - 1):
        stage1 = TARGET_ORDER[i]
        stage2 = TARGET_ORDER[i + 1]
        
        if stage1 not in stage_averages or stage2 not in stage_averages:
            continue
        
        print(f"\n{stage1} → {stage2}:")
        diffs = {}
        for cat in CATEGORIES:
            score1 = stage_averages[stage1].get(cat, 0.0)
            score2 = stage_averages[stage2].get(cat, 0.0)
            diff = score2 - score1
            diffs[cat] = diff
        
        # 按差异大小排序
        sorted_diffs = sorted(diffs.items(), key=lambda x: x[1], reverse=True)
        
        print("  维度差异（从大到小）:")
        for cat, diff in sorted_diffs:
            score1 = stage_averages[stage1].get(cat, 0.0)
            score2 = stage_averages[stage2].get(cat, 0.0)
            status = "✓" if diff > 0 else "✗"
            print(f"    {status} {cat:20s}: {score1:6.2f} → {score2:6.2f} ({diff:+7.2f})")
        
        # 建议权重调整方向
        positive_dims = [cat for cat, diff in diffs.items() if diff > 0]
        negative_dims = [cat for cat, diff in diffs.items() if diff < 0]
        
        if negative_dims:
            print(f"  建议: 降低以下维度权重以促进递增: {', '.join(negative_dims)}")
        if positive_dims:
            print(f"  建议: 增加以下维度权重以促进递增: {', '.join(positive_dims)}")


def test_preset_configs(
    stage_averages: Dict[str, Dict[str, float]]
):
    """测试预设的权重配置方案"""
    preset_configs = [
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
            "name": "方案1: 高FUNCTIONAL + 低BOUNDARY",
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
            "name": "方案2: 平衡FUNCTIONAL + EXCEPTION + DATA_STATE",
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
            "name": "方案3: 极高FUNCTIONAL",
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
            "name": "方案4: FUNCTIONAL + EXCEPTION主导",
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
            "name": "方案5: FUNCTIONAL + DATA_STATE主导",
            "weights": {
                "FUNCTIONAL": 0.55,
                "BUSINESS_FLOW": 0.05,
                "BOUNDARY": 0.02,
                "EXCEPTION": 0.10,
                "DATA_STATE": 0.25,
                "CONSISTENCY_RULE": 0.03,
            }
        },
    ]
    
    print("\n" + "="*80)
    print("测试预设权重配置")
    print("="*80)
    
    total_pairs = len(TARGET_ORDER) - 1
    valid_configs = []
    
    for config in preset_configs:
        is_valid, scores, issues, valid_pairs = test_weight_config(
            config["weights"], stage_averages
        )
        print_weight_result(
            config["weights"], scores, config["name"], 
            is_valid, issues, valid_pairs, total_pairs
        )
        if is_valid:
            valid_configs.append(config)
    
    return valid_configs


def main():
    # 评估报告根目录
    eval_root = Path("/root/project/srs/srs-gen2/eval_reports/en_iter10")
    
    print("="*80)
    print("权重优化测算脚本")
    print("="*80)
    print("\n目标顺序:")
    for idx, stage in enumerate(TARGET_ORDER):
        print(f"  [{idx:2d}] {stage}")
    
    print("\n正在加载各阶段的评估数据...")
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
    
    # 显示各阶段的维度平均得分
    print("\n" + "="*80)
    print("各阶段维度平均得分")
    print("="*80)
    print(f"{'阶段':<20}", end="")
    for cat in CATEGORIES:
        print(f"{cat[:12]:>12}", end="")
    print()
    print("-" * 92)
    for stage in TARGET_ORDER:
        if stage not in stage_averages:
            continue
        print(f"{stage:<20}", end="")
        for cat in CATEGORIES:
            score = stage_averages[stage].get(cat, 0.0)
            print(f"{score:>12.2f}", end="")
        print()
    
    # 分析维度贡献
    analyze_dimension_contributions(stage_averages)
    
    # 测试预设配置
    valid_configs = test_preset_configs(stage_averages)
    
    # 自动优化搜索
    print("\n" + "="*80)
    print("自动权重优化搜索")
    print("="*80)
    optimal_weights = optimize_weights(stage_averages, max_iterations=10000)
    
    if optimal_weights:
        print("\n" + "="*80)
        print("找到的最优权重配置")
        print("="*80)
        is_valid, scores, issues, valid_pairs = test_weight_config(
            optimal_weights, stage_averages
        )
        total_pairs = len(TARGET_ORDER) - 1
        print_weight_result(
            optimal_weights, scores, "最优配置", 
            is_valid, issues, valid_pairs, total_pairs
        )
    else:
        print("\n未找到完全满足条件的权重配置")
    
    # 总结
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    if valid_configs:
        print(f"\n找到 {len(valid_configs)} 个满足目标顺序的预设配置:")
        for config in valid_configs:
            print(f"  - {config['name']}")
    else:
        print("\n预设配置中未找到完全满足目标顺序的配置")
    
    if optimal_weights:
        print("\n通过自动搜索找到了最优权重配置（见上方）")
    else:
        print("\n自动搜索未找到完全满足条件的配置")
        print("建议：")
        print("  1. 根据维度贡献分析调整权重范围")
        print("  2. 增加搜索迭代次数")
        print("  3. 考虑分段权重策略（不同阶段使用不同权重）")


if __name__ == "__main__":
    main()



