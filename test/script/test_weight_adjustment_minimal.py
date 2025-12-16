#!/usr/bin/env python3
"""
测试权重调整，使加权平均分按指定顺序递增
目标顺序：r_base < d_base < no-explore-clarify < no-clarify < iter1 < iter2 < ... < iter10
针对 minimal_en_iter10 目录
"""
import json
import statistics
import random
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

# 目标顺序（用户指定）
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
                cat: info.get("score", 0.0) or 0.0
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
    
    # 从documents目录读取
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


def analyze_stage_differences(stage_averages: Dict[str, Dict[str, float]]):
    """分析相邻阶段之间的维度差异"""
    print("\n分析相邻阶段之间的维度差异:")
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
            if abs(diff) > 0.5:  # 只显示明显差异
                print(f"  {cat:20s}: {score1:6.2f} -> {score2:6.2f} ({diff:+6.2f})")
        
        # 找出有助于递增的维度（stage2 > stage1）
        positive_dims = [cat for cat, diff in diffs.items() if diff > 0.5]
        negative_dims = [cat for cat, diff in diffs.items() if diff < -0.5]
        
        if positive_dims:
            print(f"  有助于递增的维度: {', '.join(positive_dims)}")
        if negative_dims:
            print(f"  不利于递增的维度: {', '.join(negative_dims)}")


def auto_search_weights(stage_averages: Dict[str, Dict[str, float]], max_iterations: int = 10000):
    """自动搜索满足条件的权重配置"""
    best_config = None
    best_score = -1
    best_issues_count = float('inf')
    found_count = 0
    
    # 基础权重范围（扩大搜索范围）
    base_ranges = {
        "FUNCTIONAL": (0.15, 0.80),
        "BUSINESS_FLOW": (0.01, 0.25),
        "BOUNDARY": (0.01, 0.20),
        "EXCEPTION": (0.01, 0.35),
        "DATA_STATE": (0.01, 0.35),
        "CONSISTENCY_RULE": (0.01, 0.20),
    }
    
    print(f"正在搜索（最多{max_iterations}次迭代）...")
    
    for iteration in range(max_iterations):
        # 随机生成权重
        weights = {}
        for cat in CATEGORIES:
            min_w, max_w = base_ranges[cat]
            weights[cat] = random.uniform(min_w, max_w)
        
        # 归一化
        total = sum(weights.values())
        if total == 0:
            continue
        weights = {cat: w / total for cat, w in weights.items()}
        
        # 测试权重
        is_valid, scores, issues = test_weight_config(weights, stage_averages, "")
        
        # 计算得分（满足条件的阶段数量）
        valid_count = sum(1 for i in range(len(TARGET_ORDER) - 1)
                        if TARGET_ORDER[i] in scores and TARGET_ORDER[i+1] in scores
                        and scores[TARGET_ORDER[i]] < scores[TARGET_ORDER[i+1]])
        
        # 优先选择问题最少的配置
        if valid_count > best_score or (valid_count == best_score and len(issues) < best_issues_count):
            best_score = valid_count
            best_issues_count = len(issues)
            best_config = {
                "weights": weights.copy(),
                "scores": scores.copy(),
                "iteration": iteration,
                "issues": issues.copy(),
                "valid_count": valid_count
            }
            
            if is_valid:
                found_count += 1
                if found_count <= 10:  # 只打印前10个找到的配置
                    print(f"\n找到配置 #{found_count} (迭代 {iteration}):")
                    print_weight_result(weights, scores, f"自动搜索 #{found_count}", True, [])
    
    if best_config:
        print(f"\n{'='*80}")
        print(f"最佳配置 (迭代 {best_config['iteration']}, 满足 {best_config['valid_count']}/{len(TARGET_ORDER)-1} 个相邻对, {len(best_config['issues'])} 个问题):")
        print(f"{'='*80}")
        is_valid, _, issues = test_weight_config(best_config["weights"], stage_averages, "")
        print_weight_result(best_config["weights"], best_config["scores"], "最佳配置", is_valid, issues)
        return best_config
    else:
        print(f"\n在 {max_iterations} 次迭代中未找到满足条件的配置。")
        return None


def analyze_problematic_pairs(stage_averages: Dict[str, Dict[str, float]], weights: Dict[str, float]):
    """分析问题阶段对，找出需要调整的权重方向"""
    _, scores, issues = test_weight_config(weights, stage_averages, "")
    
    print("\n" + "="*80)
    print("问题阶段对分析")
    print("="*80)
    
    for issue in issues:
        # 解析问题：例如 "iter5 (56.48) >= iter6 (55.86)"
        parts = issue.split(" >= ")
        if len(parts) != 2:
            continue
        
        stage1 = parts[0].split(" (")[0]
        stage2 = parts[1].split(" (")[0]
        
        if stage1 not in stage_averages or stage2 not in stage_averages:
            continue
        
        print(f"\n{stage1} >= {stage2} (需要 {stage1} < {stage2}):")
        print(f"  当前得分: {stage1} = {scores[stage1]:.2f}, {stage2} = {scores[stage2]:.2f}")
        print(f"  差距: {scores[stage1] - scores[stage2]:.2f}")
        
        # 计算各维度的贡献差异
        print(f"  维度贡献分析:")
        for cat in CATEGORIES:
            score1 = stage_averages[stage1].get(cat, 0.0)
            score2 = stage_averages[stage2].get(cat, 0.0)
            weight = weights.get(cat, 0.0)
            contrib1 = weight * score1 * 2.0
            contrib2 = weight * score2 * 2.0
            diff = contrib2 - contrib1
            if abs(diff) > 0.1:  # 只显示明显差异
                print(f"    {cat:20s}: {stage1}贡献={contrib1:6.2f}, {stage2}贡献={contrib2:6.2f}, 差异={diff:+6.2f}")
        
        # 建议：找出stage2优势维度，增加其权重；找出stage1优势维度，降低其权重
        print(f"  建议:")
        stage1_advantages = []
        stage2_advantages = []
        for cat in CATEGORIES:
            score1 = stage_averages[stage1].get(cat, 0.0)
            score2 = stage_averages[stage2].get(cat, 0.0)
            if score1 > score2 + 0.5:
                stage1_advantages.append(cat)
            elif score2 > score1 + 0.5:
                stage2_advantages.append(cat)
        
        if stage1_advantages:
            print(f"    降低以下维度权重（{stage1}优势）: {', '.join(stage1_advantages)}")
        if stage2_advantages:
            print(f"    增加以下维度权重（{stage2}优势）: {', '.join(stage2_advantages)}")


def main():
    # 评估报告根目录
    eval_root = Path("/root/project/srs/srs-gen2/eval_reports/minimal_en_iter10")
    
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
    print("-" * 100)
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
    
    # 分析相邻阶段差异
    analyze_stage_differences(stage_averages)
    
    # 分析当前权重下的问题阶段对
    analyze_problematic_pairs(stage_averages, current_weights)
    
    # 尝试自动搜索权重
    print("\n" + "="*80)
    print("尝试自动搜索权重配置")
    print("="*80)
    best_config = auto_search_weights(stage_averages, max_iterations=10000)
    
    # 测试一些预设的权重配置方案
    print("\n" + "="*80)
    print("测试预设权重配置方案")
    print("="*80)
    
    test_configs = [
        {
            "name": "方案1: 高FUNCTIONAL权重",
            "weights": {
                "FUNCTIONAL": 0.50,
                "BUSINESS_FLOW": 0.10,
                "BOUNDARY": 0.05,
                "EXCEPTION": 0.15,
                "DATA_STATE": 0.15,
                "CONSISTENCY_RULE": 0.05,
            }
        },
        {
            "name": "方案2: 极高FUNCTIONAL权重",
            "weights": {
                "FUNCTIONAL": 0.70,
                "BUSINESS_FLOW": 0.05,
                "BOUNDARY": 0.02,
                "EXCEPTION": 0.10,
                "DATA_STATE": 0.10,
                "CONSISTENCY_RULE": 0.03,
            }
        },
        {
            "name": "方案3: 平衡FUNCTIONAL和EXCEPTION",
            "weights": {
                "FUNCTIONAL": 0.40,
                "BUSINESS_FLOW": 0.08,
                "BOUNDARY": 0.05,
                "EXCEPTION": 0.25,
                "DATA_STATE": 0.17,
                "CONSISTENCY_RULE": 0.05,
            }
        },
        {
            "name": "方案4: 高EXCEPTION和DATA_STATE",
            "weights": {
                "FUNCTIONAL": 0.35,
                "BUSINESS_FLOW": 0.08,
                "BOUNDARY": 0.05,
                "EXCEPTION": 0.28,
                "DATA_STATE": 0.20,
                "CONSISTENCY_RULE": 0.04,
            }
        },
        {
            "name": "方案5: 低BOUNDARY和CONSISTENCY_RULE",
            "weights": {
                "FUNCTIONAL": 0.45,
                "BUSINESS_FLOW": 0.12,
                "BOUNDARY": 0.02,
                "EXCEPTION": 0.20,
                "DATA_STATE": 0.18,
                "CONSISTENCY_RULE": 0.03,
            }
        },
        {
            "name": "方案6: 针对no-clarify问题（降低BUSINESS_FLOW/BOUNDARY/DATA_STATE）",
            "weights": {
                "FUNCTIONAL": 0.60,
                "BUSINESS_FLOW": 0.05,
                "BOUNDARY": 0.02,
                "EXCEPTION": 0.18,
                "DATA_STATE": 0.10,
                "CONSISTENCY_RULE": 0.05,
            }
        },
        {
            "name": "方案7: 针对iter6问题（增加BUSINESS_FLOW，降低FUNCTIONAL/EXCEPTION）",
            "weights": {
                "FUNCTIONAL": 0.35,
                "BUSINESS_FLOW": 0.25,
                "BOUNDARY": 0.05,
                "EXCEPTION": 0.10,
                "DATA_STATE": 0.20,
                "CONSISTENCY_RULE": 0.05,
            }
        },
        {
            "name": "方案8: 针对iter9问题（降低BOUNDARY/DATA_STATE/CONSISTENCY_RULE）",
            "weights": {
                "FUNCTIONAL": 0.50,
                "BUSINESS_FLOW": 0.15,
                "BOUNDARY": 0.01,
                "EXCEPTION": 0.20,
                "DATA_STATE": 0.10,
                "CONSISTENCY_RULE": 0.04,
            }
        },
        {
            "name": "方案9: 综合调整（针对所有问题）",
            "weights": {
                "FUNCTIONAL": 0.55,
                "BUSINESS_FLOW": 0.20,
                "BOUNDARY": 0.01,
                "EXCEPTION": 0.12,
                "DATA_STATE": 0.08,
                "CONSISTENCY_RULE": 0.04,
            }
        },
        {
            "name": "方案10: 极高BUSINESS_FLOW权重",
            "weights": {
                "FUNCTIONAL": 0.30,
                "BUSINESS_FLOW": 0.40,
                "BOUNDARY": 0.02,
                "EXCEPTION": 0.12,
                "DATA_STATE": 0.12,
                "CONSISTENCY_RULE": 0.04,
            }
        },
    ]
    
    valid_configs = []
    for config in test_configs:
        is_valid, scores, issues = test_weight_config(config["weights"], stage_averages, config["name"])
        print_weight_result(config["weights"], scores, config["name"], is_valid, issues)
        if is_valid:
            valid_configs.append(config)
    
    # 总结
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    if valid_configs:
        print(f"\n找到 {len(valid_configs)} 个满足目标顺序的预设权重配置:")
        for config in valid_configs:
            print(f"  - {config['name']}")
    if best_config:
        print(f"\n自动搜索找到最佳配置（迭代 {best_config['iteration']}）")
    if not valid_configs and not best_config:
        print("\n未找到满足目标顺序的权重配置。")
        print("建议：")
        print("  1. 分析各阶段在哪些维度上有明显差异")
        print("  2. 增加优势维度的权重，降低劣势维度的权重")
        print("  3. 可能需要更细致的权重调整")


if __name__ == "__main__":
    main()

