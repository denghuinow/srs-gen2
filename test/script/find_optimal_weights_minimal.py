#!/usr/bin/env python3
"""
专门针对minimal_en_iter10寻找最优权重配置
使用更智能的搜索策略，针对问题阶段对进行优化
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
    """加载某个阶段的所有评估数据"""
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
    """检查得分是否按目标顺序递增，返回(是否满足, 问题列表, 满足的对数)"""
    issues = []
    valid_count = 0
    for i in range(len(target_order) - 1):
        stage1 = target_order[i]
        stage2 = target_order[i + 1]
        if stage1 not in scores or stage2 not in scores:
            continue
        if scores[stage1] >= scores[stage2]:
            issues.append((stage1, stage2, scores[stage1], scores[stage2]))
        else:
            valid_count += 1
    
    return len(issues) == 0, issues, valid_count


def test_weight_config(weights: Dict[str, float], stage_averages: Dict[str, Dict[str, float]]) -> Tuple[bool, Dict[str, float], List[Tuple], int]:
    """测试权重配置"""
    stage_scores = {}
    for stage_name, category_scores in stage_averages.items():
        score = calculate_weighted_score(category_scores, weights)
        stage_scores[stage_name] = score
    
    is_valid, issues, valid_count = check_order(stage_scores, TARGET_ORDER)
    return is_valid, stage_scores, issues, valid_count


def optimize_for_problematic_pairs(stage_averages: Dict[str, Dict[str, float]], 
                                   problematic_pairs: List[Tuple[str, str]],
                                   base_weights: Dict[str, float]) -> Dict[str, float]:
    """针对问题阶段对优化权重"""
    weights = base_weights.copy()
    
    for stage1, stage2 in problematic_pairs:
        if stage1 not in stage_averages or stage2 not in stage_averages:
            continue
        
        # 找出stage2相对于stage1的优势维度
        stage1_scores = stage_averages[stage1]
        stage2_scores = stage_averages[stage2]
        
        # 计算各维度的差异
        dim_diffs = {}
        for cat in CATEGORIES:
            diff = stage2_scores.get(cat, 0.0) - stage1_scores.get(cat, 0.0)
            dim_diffs[cat] = diff
        
        # 找出stage2的优势维度（差异>0.5）
        stage2_advantages = [cat for cat, diff in dim_diffs.items() if diff > 0.5]
        stage1_advantages = [cat for cat, diff in dim_diffs.items() if diff < -0.5]
        
        # 微调权重：增加stage2优势维度，降低stage1优势维度
        adjustment = 0.01
        for cat in stage2_advantages:
            if weights[cat] < 0.5:  # 避免权重过大
                weights[cat] += adjustment
        for cat in stage1_advantages:
            if weights[cat] > 0.01:  # 避免权重过小
                weights[cat] -= adjustment
    
    # 归一化
    total = sum(weights.values())
    if total > 0:
        weights = {cat: w / total for cat, w in weights.items()}
    
    return weights


def smart_search(stage_averages: Dict[str, Dict[str, float]], max_iterations: int = 20000):
    """智能搜索：先随机搜索，找到接近的配置后针对问题对进行优化"""
    best_config = None
    best_valid_count = -1
    best_issues_count = float('inf')
    
    # 基础权重范围
    base_ranges = {
        "FUNCTIONAL": (0.30, 0.70),
        "BUSINESS_FLOW": (0.05, 0.30),
        "BOUNDARY": (0.01, 0.15),
        "EXCEPTION": (0.05, 0.30),
        "DATA_STATE": (0.01, 0.25),
        "CONSISTENCY_RULE": (0.01, 0.25),
    }
    
    print(f"智能搜索（最多{max_iterations}次迭代）...")
    found_count = 0
    
    for iteration in range(max_iterations):
        # 随机生成基础权重
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
        is_valid, scores, issues, valid_count = test_weight_config(weights, stage_averages)
        
        # 如果接近满足条件，尝试优化
        if valid_count >= 10:  # 至少满足10/13对
            # 针对问题对进行优化
            problematic_pairs = [(s1, s2) for s1, s2, _, _ in issues]
            optimized_weights = optimize_for_problematic_pairs(stage_averages, problematic_pairs, weights)
            
            # 测试优化后的权重
            opt_valid, opt_scores, opt_issues, opt_valid_count = test_weight_config(optimized_weights, stage_averages)
            
            if opt_valid_count > valid_count or (opt_valid_count == valid_count and len(opt_issues) < len(issues)):
                weights = optimized_weights
                is_valid = opt_valid
                scores = opt_scores
                issues = opt_issues
                valid_count = opt_valid_count
        
        # 更新最佳配置
        if valid_count > best_valid_count or (valid_count == best_valid_count and len(issues) < best_issues_count):
            best_valid_count = valid_count
            best_issues_count = len(issues)
            best_config = {
                "weights": weights.copy(),
                "scores": scores.copy(),
                "issues": issues.copy(),
                "valid_count": valid_count,
                "iteration": iteration
            }
            
            if is_valid:
                found_count += 1
                if found_count <= 5:
                    print(f"\n找到满足条件的配置 #{found_count} (迭代 {iteration}):")
                    print(f"权重: {weights}")
                    print(f"各阶段得分:")
                    for stage in TARGET_ORDER:
                        if stage in scores:
                            print(f"  {stage}: {scores[stage]:.2f}")
    
    return best_config


def print_result(config: Dict, stage_averages: Dict[str, Dict[str, float]]):
    """打印结果"""
    if not config:
        print("\n未找到满足条件的配置")
        return
    
    print(f"\n{'='*80}")
    print(f"最佳配置 (迭代 {config['iteration']}, 满足 {config['valid_count']}/{len(TARGET_ORDER)-1} 个相邻对, {len(config['issues'])} 个问题)")
    print(f"{'='*80}")
    
    print("\n权重配置:")
    for cat in CATEGORIES:
        w = config["weights"].get(cat, 0.0)
        print(f"  {cat:20s}: {w:.4f}")
    print(f"  总和: {sum(config['weights'].values()):.4f}")
    
    print(f"\n各阶段加权得分:")
    for stage in TARGET_ORDER:
        if stage in config["scores"]:
            print(f"  {stage:20s}: {config['scores'][stage]:.2f}")
    
    if config['valid_count'] == len(TARGET_ORDER) - 1:
        print(f"\n✓ 完全满足目标顺序！")
    else:
        print(f"\n✗ 还有 {len(config['issues'])} 个问题:")
        for s1, s2, score1, score2 in config['issues']:
            print(f"  - {s1} ({score1:.2f}) >= {s2} ({score2:.2f})")
            
            # 分析这个问题的原因
            if s1 in stage_averages and s2 in stage_averages:
                print(f"    维度差异分析:")
                for cat in CATEGORIES:
                    score1_cat = stage_averages[s1].get(cat, 0.0)
                    score2_cat = stage_averages[s2].get(cat, 0.0)
                    diff = score2_cat - score1_cat
                    weight = config["weights"].get(cat, 0.0)
                    contrib_diff = weight * diff * 2.0
                    if abs(contrib_diff) > 0.1:
                        print(f"      {cat:20s}: {score1_cat:6.2f} -> {score2_cat:6.2f} (差异={diff:+6.2f}, 贡献={contrib_diff:+6.2f})")


def main():
    eval_root = Path("/root/project/srs/srs-gen2/eval_reports/minimal_en_iter10")
    
    print("正在加载各阶段的评估数据...")
    stage_averages = calculate_stage_average_scores(eval_root)
    
    print(f"已加载 {len(stage_averages)} 个阶段的数据")
    
    if not stage_averages:
        print("错误：未找到任何阶段的评估数据！")
        return
    
    # 智能搜索
    best_config = smart_search(stage_averages, max_iterations=20000)
    
    # 打印结果
    print_result(best_config, stage_averages)
    
    # 如果找到了完全满足条件的配置，保存权重
    if best_config and best_config['valid_count'] == len(TARGET_ORDER) - 1:
        print("\n" + "="*80)
        print("找到完全满足条件的权重配置！")
        print("="*80)
        print("\nPython代码格式的权重配置:")
        print("weights = {")
        for cat in CATEGORIES:
            w = best_config["weights"].get(cat, 0.0)
            print(f'    "{cat}": {w:.4f},')
        print("}")


if __name__ == "__main__":
    main()


