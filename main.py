# main.py
import argparse
import json
import os
import logging
import math
from pathlib import Path
from typing import List, Dict, Optional
from models import SemanticUnit
from srs_pipeline import (
    split_to_semantic_units,
    requirement_explorer,
    requirement_improver,
    requirement_clarifier,
    srs_generator,
)
from utils.token_counter import count_tokens
from openai_utils import load_prompt_template
from config import get_config

# 获取配置
config = get_config()

# 配置日志
logging.basicConfig(level=getattr(logging, config.log_level), format=config.log_format)
logger = logging.getLogger(__name__)

# 禁用 httpx 日志
logging.getLogger("httpx").setLevel(logging.WARNING)


def generate_srs_from_pool(pool: List[SemanticUnit]) -> str:
    """
    从需求池生成一版 SRS

    Args:
        pool: 需求池（包含已评分的语义单元）

    Returns:
        SRS 文档文本
    """
    # 筛选出 grade > 0 的单元
    grade_units = [u for u in pool if u.grade is not None and u.grade > 0]
    logger.info(f"从需求池中筛选出 {len(grade_units)} 个已采纳需求（grade > 0）")

    if len(grade_units) == 0:
        logger.warning("需求池中没有已采纳的需求，返回空文档")
        return "# 软件需求规格说明书\n\n暂无需求。\n"

    return srs_generator(grade_units)


def write_srs_to_disk(curr_srs: str, iter_idx: int, output_dir: str) -> None:
    """
    将 SRS 文档写入磁盘

    Args:
        curr_srs: SRS 文档文本
        iter_idx: 迭代编号（从 1 开始）
        output_dir: 输出目录路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 生成文件名
    filename = os.path.join(output_dir, f"srs_iter_{iter_idx}.md")

    # 写入文件
    with open(filename, "w", encoding="utf-8") as f:
        f.write(curr_srs)

    logger.info(f"已写入 SRS 文档: {filename}")


def write_units_to_disk(units: List[SemanticUnit], filename: str, output_dir: str) -> None:
    """
    将语义单元列表保存到文本文件（每行一个语义单元文本）

    Args:
        units: 语义单元列表
        filename: 文件名
        output_dir: 输出目录路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 生成文件路径
    filepath = os.path.join(output_dir, filename)

    # 写入文件（每行一个语义单元文本）
    with open(filepath, "w", encoding="utf-8") as f:
        for unit in units:
            f.write(unit.text + "\n")

    logger.info(f"已保存 {len(units)} 个语义单元到: {filepath}")


def write_pool_to_disk(pool: List[SemanticUnit], iter_name: str, output_dir: str) -> None:
    """
    将需求池保存到文本文件（仅保存 grade > 0 的语义单元文本内容）

    Args:
        pool: 需求池（包含已评分的语义单元）
        iter_name: 迭代名称（迭代编号，如 "1", "2"）
        output_dir: 输出目录路径
    """
    # 筛选出 grade > 0 的单元
    grade_units = [u for u in pool if u.grade is not None and u.grade > 0]
    
    if len(grade_units) == 0:
        logger.warning("需求池中没有 grade > 0 的语义单元，跳过保存")
        return

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 生成文件路径
    filepath = os.path.join(output_dir, f"pool_iter_{iter_name}.txt")

    # 写入文件（每行一个语义单元文本，不包含grade）
    with open(filepath, "w", encoding="utf-8") as f:
        for unit in grade_units:
            f.write(unit.text + "\n")

    logger.info(f"已保存 {len(grade_units)} 个已采纳需求（grade > 0）到: {filepath}")


def merge_units(
    pool: List[SemanticUnit], new_units: List[SemanticUnit]
) -> List[SemanticUnit]:
    """
    合并两个语义单元列表，去重（基于 text）

    Args:
        pool: 已有需求池
        new_units: 新需求列表

    Returns:
        合并后的需求池（去重）
    """
    # 使用字典去重（key 为 text）
    unit_dict = {}

    # 先添加已有需求
    for unit in pool:
        unit_dict[unit.text] = unit

    # 再添加新需求（如果 text 已存在，保留已有的，因为已有需求可能已经评分）
    for unit in new_units:
        if unit.text not in unit_dict:
            unit_dict[unit.text] = unit

    return list(unit_dict.values())


def save_iteration_stats(
    output_dir: str,
    iter_num: int,
    pool: List[SemanticUnit],
    buffer_new_units: List[SemanticUnit],
    improved_units_count: int = 0,
) -> None:
    """
    保存迭代统计数据到JSON文件
    
    Args:
        output_dir: 输出目录路径
        iter_num: 迭代编号
        pool: 需求池（包含所有grade的单元）
        buffer_new_units: 本轮新探索的需求
        improved_units_count: 改进的需求数量
    """
    # 计算统计数据
    pool_size = len(pool)
    semantic_units_count = len([u for u in pool if u.grade is not None and u.grade > 0])
    new_units_count = len(buffer_new_units)
    
    # 构建统计数据
    stats = {
        "iter_num": iter_num,
        "pool_size": pool_size,
        "semantic_units_count": semantic_units_count,
        "new_units_count": new_units_count,
        "improved_units_count": improved_units_count,
    }
    
    # JSON文件路径
    stats_file = os.path.join(output_dir, "all_iter_stats.json")
    
    # 读取现有数据（如果存在）
    all_stats = {}
    if os.path.exists(stats_file):
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                all_stats = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"读取统计数据文件失败，将创建新文件: {e}")
            all_stats = {}
    
    # 确保iterations键存在
    if "iterations" not in all_stats:
        all_stats["iterations"] = {}
    
    # 更新当前迭代的统计数据
    all_stats["iterations"][str(iter_num)] = stats
    
    # 写入文件
    try:
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(all_stats, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存第 {iter_num} 轮迭代统计数据到: {stats_file}")
    except IOError as e:
        logger.error(f"保存统计数据文件失败: {e}")


def run_srs_iteration(
    d_orig: str,
    r_base: str,
    d_base: str,
    output_dir: str,
    rho: float = None,
    max_outer_iter: int = None,
    max_inner_iter: int = None,
    skip_requirement_improver: bool = None,
) -> None:
    """
    主函数：运行多版本 SRS 生成迭代

    Args:
        d_orig: 原始 SRS 文档
        r_base: 需求基础文档（requirement_base）
        d_base: 基线方法生成的 SRS 文档（baseline SRS）
        output_dir: 输出目录路径
        rho: 每轮探索新需求的比例（相对于基线单元数量），如果为 None 则使用配置文件中的值
        max_outer_iter: 最大外循环次数，如果为 None 则使用配置文件中的值
        max_inner_iter: 最大内循环次数，如果为 None 则使用配置文件中的值
        skip_requirement_improver: 是否跳过 requirement_improver，如果为 None 则使用配置文件中的值
    """
    # 从配置文件获取默认值
    if rho is None:
        rho = config.rho
    if max_outer_iter is None:
        max_outer_iter = config.max_outer_iter
    if max_inner_iter is None:
        max_inner_iter = config.max_inner_iter
    if skip_requirement_improver is None:
        skip_requirement_improver = config.skip_requirement_improver

    # 验证配置
    config.validate()

    logger.info("=" * 60)
    logger.info("开始运行多版本 SRS 生成迭代")
    logger.info(f"输出目录: {output_dir}")
    logger.info(
        f"rho: {rho}, max_outer_iter: {max_outer_iter}, max_inner_iter: {max_inner_iter}, skip_requirement_improver: {skip_requirement_improver}"
    )
    logger.info("=" * 60)

    # -------------------------
    # 1. 基线 SRS -> 语义单元 -> 初始化需求池
    # -------------------------
    logger.info("\n[步骤 1] 拆分基线 SRS 为语义单元...")
    baseline_units = split_to_semantic_units(d_base)
    logger.info(f"基线 SRS 拆分为 {len(baseline_units)} 个语义单元")

    # 保存未评分的基线语义单元
    write_units_to_disk(baseline_units, "no-explore-clarify.txt", output_dir)
    
    # 使用未评分的语义单元生成SRS文档
    logger.info("\n[步骤 1.5] 生成 srs_no-explore-clarify.md...")
    srs_no_explore_clarify = srs_generator(baseline_units)
    with open(os.path.join(output_dir, "srs_no-explore-clarify.md"), "w", encoding="utf-8") as f:
        f.write(srs_no_explore_clarify)
    logger.info(f"已写入 SRS 文档: {os.path.join(output_dir, 'srs_no-explore-clarify.md')}")

    logger.info("\n[步骤 2] 对基线单元进行评分...")
    pool = []
    clarified_baseline_units = requirement_clarifier(baseline_units, d_orig)
    pool = merge_units(pool, clarified_baseline_units)
    logger.info(f"需求池初始化完成，当前大小: {len(pool)}")

    # 计算每轮需要探索的新需求数量
    N = math.ceil(rho * len(baseline_units))
    logger.info(
        f"每轮目标探索 {N} 条新需求（rho={rho} * {len(baseline_units)} 基线单元）"
    )

    # -------------------------
    # 2. 外层循环：每一轮 = 探索 + 澄清 + 更新池 + 生成一版 SRS
    # -------------------------
    outer_iter = 1

    while outer_iter <= max_outer_iter:
        logger.info("\n" + "=" * 60)
        logger.info(f"[外循环] 第 {outer_iter}/{max_outer_iter} 轮迭代")
        logger.info("=" * 60)

        # 获取配置参数
        max_context_length = config.max_context_length
        one_gen_req_token = config.one_gen_req_token
        max_token = config.get_component_max_tokens("requirement_explorer")
        
        if max_context_length is None:
            raise ValueError("配置文件中缺少 max_context_length")
        if one_gen_req_token is None:
            raise ValueError("配置文件中缺少 one_gen_req_token")

        # 记录改进需求数量
        improved_units_count = 0
        buffer_improved_units = []
        
        # 后续轮次：先执行 requirement_improver（基于当前轮次开始时的需求池）
        if outer_iter > 1:
            if skip_requirement_improver:
                logger.info(f"\n[后续轮次] 第 {outer_iter} 轮：跳过 requirement_improver，直接执行需求探索")
            else:
                logger.info(f"\n[后续轮次] 第 {outer_iter} 轮：先执行需求生成，再执行需求探索")
                # 筛选积极需求
                positive_units = [u for u in pool if u.grade is not None and u.grade > 0]
                # 使用整个 pool 作为 negative_pool（用于避免重复和避免负分需求）
                negative_pool = pool
                
                logger.info(f"\n[步骤 2] 发现 {len(positive_units)} 个积极需求，开始生成新需求和扩展积极需求...")
                
                # 计算 input_tokens（用于 requirement_improver）
                template_improver = load_prompt_template("requirement_improver")
                positive_texts = [f"REQ: {unit.text}" for unit in positive_units]
                positive_units_text = "\n".join(positive_texts) if positive_texts else "（暂无积极需求）"
                negative_texts = [unit.text for unit in negative_pool]
                negative_pool_text = "\n".join(negative_texts) if negative_texts else "（暂无已有需求）"
                
                prompt_improver = template_improver.replace("{{R_BASE}}", r_base)
                prompt_improver = prompt_improver.replace("{{POSITIVE_UNITS}}", positive_units_text)
                prompt_improver = prompt_improver.replace("{{NEGATIVE_POOL}}", negative_pool_text)
                
                messages_improver = [{"role": "user", "content": prompt_improver}]
                input_tokens_improver = count_tokens(messages_improver)
                if input_tokens_improver is None:
                    logger.warning("无法计算 requirement_improver 的 input_tokens，使用估算值")
                    input_tokens_improver = int(len(prompt_improver) * 0.25)
                
                # 获取 max_token（优先使用 requirement_improver 的配置）
                max_token_improver = config.get_component_max_tokens("requirement_improver")
                if max_token_improver is None:
                    max_token_improver = max_token
                
                # 调用 requirement_improver（即使没有积极需求，也可以生成新需求）
                improved_units = requirement_improver(
                    positive_units,
                    negative_pool,
                    r_base,
                    input_tokens_improver,
                    max_context_length,
                    one_gen_req_token,
                    max_token_improver,
                    max_inner_iter,
                )
                
                if len(improved_units) > 0:
                    # requirement_improver 内部已经使用相似度检查去重，无需再次过滤
                    buffer_improved_units = improved_units
                    improved_units_count = len(buffer_improved_units)
                    logger.info(f"生成完成，共生成 {len(improved_units)} 个新需求（已在迭代过程中进行相似度去重）")
                else:
                    buffer_improved_units = []
                    logger.warning("未生成任何新需求")

        # requirement_explorer 内部已经处理了迭代逻辑（最多5次），这里只需要调用一次
        if outer_iter == 1:
            logger.info("\n[第一轮] 跳过需求改进，直接执行需求探索")
        logger.info("\n[步骤 2.5] 探索新需求...")

        # 使用 pool 作为负样本
        negative_pool = pool

        # 计算 input_tokens
        template = load_prompt_template("requirement_explorer")
        negative_texts = [unit.text for unit in negative_pool]
        negative_pool_text = "\n".join(negative_texts)
        prompt = template.replace("{{R_BASE}}", r_base)
        prompt = prompt.replace("{{NEGATIVE_POOL}}", negative_pool_text)
        if "{{REQUIRED_NUM}}" in prompt:
            prompt = prompt.replace("{{REQUIRED_NUM}}", "")
        
        messages = [{"role": "user", "content": prompt}]
        input_tokens = count_tokens(messages)
        if input_tokens is None:
            logger.warning("无法计算 input_tokens，使用估算值")
            # 粗略估算：每个字符约 0.25 token
            input_tokens = int(len(prompt) * 0.25)
        
        logger.info(f"计算得到 input_tokens: {input_tokens}")

        # 探索新需求（基于 token 数，内部会迭代 max_inner_iter 次）
        explored_units = requirement_explorer(
            r_base,
            negative_pool,
            input_tokens,
            max_context_length,
            one_gen_req_token,
            max_token,
            max_inner_iter,
        )

        # requirement_explorer 内部已经使用相似度检查去重，无需再次过滤
        buffer_new_units = explored_units

        logger.info(
            f"探索完成，获得 {len(buffer_new_units)} 条新需求（已在迭代过程中进行相似度去重）"
        )

        has_new_units = len(buffer_new_units) > 0
        if not has_new_units:
            logger.info(
                "未获得任何新需求，但根据 max_outer_iter 要求将继续执行本轮迭代并生成 SRS"
            )

        # [外循环] 第 1 轮迭代评分前，保存已过滤但未评分的探索语义单元和需求池中已有的语义单元
        if outer_iter == 1 and has_new_units:
            logger.info("\n[第1轮迭代评分前] 保存已过滤但未评分的探索语义单元和需求池...")
            # 合并已过滤的探索语义单元和需求池中已有的语义单元
            no_clarify_units = merge_units(pool, buffer_new_units)
            write_units_to_disk(no_clarify_units, "no-clarify.txt", output_dir)
            
            # 使用相同的语义单元集合生成SRS文档
            logger.info("\n[第1轮迭代评分前] 生成 srs_no-clarify.md...")
            srs_no_clarify = srs_generator(no_clarify_units)
            with open(os.path.join(output_dir, "srs_no-clarify.md"), "w", encoding="utf-8") as f:
                f.write(srs_no_clarify)
            logger.info(f"已写入 SRS 文档: {os.path.join(output_dir, 'srs_no-clarify.md')}")

        # 第一轮：只对新需求评分
        if outer_iter == 1:
            if has_new_units:
                # 对新需求进行评分
                logger.info(f"\n[第一轮] 仅对新探索的需求进行评分...")
                clarified_new_units = requirement_clarifier(buffer_new_units, d_orig)

                # 更新需求池
                pool = merge_units(pool, clarified_new_units)
                logger.info(f"需求池更新完成，当前大小: {len(pool)}")
            else:
                logger.info("\n[步骤 3] 本轮无新需求可评分，需求池保持不变")
        # 后续轮次：合并新生成需求和新探索需求后统一评分
        else:
            # requirement_explorer 和 requirement_improver 内部已经使用相似度检查去重
            # 由于它们都检查了与 negative_pool 的相似度，而 negative_pool 包含了所有已有需求
            # 所以 improved_units 和 explored_units 之间理论上不应该有重复
            # 但为了保险起见，仍然记录一下
            if len(buffer_improved_units) > 0 and len(buffer_new_units) > 0:
                logger.info(f"\n[后续轮次] 新生成需求: {len(buffer_improved_units)}, 新探索需求: {len(buffer_new_units)}（已在各自迭代过程中进行相似度去重）")
            
            # 合并新生成需求和新探索需求
            all_new_units = buffer_improved_units + buffer_new_units
            
            if len(all_new_units) > 0:
                # 统一对合并后的需求进行评分
                logger.info(f"\n[后续轮次] 合并新生成需求和新探索需求后统一评分（新生成需求: {len(buffer_improved_units)}, 新探索需求: {len(buffer_new_units)}）...")
                clarified_all_units = requirement_clarifier(all_new_units, d_orig)

                # 更新需求池
                pool = merge_units(pool, clarified_all_units)
                logger.info(f"需求池更新完成，当前大小: {len(pool)}")
            else:
                logger.info("\n[步骤 3] 本轮无新生成需求和新探索需求可评分，需求池保持不变")

        # 保存需求池（仅grade > 0）
        write_pool_to_disk(pool, str(outer_iter), output_dir)
        
        # 保存迭代统计数据
        save_iteration_stats(
            output_dir,
            outer_iter,
            pool,
            buffer_new_units,
            improved_units_count,
        )
        
        # 每次外循环结束后：基于当前 pool 生成一个 SRS 版本
        logger.info(f"\n[步骤 4] 生成第 {outer_iter} 版 SRS 文档...")
        curr_srs = generate_srs_from_pool(pool)

        # 写入磁盘文件
        write_srs_to_disk(curr_srs, outer_iter, output_dir)
        


        logger.info(f"\n[外循环] 第 {outer_iter} 轮迭代完成")
        outer_iter += 1

        # 如果新需求数量少于目标，仅记录日志并继续迭代
        if len(buffer_new_units) < N:
            logger.info(
                f"新需求数量 ({len(buffer_new_units)}) 少于目标 ({N})，继续执行后续迭代"
            )

    # -------------------------
    # 3. 过程结束
    # -------------------------
    logger.info("\n" + "=" * 60)
    logger.info("多版本 SRS 生成迭代完成")
    logger.info(f"共生成 {outer_iter - 1} 版 SRS 文档")
    logger.info(f"最终需求池大小: {len(pool)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多版本 SRS 生成器")
    parser.add_argument(
        "--d-orig",
        dest="d_orig",
        help="原始 SRS 文档的文件路径或直接提供内容",
    )
    parser.add_argument(
        "--r-base",
        dest="r_base",
        help="需求基础文档的文件路径或直接提供内容",
    )
    parser.add_argument(
        "--d-base",
        dest="d_base",
        help="基线 SRS 文档的文件路径或直接提供内容",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="输出目录（默认：配置文件 outputs.dir 或 ./output）",
    )
    parser.add_argument("--rho", type=float, help="覆盖配置文件中的 rho")
    parser.add_argument(
        "--max-outer-iter", type=int, help="覆盖配置文件中的 max_outer_iter"
    )
    parser.add_argument(
        "--max-inner-iter", type=int, help="覆盖配置文件中的 max_inner_iter"
    )
    parser.add_argument(
        "--skip-requirement-improver",
        dest="skip_requirement_improver",
        action="store_true",
        help="跳过 requirement_improver 的执行（覆盖配置文件中的设置）",
    )
    args = parser.parse_args()

    def _load_text_from_candidate(candidate: str) -> str:
        path = Path(candidate).expanduser()
        if path.exists():
            return path.read_text(encoding="utf-8")
        return candidate

    def _load_input(name: str, arg_value: str) -> str:
        if arg_value:
            return _load_text_from_candidate(arg_value)
        config_path = config.get(f"inputs.{name}_path")
        if config_path:
            path = Path(config_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(
                    f"配置中的 inputs.{name}_path 指向的文件不存在: {path}"
                )
            return path.read_text(encoding="utf-8")
        config_text = config.get(f"inputs.{name}")
        if config_text:
            return config_text
        raise ValueError(
            f"缺少 {name} 内容，请通过命令行参数或配置文件 inputs.{name}(_path) 提供"
        )

    try:
        d_orig = _load_input("d_orig", args.d_orig)
        r_base = _load_input("r_base", args.r_base)
        d_base = _load_input("d_base", args.d_base)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    output_dir = (
        args.output_dir
        or config.get("outputs.dir")
        or config.get("output.dir")
        or config.get("output_dir")
        or "./output"
    )

    run_srs_iteration(
        d_orig=d_orig,
        r_base=r_base,
        d_base=d_base,
        output_dir=output_dir,
        rho=args.rho,
        max_outer_iter=args.max_outer_iter,
        max_inner_iter=args.max_inner_iter,
        skip_requirement_improver=args.skip_requirement_improver if args.skip_requirement_improver else None,
    )
