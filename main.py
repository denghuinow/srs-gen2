# main.py
import argparse
import os
import logging
import math
from pathlib import Path
from typing import List
from models import SemanticUnit
from srs_pipeline import (
    split_to_semantic_units,
    requirement_explorer,
    requirement_clarifier,
    srs_generator,
    filter_low_similarity,
)
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


def run_srs_iteration(
    d_orig: str,
    r_base: str,
    d_base: str,
    output_dir: str,
    rho: float = None,
    max_outer_iter: int = None,
    max_inner_iter: int = None,
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
    """
    # 从配置文件获取默认值
    if rho is None:
        rho = config.rho
    if max_outer_iter is None:
        max_outer_iter = config.max_outer_iter
    if max_inner_iter is None:
        max_inner_iter = config.max_inner_iter

    # 验证配置
    config.validate()

    logger.info("=" * 60)
    logger.info("开始运行多版本 SRS 生成迭代")
    logger.info(f"输出目录: {output_dir}")
    logger.info(
        f"rho: {rho}, max_outer_iter: {max_outer_iter}, max_inner_iter: {max_inner_iter}"
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
        logger.info(f"[外循环] 第 {outer_iter} 轮迭代")
        logger.info("=" * 60)

        buffer_new_units = []
        inner_iter = 0

        # 内层循环：多次 explorer，尝试凑够 N 条 brand-new requirements
        while len(buffer_new_units) < N and inner_iter < max_inner_iter:
            logger.info(
                f"\n[内循环] 第 {inner_iter + 1} 次探索（目标: {N} 条，当前: {len(buffer_new_units)} 条）"
            )
            # 至少探索10条新需求，否则容易出现探索不出新需求的情况
            required_num = max(10, math.ceil(( N - len(buffer_new_units))*1.5))

            # 合并 pool 和 buffer_new_units 作为负样本
            negative_pool = merge_units(pool, buffer_new_units)

            # 探索新需求
            explored_units = requirement_explorer(
                r_base,
                negative_pool,
                required_num,
            )

            # 过滤相似需求（使用配置文件中的阈值）
            brand_new_units = filter_low_similarity(
                explored_units,
                negative_pool,
            )

            # 合并到 buffer
            buffer_new_units = merge_units(buffer_new_units, brand_new_units)

            logger.info(
                f"本轮探索获得 {len(brand_new_units)} 条新需求，buffer 当前大小: {len(buffer_new_units)}"
            )

            inner_iter += 1

        # ---- 内层结束，检查本轮有没有有效增量 ----
        logger.info(
            f"\n[内循环结束] 共探索 {inner_iter} 次，获得 {len(buffer_new_units)} 条新需求"
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

        if has_new_units:
            # 对新需求进行评分（无论数量是否达到目标）
            logger.info(f"\n[步骤 3] 对新需求进行评分...")
            clarified_new_units = requirement_clarifier(buffer_new_units, d_orig)

            # 更新需求池
            pool = merge_units(pool, clarified_new_units)
            logger.info(f"需求池更新完成，当前大小: {len(pool)}")
        else:
            logger.info("\n[步骤 3] 本轮无新需求可评分，需求池保持不变")

        # 保存需求池（仅grade > 0）
        write_pool_to_disk(pool, str(outer_iter), output_dir)
        
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
    )
