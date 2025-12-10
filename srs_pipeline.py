# srs_pipeline.py
import json
import logging
import math
import re
import time
from typing import Callable, List, Optional, TypeVar
from models import SemanticUnit
from openai_utils import (
    load_prompt_template,
    call_llm,
    get_embedding,
    parse_json_response,
)
from config import get_config

# 配置日志
logger = logging.getLogger(__name__)

# 获取配置
config = get_config()

T = TypeVar("T")


def _parse_req_prefixed_lines(response: str) -> List[str]:
    """
    将 LLM 输出中以 REQ: 开头的行解析为需求文本。
    """
    req_pattern = re.compile(r"^\s*REQ\s*[:：]\s*(.+)$")
    units: List[str] = []
    current: Optional[str] = None
    for raw_line in response.splitlines():
        line = raw_line.rstrip()
        match = req_pattern.match(line)
        if match:
            if current:
                units.append(current.strip())
            current = match.group(1).strip()
            continue
        if current is not None:
            stripped = line.strip()
            if stripped:
                current += " " + stripped
    if current:
        units.append(current.strip())
    if not units:
        raise ValueError("LLM 输出中未找到任何以 'REQ:' 开头的需求行")
    return units


def _run_with_component_retry(component_name: str, func: Callable[[], T]) -> T:
    """
    对单个组件执行带重试的调用，任何异常都会触发重试
    """
    max_attempts = max(1, int(config.get(f"robustness.{component_name}.max_attempts", 3)))
    delay_seconds = max(
        0.0, float(config.get(f"robustness.{component_name}.delay_seconds", 2.0))
    )
    
    last_error: Optional[Exception] = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                logger.error(
                    f"{component_name} 连续失败 {attempt} 次，放弃重试。错误: {exc}"
                )
                raise
            logger.warning(
                f"{component_name} 第 {attempt}/{max_attempts} 次失败，将在 {delay_seconds:.1f} 秒后重试。错误: {exc}"
            )
            if delay_seconds > 0:
                time.sleep(delay_seconds)
    
    if last_error:
        raise last_error
    raise RuntimeError(f"{component_name} 未能完成且未记录异常")


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    计算两个向量的余弦相似度
    
    Args:
        a: 向量 a
        b: 向量 b
    
    Returns:
        余弦相似度（0-1 之间）
    """
    if len(a) != len(b):
        raise ValueError("向量长度必须相同")
    
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def split_to_semantic_units(d_base: str) -> List[SemanticUnit]:
    """
    将基线 SRS 文档拆分为语义单元
    
    Args:
        d_base: 基线 SRS 文档文本
    
    Returns:
        语义单元列表
    """
    logger.info("开始拆分基线 SRS 为语义单元...")
    
    # 加载模板
    template = load_prompt_template("split_to_semantic_units")
    
    # 替换占位符
    prompt = template.replace("{{D_BASE}}", d_base)
    
    # 获取组件特定的模型和温度配置
    model = config.get_component_model("split_to_semantic_units")
    temperature = config.get_component_temperature("split_to_semantic_units")
    max_tokens = config.get_component_max_tokens("split_to_semantic_units")
    max_continuations = config.get_component_max_continuations("split_to_semantic_units")
    
    def _invoke():
        response = call_llm(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_continuations=max_continuations,
        )
        unit_texts = _parse_req_prefixed_lines(response)
        units_local = []
        for text in unit_texts:
            unit = SemanticUnit(
                text=text,
                grade=None,
                vector=None,
            )
            units_local.append(unit)
        return units_local
    
    units = _run_with_component_retry("split_to_semantic_units", _invoke)
    
    logger.info(f"成功拆分出 {len(units)} 个语义单元")
    return units


def requirement_explorer(
    r_base: str,
    negative_pool: List[SemanticUnit],
    required_num: int,
) -> List[SemanticUnit]:
    """
    探索新需求
    
    Args:
        r_base: 需求基础文档
        negative_pool: 已有需求列表（负样本）
        required_num: 目标新增条数
    
    Returns:
        新探索的语义单元列表
    """
    logger.info(f"开始探索新需求，目标数量: {required_num}")
    
    # 加载模板
    template = load_prompt_template("requirement_explorer")
    
    # 准备已有需求列表（只取 text）
    negative_texts = [unit.text for unit in negative_pool]
    negative_pool_text = "\n".join(negative_texts)
    
    # 替换占位符
    prompt = template.replace("{{R_BASE}}", r_base)
    prompt = prompt.replace("{{NEGATIVE_POOL}}", negative_pool_text)
    prompt = prompt.replace("{{REQUIRED_NUM}}", str(required_num))
    
    # 获取组件特定的模型和温度配置
    model = config.get_component_model("requirement_explorer")
    temperature = config.get_component_temperature("requirement_explorer")
    max_tokens = config.get_component_max_tokens("requirement_explorer")
    max_continuations = config.get_component_max_continuations("requirement_explorer")
    
    def _invoke():
        response = call_llm(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_continuations=max_continuations,
        )
        unit_texts = _parse_req_prefixed_lines(response)
        units_local = []
        for text in unit_texts:
            unit = SemanticUnit(
                text=text,
                grade=None,
                vector=None,
            )
            units_local.append(unit)
        return units_local
    
    units = _run_with_component_retry("requirement_explorer", _invoke)
    
    logger.info(f"成功探索出 {len(units)} 个新需求")
    return units


def requirement_clarifier(
    units: List[SemanticUnit],
    d_orig: str,
) -> List[SemanticUnit]:
    """
    对语义单元进行评分
    
    Args:
        units: 待评分的语义单元列表
        d_orig: 原始 SRS 文档
    
    Returns:
        已评分的语义单元列表（grade 已更新）
    """
    logger.info(f"开始对 {len(units)} 个语义单元进行评分...")
    
    # 加载模板
    template = load_prompt_template("requirement_clarifier")
    
    # 准备带 id 的单元列表（用于 LLM 输入）
    units_with_id = [
        {"id": i + 1, "text": unit.text}
        for i, unit in enumerate(units)
    ]
    units_json = json.dumps({"units": units_with_id}, ensure_ascii=False, indent=2)
    
    # 替换占位符
    prompt = template.replace("{{D_ORIG}}", d_orig)
    prompt = prompt.replace("{{UNITS_JSON}}", units_json)
    
    # 获取组件特定的模型和温度配置
    model = config.get_component_model("requirement_clarifier")
    temperature = config.get_component_temperature("requirement_clarifier")
    max_tokens = config.get_component_max_tokens("requirement_clarifier")
    max_continuations = config.get_component_max_continuations("requirement_clarifier")
    
    def _invoke():
        response = call_llm(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_continuations=max_continuations,
        )
        result = parse_json_response(response)
        grade_map_local = {}
        for item in result.get("units", []):
            grade_map_local[item["id"]] = item["grade"]
        return grade_map_local
    
    grade_map = _run_with_component_retry("requirement_clarifier", _invoke)
    
    # 更新 units 的 grade（不修改 text 和 vector）
    for i, unit in enumerate(units):
        unit_id = i + 1
        if unit_id in grade_map:
            unit.grade = grade_map[unit_id]
        else:
            logger.warning(f"单元 {unit_id} 未找到对应的评分，设置为 0")
            unit.grade = 0
    
    # 统计评分分布
    grade_dist = {}
    for unit in units:
        grade = unit.grade
        grade_dist[grade] = grade_dist.get(grade, 0) + 1
    
    logger.info(f"评分完成，分布: {grade_dist}")
    return units


def srs_generator(grade_units: List[SemanticUnit]) -> str:
    """
    根据已评分的语义单元生成 SRS 文档
    
    Args:
        grade_units: 已评分的语义单元列表（grade > 0）
    
    Returns:
        SRS 文档文本（Markdown 格式）
    """
    logger.info(f"开始生成 SRS 文档，输入 {len(grade_units)} 个已采纳需求...")
    
    # 加载模板
    template = load_prompt_template("srs_generator")
    
    # 准备单元列表（只包含 text 和 grade）
    units_data = [
        {"text": unit.text, "grade": unit.grade}
        for unit in grade_units
    ]
    units_json = json.dumps({"units": units_data}, ensure_ascii=False, indent=2)
    
    # 替换占位符
    prompt = template.replace("{{UNITS_JSON}}", units_json)
    
    # 获取组件特定的模型和温度配置
    model = config.get_component_model("srs_generator")
    temperature = config.get_component_temperature("srs_generator")
    max_tokens = config.get_component_max_tokens("srs_generator")
    max_continuations = config.get_component_max_continuations("srs_generator")
    
    def _invoke():
        response = call_llm(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_continuations=max_continuations,
        )
        return response.strip()
    
    srs_text = _run_with_component_retry("srs_generator", _invoke)
    
    # 返回生成的 SRS 文本（已经是 Markdown 格式）
    logger.info("SRS 文档生成完成")
    return srs_text


def filter_low_similarity(
    explored_units: List[SemanticUnit],
    existing_pool: List[SemanticUnit],
    threshold: float = None,
) -> List[SemanticUnit]:
    """
    使用向量相似度过滤掉与已有需求过于相似的语义单元
    
    Args:
        explored_units: 待过滤的候选语义单元列表
        existing_pool: 已有需求池
        threshold: 相似度阈值，如果为 None 则使用配置文件中的值，超过此值视为"太像"
    
    Returns:
        保留下来的语义单元列表
    """
    # 从配置文件获取默认值
    if threshold is None:
        threshold = config.similarity_threshold
    
    logger.info(f"开始过滤相似需求，候选数量: {len(explored_units)}, 已有池大小: {len(existing_pool)}, 阈值: {threshold}")
    
    # 1. 为 existing_pool 中的每个单元准备向量
    for unit in existing_pool:
        if unit.vector is None:
            logger.debug(f"为已有需求计算 embedding: {unit.text[:50]}...")
            unit.vector = get_embedding(unit.text)
    
    # 2. 对 explored_units 中每个候选进行过滤
    brand_new_units = []
    
    for candidate in explored_units:
        # 为候选单元计算向量（如果还没有）
        if candidate.vector is None:
            logger.debug(f"为候选需求计算 embedding: {candidate.text[:50]}...")
            candidate.vector = get_embedding(candidate.text)
        
        # 检查与 existing_pool 的相似度
        is_too_similar = False
        for existing in existing_pool:
            if existing.vector is None:
                continue  # 理论上不会发生，但保险起见
            
            similarity = cosine_similarity(candidate.vector, existing.vector)
            if similarity >= threshold:
                logger.debug(
                    f"候选需求与已有需求相似度过高 ({similarity:.3f} >= {threshold}): "
                    f"候选='{candidate.text[:50]}...'"
                )
                is_too_similar = True
                break
        
        if not is_too_similar:
            brand_new_units.append(candidate)
    
    logger.info(f"过滤完成，保留 {len(brand_new_units)} 个新需求")
    return brand_new_units
