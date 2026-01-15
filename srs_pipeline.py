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


def _is_similar_to_any(
    candidate: SemanticUnit,
    existing_units: List[SemanticUnit],
    threshold: float,
) -> bool:
    """
    检查候选需求是否与已有需求列表中的任何需求相似
    
    Args:
        candidate: 候选需求
        existing_units: 已有需求列表
        threshold: 相似度阈值
    
    Returns:
        如果相似度 >= threshold，返回 True；否则返回 False
    """
    # 为候选需求计算向量（如果还没有）
    if candidate.vector is None:
        candidate.vector = get_embedding(candidate.text)
    
    # 检查与已有需求的相似度
    for existing in existing_units:
        # 为已有需求计算向量（如果还没有）
        if existing.vector is None:
            existing.vector = get_embedding(existing.text)
        
        similarity = cosine_similarity(candidate.vector, existing.vector)
        if similarity >= threshold:
            return True
    
    return False


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
    input_tokens: int,
    max_context_length: int,
    one_gen_req_token: int,
    max_token: Optional[int],
    max_inner_iter: int = 5,
) -> List[SemanticUnit]:
    """
    探索新需求（基于 token 数的迭代生成）
    
    Args:
        r_base: 需求基础文档
        negative_pool: 已有需求列表（负样本）
        input_tokens: 当前 prompt 的 token 数
        max_context_length: 最大上下文长度
        one_gen_req_token: 单次生成需求的目标 token 数
        max_token: 最大 token 数限制（从配置获取，可能为 None）
        max_inner_iter: 最大内循环次数
    
    Returns:
        新探索的语义单元列表（已计算 token_count）
    """
    logger.info(f"开始探索新需求，input_tokens={input_tokens}, max_context_length={max_context_length}, one_gen_req_token={one_gen_req_token}, max_token={max_token}, max_inner_iter={max_inner_iter}")
    
    # 计算目标 token 数
    if max_token is not None:
        target_tokens = min(max_context_length - input_tokens, one_gen_req_token, max_token)
    else:
        target_tokens = min(max_context_length - input_tokens, one_gen_req_token)
    
    logger.info(f"目标 token 数: {target_tokens}")
    
    # 加载模板
    template = load_prompt_template("requirement_explorer")
    
    # 准备已有需求列表（只取 text）
    negative_texts = [unit.text for unit in negative_pool]
    negative_pool_text = "\n".join(negative_texts)
    
    # 替换占位符（不再使用 REQUIRED_NUM）
    prompt = template.replace("{{R_BASE}}", r_base)
    prompt = prompt.replace("{{NEGATIVE_POOL}}", negative_pool_text)
    # 移除 REQUIRED_NUM 占位符（如果存在）
    if "{{REQUIRED_NUM}}" in prompt:
        prompt = prompt.replace("{{REQUIRED_NUM}}", "")
    
    # 获取组件特定的模型和温度配置
    model = config.get_component_model("requirement_explorer")
    temperature = config.get_component_temperature("requirement_explorer")
    max_continuations = config.get_component_max_continuations("requirement_explorer")
    
    # 迭代生成逻辑
    current_tokens = 0
    unique_units = []
    # 获取相似度阈值
    similarity_threshold = config.similarity_threshold
    
    # 为 negative_pool 中的需求准备向量（用于相似度检查）
    for unit in negative_pool:
        if unit.vector is None:
            unit.vector = get_embedding(unit.text)
    
    for iteration in range(max_inner_iter):
        logger.info(f"[requirement_explorer 迭代] 第 {iteration + 1}/{max_inner_iter} 次，当前 token 数: {current_tokens}/{target_tokens}")
        
        def _invoke():
            # 调用 LLM，将 target_tokens 作为 max_tokens 传入
            response = call_llm(
                prompt,
                model=model,
                temperature=temperature,
                max_tokens=target_tokens,  # 使用计算出的 target_tokens
                max_continuations=max_continuations,
            )
            unit_texts = _parse_req_prefixed_lines(response)
            
            # 去掉最后一条需求（避免不完整）
            if len(unit_texts) > 0:
                unit_texts = unit_texts[:-1]
                logger.info(f"[requirement_explorer 迭代] 生成 {len(unit_texts)} 条需求（已移除最后一条）")
            
            units_local = []
            for text in unit_texts:
                unit = SemanticUnit(
                    text=text,
                    grade=None,
                    vector=None,
                    token_count=None,
                )
                # 计算 token 数
                unit.calculate_token_count()
                units_local.append(unit)
            return units_local
        
        try:
            generated = _run_with_component_retry("requirement_explorer", _invoke)
        except Exception as e:
            logger.error(f"[requirement_explorer 迭代] 第 {iteration + 1} 次迭代失败: {e}")
            if iteration == 0:
                # 第一次迭代失败，直接返回空列表
                return []
            # 其他迭代失败，使用已有结果
            break
        
        # 使用相似度检查去重并累计 token
        new_units_this_iter = []
        duplicate_count = 0
        for unit in generated:
            # 检查与 negative_pool 和本次迭代已生成需求的相似度
            all_existing = negative_pool + unique_units
            if not _is_similar_to_any(unit, all_existing, similarity_threshold):
                unique_units.append(unit)
                if unit.token_count is not None:
                    current_tokens += unit.token_count
                new_units_this_iter.append(unit)
            else:
                duplicate_count += 1
        
        if duplicate_count > 0:
            logger.info(f"[requirement_explorer 迭代] 本次生成 {len(generated)} 条需求，其中 {duplicate_count} 条相似（与已有需求或本次迭代重复，相似度阈值: {similarity_threshold}），新增 {len(new_units_this_iter)} 条未重复需求，累计 token 数: {current_tokens}/{target_tokens}")
        else:
            logger.info(f"[requirement_explorer 迭代] 本次新增 {len(new_units_this_iter)} 条未重复需求，累计 token 数: {current_tokens}/{target_tokens}")
        
        # 如果达到目标 token 数，停止迭代
        if current_tokens >= target_tokens:
            logger.info(f"[requirement_explorer 迭代] 已达到目标 token 数 ({current_tokens} >= {target_tokens})，停止迭代")
            break
        
        # 第5次迭代如果未达到，允许计入部分相似需求补足（使用更宽松的阈值）
        if iteration == 4 and current_tokens < target_tokens:
            logger.info(f"[requirement_explorer 迭代] 第5次迭代仍未达到目标，当前 {current_tokens}/{target_tokens}，尝试补足...")
            remaining = target_tokens - current_tokens
            # 使用更宽松的相似度阈值（0.9）来补足
            relaxed_threshold = 0.9
            # 从本次生成的需求中，找出那些与已有需求池不相似，但与本次迭代已生成需求相似的需求
            for unit in generated:
                # 首先检查是否与已有需求池相似（使用原始阈值），如果相似则跳过
                if _is_similar_to_any(unit, negative_pool, similarity_threshold):
                    continue
                
                # 检查是否与本次迭代已生成的需求相似（使用更宽松的阈值）
                if _is_similar_to_any(unit, unique_units, relaxed_threshold):
                    # 与本次迭代内需求相似，但与已有需求池不相似，可以添加用于补足
                    if unit.token_count is not None:
                        if current_tokens + unit.token_count <= target_tokens + remaining * 0.1:  # 允许超出10%
                            unique_units.append(unit)
                            current_tokens += unit.token_count
                            if current_tokens >= target_tokens:
                                break
            logger.info(f"[requirement_explorer 迭代] 补足后 token 数: {current_tokens}/{target_tokens}")
    
    logger.info(f"[requirement_explorer] 迭代完成，共生成 {len(unique_units)} 个新需求，总 token 数: {current_tokens}/{target_tokens}")
    return unique_units


def requirement_improver(
    positive_units: List[SemanticUnit],
    negative_pool: List[SemanticUnit],
    r_base: str,
    input_tokens: int,
    max_context_length: int,
    one_gen_req_token: int,
    max_token: Optional[int],
    max_inner_iter: int = 5,
) -> List[SemanticUnit]:
    """
    生成新需求和扩展积极需求（基于 token 数的迭代生成）
    
    Args:
        positive_units: 积极需求列表（grade > 0），用于扩展
        negative_pool: 负样本池（包含所有已有需求），用于避免重复和避免负分需求
        r_base: 需求基础文档
        input_tokens: 当前 prompt 的 token 数
        max_context_length: 最大上下文长度
        one_gen_req_token: 单次生成需求的目标 token 数
        max_token: 最大 token 数限制（从配置获取，可能为 None）
        max_inner_iter: 最大内循环次数
    
    Returns:
        新生成的语义单元列表（已计算 token_count）
    """
    logger.info(f"开始生成新需求和扩展积极需求，积极需求数量: {len(positive_units)}, 负样本池大小: {len(negative_pool)}, input_tokens={input_tokens}, max_context_length={max_context_length}, one_gen_req_token={one_gen_req_token}, max_token={max_token}, max_inner_iter={max_inner_iter}")
    
    # 计算目标 token 数
    if max_token is not None:
        target_tokens = min(max_context_length - input_tokens, one_gen_req_token, max_token)
    else:
        target_tokens = min(max_context_length - input_tokens, one_gen_req_token)
    
    logger.info(f"目标 token 数: {target_tokens}")
    
    # 加载模板
    template = load_prompt_template("requirement_improver")
    
    # 准备积极需求文本列表
    positive_texts = [f"REQ: {unit.text}" for unit in positive_units]
    positive_units_text = "\n".join(positive_texts) if positive_texts else "（暂无积极需求）"
    
    # 准备负样本池文本列表（只取 text）
    negative_texts = [unit.text for unit in negative_pool]
    negative_pool_text = "\n".join(negative_texts) if negative_texts else "（暂无已有需求）"
    
    # 替换占位符
    prompt = template.replace("{{R_BASE}}", r_base)
    prompt = prompt.replace("{{POSITIVE_UNITS}}", positive_units_text)
    prompt = prompt.replace("{{NEGATIVE_POOL}}", negative_pool_text)
    
    # 获取组件特定的模型和温度配置
    # 优先使用 requirement_improver 的配置，如果不存在则使用 requirement_explorer 的配置
    if config.get("openai.components.requirement_improver.model") is not None:
        model = config.get_component_model("requirement_improver")
    else:
        model = config.get_component_model("requirement_explorer")
    
    if config.get("openai.components.requirement_improver.temperature") is not None:
        temperature = config.get_component_temperature("requirement_improver")
    else:
        temperature = config.get_component_temperature("requirement_explorer")
    
    if config.get("openai.components.requirement_improver.max_continuations") is not None:
        max_continuations = config.get_component_max_continuations("requirement_improver")
    else:
        max_continuations = config.get_component_max_continuations("requirement_explorer")
    
    # 迭代生成逻辑
    current_tokens = 0
    unique_units = []
    # 获取相似度阈值
    similarity_threshold = config.similarity_threshold
    
    # 为 negative_pool 中的需求准备向量（用于相似度检查）
    for unit in negative_pool:
        if unit.vector is None:
            unit.vector = get_embedding(unit.text)
    
    for iteration in range(max_inner_iter):
        logger.info(f"[requirement_improver 迭代] 第 {iteration + 1}/{max_inner_iter} 次，当前 token 数: {current_tokens}/{target_tokens}")
        
        def _invoke():
            # 调用 LLM，将 target_tokens 作为 max_tokens 传入
            response = call_llm(
                prompt,
                model=model,
                temperature=temperature,
                max_tokens=target_tokens,  # 使用计算出的 target_tokens
                max_continuations=max_continuations,
            )
            unit_texts = _parse_req_prefixed_lines(response)
            
            # 去掉最后一条需求（避免不完整）
            if len(unit_texts) > 0:
                unit_texts = unit_texts[:-1]
                logger.info(f"[requirement_improver 迭代] 生成 {len(unit_texts)} 条新需求（已移除最后一条）")
            
            units_local = []
            for text in unit_texts:
                unit = SemanticUnit(
                    text=text,
                    grade=None,  # 改进后的需求需要重新评分
                    vector=None,
                    token_count=None,
                )
                # 计算 token 数
                unit.calculate_token_count()
                units_local.append(unit)
            return units_local
        
        try:
            generated = _run_with_component_retry("requirement_improver", _invoke)
        except Exception as e:
            logger.error(f"[requirement_improver 迭代] 第 {iteration + 1} 次迭代失败: {e}")
            if iteration == 0:
                # 第一次迭代失败，直接返回空列表
                return []
            # 其他迭代失败，使用已有结果
            break
        
        # 使用相似度检查去重并累计 token
        new_units_this_iter = []
        duplicate_count = 0
        for unit in generated:
            # 检查与 negative_pool 和本次迭代已生成需求的相似度
            all_existing = negative_pool + unique_units
            if not _is_similar_to_any(unit, all_existing, similarity_threshold):
                unique_units.append(unit)
                if unit.token_count is not None:
                    current_tokens += unit.token_count
                new_units_this_iter.append(unit)
            else:
                duplicate_count += 1
        
        if duplicate_count > 0:
            logger.info(f"[requirement_improver 迭代] 本次生成 {len(generated)} 条需求，其中 {duplicate_count} 条相似（与已有需求或本次迭代重复，相似度阈值: {similarity_threshold}），新增 {len(new_units_this_iter)} 条未重复新需求，累计 token 数: {current_tokens}/{target_tokens}")
        else:
            logger.info(f"[requirement_improver 迭代] 本次新增 {len(new_units_this_iter)} 条未重复新需求，累计 token 数: {current_tokens}/{target_tokens}")
        
        # 如果达到目标 token 数，停止迭代
        if current_tokens >= target_tokens:
            logger.info(f"[requirement_improver 迭代] 已达到目标 token 数 ({current_tokens} >= {target_tokens})，停止迭代")
            break
        
        # 第5次迭代如果未达到，允许计入部分相似需求补足（使用更宽松的阈值）
        if iteration == 4 and current_tokens < target_tokens:
            logger.info(f"[requirement_improver 迭代] 第5次迭代仍未达到目标，当前 {current_tokens}/{target_tokens}，尝试补足...")
            remaining = target_tokens - current_tokens
            # 使用更宽松的相似度阈值（0.9）来补足
            relaxed_threshold = 0.9
            # 从本次生成的需求中，找出那些与已有需求池不相似，但与本次迭代已生成需求相似的需求
            for unit in generated:
                # 首先检查是否与已有需求池相似（使用原始阈值），如果相似则跳过
                if _is_similar_to_any(unit, negative_pool, similarity_threshold):
                    continue
                
                # 检查是否与本次迭代已生成的需求相似（使用更宽松的阈值）
                if _is_similar_to_any(unit, unique_units, relaxed_threshold):
                    # 与本次迭代内需求相似，但与已有需求池不相似，可以添加用于补足
                    if unit.token_count is not None:
                        if current_tokens + unit.token_count <= target_tokens + remaining * 0.1:  # 允许超出10%
                            unique_units.append(unit)
                            current_tokens += unit.token_count
                            if current_tokens >= target_tokens:
                                break
            logger.info(f"[requirement_improver 迭代] 补足后 token 数: {current_tokens}/{target_tokens}")
    
    logger.info(f"[requirement_improver] 迭代完成，共生成 {len(unique_units)} 个新需求，总 token 数: {current_tokens}/{target_tokens}")
    return unique_units


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
