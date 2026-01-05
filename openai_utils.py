# openai_utils.py
import os
import json
import logging
import copy
import time
from typing import Any, Callable, List, Optional
from openai import OpenAI
from config import get_config

# 配置日志
logger = logging.getLogger(__name__)

# 获取配置
config = get_config()

# 初始化 OpenAI 客户端
client_kwargs = {"api_key": config.openai_api_key}
if config.openai_base_url:
    client_kwargs["base_url"] = config.openai_base_url
client = OpenAI(**client_kwargs)


def _execute_with_retry(action_name: str, func: Callable[[], Any]) -> Any:
    """
    通用重试逻辑，应用于所有 OpenAI API 调用
    """
    max_attempts = max(1, config.api_retry_max_attempts)
    base_delay = max(0.0, config.api_retry_base_delay)
    max_delay = max(base_delay, config.api_retry_max_delay)

    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as exc:
            if attempt >= max_attempts:
                logger.error(
                    f"{action_name} 失败（已达到最大重试次数 {max_attempts}）：{exc}"
                )
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            logger.warning(
                f"{action_name} 失败（第 {attempt}/{max_attempts} 次），"
                f"{delay:.2f} 秒后重试。错误：{exc}"
            )
            time.sleep(delay)


def load_prompt_template(name: str, language: str = None) -> str:
    """
    加载提示词模板文件
    
    Args:
        name: 模板文件名（不含扩展名），例如 "split_to_semantic_units"
        language: 语言代码（"zh" 或 "en"），如果为 None 则从配置文件读取
    
    Returns:
        模板文件内容（字符串）
    """
    # 获取语言配置
    if language is None:
        language = config.prompt_language
    
    base_dir = os.path.join(os.path.dirname(__file__), "prompts")
    
    # 优先尝试从语言目录加载
    lang_path = os.path.join(base_dir, language, f"{name}.md")
    
    # 如果指定语言的文件不存在，尝试回退
    if not os.path.exists(lang_path):
        # 如果配置的语言不是默认的，尝试使用默认语言（en）
        if language != "en":
            logger.warning(
                f"提示词文件不存在: {lang_path}，尝试使用默认语言 (en)"
            )
            lang_path = os.path.join(base_dir, "en", f"{name}.md")
        
        # 如果英文版本也不存在，尝试中文版本
        if not os.path.exists(lang_path):
            logger.warning(
                f"提示词文件不存在: {lang_path}，尝试使用中文版本 (zh)"
            )
            lang_path = os.path.join(base_dir, "zh", f"{name}.md")
        
        # 如果都不存在，尝试直接从 prompts 目录加载（向后兼容）
        if not os.path.exists(lang_path):
            logger.warning(
                f"提示词文件不存在: {lang_path}，尝试从 prompts 根目录加载"
            )
            lang_path = os.path.join(base_dir, f"{name}.md")
    
    try:
        with open(lang_path, "r", encoding="utf-8") as f:
            logger.debug(f"成功加载提示词: {lang_path}")
            return f.read()
    except FileNotFoundError:
        logger.error(f"模板文件未找到: {lang_path}")
        raise
    except Exception as e:
        logger.error(f"加载模板文件失败: {lang_path}, 错误: {e}")
        raise


def call_llm(
    prompt: str,
    model: str = None,
    temperature: float = None,
    max_tokens: Optional[int] = None,
    max_continuations: Optional[int] = None,
) -> str:
    """
    调用 OpenAI Chat LLM，支持接续生成
    
    Args:
        prompt: 提示词内容
        model: 模型名称，如果为 None 则使用配置文件中的值
        temperature: 温度参数，如果为 None 则使用配置文件中的值
        max_tokens: 最大输出 token 数，如果为 None 则使用配置文件中的值
        max_continuations: 最大接续次数，如果为 None 则使用配置文件中的值
    
    Returns:
        LLM 返回的完整文本内容（如果被截断会自动接续）
    """
    # 从配置文件获取默认值
    if model is None:
        model = config.chat_model
    if temperature is None:
        temperature = config.chat_temperature
    if max_tokens is None:
        max_tokens = config.chat_max_tokens
    if max_continuations is None:
        max_continuations = config.chat_max_continuations
    
    # 构建原始 messages
    base_messages = [
        {"role": "user", "content": prompt},
    ]
    
    # 构建 API 参数
    api_params = {
        "model": model,
        "messages": base_messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        api_params["max_tokens"] = max_tokens
    
    try:
        # 首次调用
        resp = _execute_with_retry(
            "ChatCompletion",
            lambda: client.chat.completions.create(**api_params),
        )
        result_text = resp.choices[0].message.content or ""
        finish_reason = resp.choices[0].finish_reason
        
        logger.debug(f"API响应 finish_reason: {finish_reason}")
        
        # 如果被截断，尝试接续
        if finish_reason == "length" and max_continuations > 0:
            logger.warning(f"响应达到 max_tokens 被截断，尝试接续生成（最多 {max_continuations} 次）...")
            result_text, finish_reason = _continue_on_truncation(
                api_params, result_text, finish_reason, max_continuations
            )
        
        return result_text
    except Exception as e:
        logger.error(f"调用 LLM 失败: {e}")
        raise


def get_embedding(text: str, model: str = None) -> List[float]:
    """
    获取文本的向量嵌入
    
    Args:
        text: 待嵌入的文本
        model: 嵌入模型名称，如果为 None 则使用配置文件中的值
    
    Returns:
        向量表示（浮点数列表）
    """
    # 从配置文件获取默认值
    if model is None:
        model = config.embedding_model
    
    try:
        resp = _execute_with_retry(
            "Embedding",
            lambda: client.embeddings.create(
                model=model,
                input=text,
            ),
        )
        return resp.data[0].embedding
    except Exception as e:
        logger.error(f"获取 embedding 失败: {e}")
        raise


def _continue_on_truncation(
    api_params: dict,
    accumulated_text: str,
    finish_reason: str,
    max_continuations: int,
) -> tuple[str, str]:
    """
    如果模型因为达到max_tokens而提前结束，自动请求接续
    
    使用对话前缀续写方式：将已生成的内容作为 assistant 消息，设置 prefix: True，
    让模型从该前缀继续生成，避免重复输出。
    
    Args:
        api_params: 原始 API 参数
        accumulated_text: 已累积的文本
        finish_reason: 首次响应的 finish_reason
        max_continuations: 最大接续次数
    
    Returns:
        (完整文本, 最终finish_reason)
    """
    if finish_reason != "length":
        return accumulated_text, finish_reason
    
    if max_continuations <= 0:
        logger.warning("输出被截断，但未开启接续功能")
        return accumulated_text, finish_reason
    
    base_messages = api_params.get("messages")
    if not base_messages:
        logger.warning("输出被截断，但缺少原始消息上下文，无法接续")
        return accumulated_text, finish_reason
    
    combined_text = accumulated_text
    attempt = 0
    total_attempts = max_continuations
    
    # 继续接续直到 finish_reason 不是 "length" 或达到最大接续次数
    while finish_reason == "length" and attempt < total_attempts:
        attempt += 1
        logger.warning(
            f"响应达到 max_tokens，自动发送第 {attempt}/{total_attempts} 次接续请求..."
        )
        
        # 记录接续前的内容（最后500字符）
        preview_length = 500
        before_text = (
            combined_text[-preview_length:]
            if len(combined_text) > preview_length
            else combined_text
        )
        logger.info(
            f"接续前内容（最后{len(before_text)}字符，总长度{len(combined_text)}字符）:"
        )
        logger.info("=" * 80)
        logger.info(before_text)
        logger.info("=" * 80)
        
        # 使用对话前缀续写：更新最后一个 assistant 消息，而不是追加新消息
        continuation_messages = copy.deepcopy(base_messages)
        # 如果最后一个消息是 assistant 消息，更新它；否则追加新的
        if continuation_messages and continuation_messages[-1].get("role") == "assistant":
            continuation_messages[-1] = {
                "role": "assistant",
                "content": combined_text,
                "prefix": True,
                "partial": True,
            }
        else:
            continuation_messages.append(
                {
                    "role": "assistant",
                    "content": combined_text,
                    "prefix": True,
                    "partial": True,
                }
            )
        
        continuation_params = {
            k: v for k, v in api_params.items() if k != "messages"
        }
        continuation_params["messages"] = continuation_messages
        
        start_time = time.time()
        try:
            response = _execute_with_retry(
                "ChatCompletion",
                lambda: client.chat.completions.create(**continuation_params),
            )
            elapsed = time.time() - start_time
            logger.info(f"接续请求 #{attempt} 完成，耗时: {elapsed:.2f}秒")
            
            if not response or not response.choices:
                logger.warning("接续请求返回无效响应，停止继续尝试")
                break
            
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                logger.info(
                    f"接续 Token使用 - prompt_tokens: {usage.prompt_tokens}, "
                    f"completion_tokens: {usage.completion_tokens}, "
                    f"total_tokens: {usage.total_tokens}"
                )
            
            extra_text = response.choices[0].message.content or ""
            if not extra_text:
                logger.warning("接续响应为空，停止继续尝试")
                break
            
            # 记录接续后的内容
            logger.info(f"接续后新增内容（长度: {len(extra_text)}字符）:")
            logger.info("=" * 80)
            logger.info(extra_text)
            logger.info("=" * 80)
            
            combined_text += extra_text
            finish_reason = response.choices[0].finish_reason
            logger.debug(f"接续后 finish_reason: {finish_reason}")
            
            # 记录合并后的内容（最后500字符，用于验证接续是否连贯）
            after_text = (
                combined_text[-preview_length:]
                if len(combined_text) > preview_length
                else combined_text
            )
            logger.info(
                f"接续后合并内容（最后{len(after_text)}字符，总长度{len(combined_text)}字符）:"
            )
            logger.info("=" * 80)
            logger.info(after_text)
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"接续请求失败: {e}")
            break
    
    if finish_reason == "length":
        logger.warning(
            f"在尝试 {total_attempts} 次接续后仍被截断，返回当前结果"
        )
    
    return combined_text, finish_reason


def parse_json_response(response: str) -> dict:
    """
    解析 LLM 返回的 JSON 字符串
    
    Args:
        response: LLM 返回的文本（可能包含 JSON）
    
    Returns:
        解析后的字典
    """
    import re
    
    original_response = response
    response = response.strip()
    
    # 策略1: 尝试查找 markdown 代码块中的 JSON（```json 或 ```）
    code_block_pattern = r'```(?:json)?\s*\n(.*?)\n```'
    match = re.search(code_block_pattern, response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass  # 继续尝试其他策略
    
    # 策略2: 如果整个响应以 ``` 开头和结尾，提取内容
    if response.startswith("```"):
        lines = response.split("\n")
        # 跳过第一行（```json 或 ```）
        start_idx = 1
        # 找到最后一个 ``` 的位置
        end_idx = len(lines) - 1
        if lines[end_idx].strip() == "```":
            json_str = "\n".join(lines[start_idx:end_idx]).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass  # 继续尝试其他策略
    
    # 策略3: 尝试直接解析整个响应
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass  # 继续尝试其他策略
    
    # 策略4: 尝试查找第一个 { 和最后一个 } 之间的内容
    first_brace = response.find('{')
    last_brace = response.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = response[first_brace:last_brace + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass  # 所有策略都失败
    
    # 所有策略都失败，记录错误并抛出异常
    logger.error(f"JSON 解析失败: 无法从响应中提取有效的 JSON")
    logger.error(f"响应长度: {len(original_response)} 字符")
    # 记录响应前后各 500 字符（如果响应较长）
    if len(original_response) > 1000:
        logger.error(f"响应内容（前500字符）: {original_response[:500]}")
        logger.error(f"响应内容（后500字符）: {original_response[-500:]}")
    else:
        logger.error(f"响应内容: {original_response}")
    raise json.JSONDecodeError("无法从响应中提取有效的 JSON", original_response, 0)
