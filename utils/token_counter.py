"""Token 计算工具模块：使用 deepseek_tokenizer 计算输入 token 数量"""
import os
from pathlib import Path
from typing import List, Dict, Optional

# 尝试导入 transformers，如果失败则使用备用方法
try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def _get_tokenizer():
    """获取 tokenizer 实例（单例模式）"""
    if not TRANSFORMERS_AVAILABLE:
        if not hasattr(_get_tokenizer, "_warned_no_transformers"):
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("transformers 库未安装，无法计算 token 数量")
            _get_tokenizer._warned_no_transformers = True
        return None
    
    if not hasattr(_get_tokenizer, "_tokenizer"):
        # 获取 tokenizer 目录路径
        # 从当前文件位置向上查找 deepseek_v3_tokenizer 目录
        # utils/token_counter.py -> utils/ -> srs-gen2/ -> deepseek_v3_tokenizer
        current_file = Path(__file__)
        project_root = current_file.parent.parent  # utils/ -> srs-gen2/
        tokenizer_dir = project_root / "deepseek_v3_tokenizer"
        
        if not tokenizer_dir.exists():
            # 如果找不到，尝试从环境变量获取
            tokenizer_dir_env = os.getenv("DEEPSEEK_TOKENIZER_DIR")
            if tokenizer_dir_env:
                tokenizer_dir = Path(tokenizer_dir_env)
            else:
                if not hasattr(_get_tokenizer, "_warned_no_dir"):
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"找不到 tokenizer 目录: {tokenizer_dir}，无法计算 token 数量。"
                        f"请设置环境变量 DEEPSEEK_TOKENIZER_DIR 指定 tokenizer 目录路径。"
                    )
                    _get_tokenizer._warned_no_dir = True
                return None
        
        try:
            _get_tokenizer._tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(tokenizer_dir), trust_remote_code=True
            )
            # 设置一个很大的 model_max_length 来消除警告
            # tokenizer 的默认 model_max_length 是 16384，但实际模型可能支持更大的上下文
            if hasattr(_get_tokenizer._tokenizer, 'model_max_length'):
                _get_tokenizer._tokenizer.model_max_length = 1000000  # 设置为 100 万，足够大
        except Exception as e:
            if not hasattr(_get_tokenizer, "_warned_load_failed"):
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"加载 tokenizer 失败: {e}，无法计算 token 数量")
                _get_tokenizer._warned_load_failed = True
            return None
    
    return getattr(_get_tokenizer, "_tokenizer", None)


def count_text_tokens(text: str) -> Optional[int]:
    """计算单个文本字符串的 token 数量
    
    Args:
        text: 文本字符串
    
    Returns:
        token 数量，如果无法计算则返回 None
    """
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        return None
    
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception:
        return None


def count_tokens(messages: List[Dict[str, str]]) -> Optional[int]:
    """计算消息列表的 token 数量
    
    Args:
        messages: 消息列表，格式为 [{"role": "user", "content": "..."}, ...]
    
    Returns:
        token 数量，如果无法计算则返回 None
    """
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        return None
    
    try:
        # 将消息列表转换为 tokenizer 可以处理的格式
        # 对于 chat 模型，通常需要特殊处理
        total_tokens = 0
        
        # 尝试使用 apply_chat_template 方法（如果可用）
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                # 将消息转换为 tokenizer 期望的格式
                formatted_messages = []
                for msg in messages:
                    formatted_messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
                
                # 应用 chat template 并编码
                formatted_text = tokenizer.apply_chat_template(
                    formatted_messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                tokens = tokenizer.encode(formatted_text, add_special_tokens=True)
                return len(tokens)
            except Exception:
                pass
        
        # 备用方法：直接编码每个消息的内容
        for msg in messages:
            content = msg.get("content", "")
            if content:
                tokens = tokenizer.encode(content, add_special_tokens=False)
                total_tokens += len(tokens)
        
        # 添加特殊 token（如 role tokens）的估计
        # 通常每个消息会有几个额外的 token（role, format tokens 等）
        total_tokens += len(messages) * 4  # 粗略估计每个消息 4 个额外 token
        
        return total_tokens
    except Exception:
        return None


