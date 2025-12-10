# config.py
"""配置管理模块"""
import os
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class Config:
    """系统配置类"""
    
    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self) -> None:
        """加载配置文件"""
        # 查找配置文件路径
        config_path = self._find_config_file()
        
        if config_path and config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self._config = yaml.safe_load(f) or {}
                logger.info(f"成功加载配置文件: {config_path}")
            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}，使用默认配置")
                self._config = {}
        else:
            logger.warning("未找到配置文件，使用默认配置")
            self._config = {}
    
    def _find_config_file(self) -> Optional[Path]:
        """查找配置文件路径"""
        # 优先使用环境变量指定的路径
        config_env = os.getenv("SRS_GEN2_CONFIG")
        if config_env:
            path = Path(config_env)
            if path.exists():
                return path
        
        # 查找项目根目录下的 config.yaml
        current_file = Path(__file__)
        project_root = current_file.parent
        config_path = project_root / "config.yaml"
        
        if config_path.exists():
            return config_path
        
        return None
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值（支持点号分隔的路径）
        
        Args:
            key_path: 配置键路径，例如 "openai.chat.model"
            default: 默认值
        
        Returns:
            配置值
        """
        keys = key_path.split(".")
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    @property
    def openai_api_key(self) -> str:
        """获取 OpenAI API Key"""
        # 优先使用环境变量
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            return env_key
        
        # 从配置文件读取
        return self.get("openai.api_key", "")
    
    @property
    def openai_base_url(self) -> Optional[str]:
        """获取 OpenAI Base URL"""
        # 优先使用环境变量
        env_url = os.getenv("OPENAI_BASE_URL")
        if env_url:
            return env_url
        
        # 从配置文件读取
        url = self.get("openai.base_url")
        return url if url else None
    
    @property
    def chat_model(self) -> str:
        """获取 Chat 模型名称"""
        return self.get("openai.chat.model", "gpt-4o-mini")
    
    @property
    def chat_temperature(self) -> float:
        """获取 Chat 温度参数"""
        return self.get("openai.chat.temperature", 0.2)
    
    @property
    def chat_max_tokens(self) -> Optional[int]:
        """获取 Chat 最大输出 token 数"""
        return self.get("openai.chat.max_tokens")
    
    @property
    def chat_max_continuations(self) -> int:
        """获取 Chat 最大接续次数"""
        return self.get("openai.chat.max_continuations", 0)
    
    @property
    def embedding_model(self) -> str:
        """获取 Embedding 模型名称"""
        return self.get("openai.embedding.model", "text-embedding-3-small")
    
    @property
    def rho(self) -> float:
        """获取每轮探索新需求的比例"""
        return self.get("iteration.rho", 0.5)
    
    @property
    def max_outer_iter(self) -> int:
        """获取最大外循环次数"""
        return self.get("iteration.max_outer_iter", 5)
    
    @property
    def max_inner_iter(self) -> int:
        """获取最大内循环次数"""
        return self.get("iteration.max_inner_iter", 3)
    
    @property
    def similarity_threshold(self) -> float:
        """获取相似度阈值"""
        return self.get("similarity.threshold", 0.8)
    
    @property
    def log_level(self) -> str:
        """获取日志级别"""
        return self.get("logging.level", "INFO")
    
    @property
    def log_format(self) -> str:
        """获取日志格式"""
        return self.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    @property
    def api_retry_max_attempts(self) -> int:
        """获取 OpenAI API 请求的最大重试次数"""
        return int(self.get("openai.retry.max_attempts", 3))

    @property
    def api_retry_base_delay(self) -> float:
        """获取 OpenAI API 重试的基础等待时间（秒）"""
        return float(self.get("openai.retry.base_delay", 2.0))

    @property
    def api_retry_max_delay(self) -> float:
        """获取 OpenAI API 重试的最大等待时间（秒）"""
        return float(self.get("openai.retry.max_delay", 30.0))
    
    @property
    def prompt_language(self) -> str:
        """获取提示词语言（zh 或 en）"""
        return self.get("prompt.language", "en")
    
    def get_component_model(self, component_name: str) -> str:
        """
        获取指定组件的模型配置
        
        Args:
            component_name: 组件名称（split_to_semantic_units, requirement_explorer, requirement_clarifier, srs_generator）
        
        Returns:
            模型名称，如果未配置则返回默认的 chat.model
        """
        model = self.get(f"openai.components.{component_name}.model")
        if model is None:
            return self.chat_model
        return model
    
    def get_component_temperature(self, component_name: str) -> float:
        """
        获取指定组件的温度配置
        
        Args:
            component_name: 组件名称（split_to_semantic_units, requirement_explorer, requirement_clarifier, srs_generator）
        
        Returns:
            温度值，如果未配置则返回默认的 chat.temperature
        """
        temperature = self.get(f"openai.components.{component_name}.temperature")
        if temperature is None:
            return self.chat_temperature
        return temperature
    
    def get_component_max_tokens(self, component_name: str) -> Optional[int]:
        """
        获取指定组件的最大输出 token 数配置
        
        Args:
            component_name: 组件名称（split_to_semantic_units, requirement_explorer, requirement_clarifier, srs_generator）
        
        Returns:
            最大 token 数，如果未配置则返回默认的 chat.max_tokens
        """
        max_tokens = self.get(f"openai.components.{component_name}.max_tokens")
        if max_tokens is None:
            return self.chat_max_tokens
        return max_tokens
    
    def get_component_max_continuations(self, component_name: str) -> int:
        """
        获取指定组件的最大接续次数配置
        
        Args:
            component_name: 组件名称（split_to_semantic_units, requirement_explorer, requirement_clarifier, srs_generator）
        
        Returns:
            最大接续次数，如果未配置则返回默认的 chat.max_continuations
        """
        max_continuations = self.get(f"openai.components.{component_name}.max_continuations")
        if max_continuations is None:
            return self.chat_max_continuations
        return max_continuations
    
    def validate(self) -> None:
        """验证配置"""
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API Key 未设置。请通过以下方式之一设置：\n"
                "1. 环境变量 OPENAI_API_KEY\n"
                "2. 配置文件 config.yaml 中的 openai.api_key"
            )


# 全局配置实例
def get_config() -> Config:
    """获取配置实例"""
    return Config()
