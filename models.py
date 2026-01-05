# models.py
from typing import Optional, List


class SemanticUnit:
    """语义单元，表示一个独立的需求描述"""
    
    def __init__(
        self,
        text: str,
        grade: Optional[float] = None,
        vector: Optional[List[float]] = None,
        token_count: Optional[int] = None,
    ):
        self.text = text
        self.grade = grade  # 评分：+2, +1, 0, -1, -2
        self.vector = vector  # 用于缓存 OpenAI embedding
        self.token_count = token_count  # Token 数量

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "text": self.text,
            "grade": self.grade,
            "vector": self.vector,
            "token_count": self.token_count,
        }

    @staticmethod
    def from_dict(d: dict) -> "SemanticUnit":
        """从字典反序列化"""
        return SemanticUnit(
            text=d["text"],
            grade=d.get("grade"),
            vector=d.get("vector"),
            token_count=d.get("token_count"),
        )
    
    def calculate_token_count(self) -> Optional[int]:
        """计算并更新 token 数量
        
        Returns:
            token 数量，如果无法计算则使用估算值
        """
        try:
            from utils.token_counter import count_text_tokens
            self.token_count = count_text_tokens(self.text)
            if self.token_count is not None:
                return self.token_count
        except Exception:
            pass
        
        # 回退方案：使用估算方法（每个字符约 0.25 token，中文约 1.5 token/字符）
        # 简单估算：英文和数字按 4 字符 = 1 token，中文按 1.5 字符 = 1 token
        import re
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', self.text))
        other_chars = len(self.text) - chinese_chars
        estimated_tokens = int(chinese_chars / 1.5 + other_chars / 4)
        # 至少为 1
        self.token_count = max(1, estimated_tokens)
        return self.token_count
    
    def __eq__(self, other):
        """用于去重：基于 text 判断相等"""
        if not isinstance(other, SemanticUnit):
            return False
        return self.text == other.text
    
    def __hash__(self):
        """用于去重：基于 text 计算哈希"""
        return hash(self.text)
    
    def __repr__(self):
        return f"SemanticUnit(text='{self.text[:50]}...', grade={self.grade})"

