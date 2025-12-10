# models.py
from typing import Optional, List


class SemanticUnit:
    """语义单元，表示一个独立的需求描述"""
    
    def __init__(
        self,
        text: str,
        grade: Optional[float] = None,
        vector: Optional[List[float]] = None,
    ):
        self.text = text
        self.grade = grade  # 评分：+2, +1, 0, -1, -2
        self.vector = vector  # 用于缓存 OpenAI embedding

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {"text": self.text, "grade": self.grade, "vector": self.vector}

    @staticmethod
    def from_dict(d: dict) -> "SemanticUnit":
        """从字典反序列化"""
        return SemanticUnit(
            text=d["text"],
            grade=d.get("grade"),
            vector=d.get("vector"),
        )
    
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

