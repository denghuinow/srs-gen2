#!/usr/bin/env python3
"""
测试向量维度脚本

用于测试当前配置的 embedding 模型返回的向量维度
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from openai_utils import get_embedding
from config import get_config

def test_vector_dimension():
    """测试向量维度"""
    config = get_config()
    
    print("=" * 60)
    print("向量维度测试")
    print("=" * 60)
    print(f"Embedding 模型: {config.embedding_model}")
    print()
    
    # 测试文本
    test_text = "这是一个测试文本，用于获取向量嵌入。"
    print(f"测试文本: {test_text}")
    print()
    
    try:
        # 获取向量
        print("正在调用 embedding API...")
        vector = get_embedding(test_text)
        
        # 输出结果
        print("=" * 60)
        print("测试结果")
        print("=" * 60)
        print(f"向量维度: {len(vector)}")
        print(f"向量类型: {type(vector)}")
        print(f"向量前5个值: {vector[:5]}")
        print(f"向量后5个值: {vector[-5:]}")
        print()
        
        # 验证向量值范围
        min_val = min(vector)
        max_val = max(vector)
        print(f"向量值范围: [{min_val:.6f}, {max_val:.6f}]")
        print()
        
        print("✓ 测试完成")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_vector_dimension()

