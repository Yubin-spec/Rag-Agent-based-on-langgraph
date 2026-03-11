"""
项目配置：从 .env 与环境变量加载。
大模型仅允许 DeepSeek，向量/重排仅允许 BGE-M3、BGE Reranker Large，禁止 OpenAI。
"""
from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
