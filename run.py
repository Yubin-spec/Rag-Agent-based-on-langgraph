#!/usr/bin/env python3
"""
从项目根目录启动知识库智能体 API（默认 0.0.0.0:8000，reload 开发模式）。
对话与知识库仅使用 DeepSeek、BGE，不调用 OpenAI。
"""
import sys
from pathlib import Path

# 保证以项目根为 path，可正确导入 api.main、config、src
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
