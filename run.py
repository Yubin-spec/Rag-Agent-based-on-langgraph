#!/usr/bin/env python3
"""
从项目根目录启动知识库智能体 API（默认 0.0.0.0:8000）。

- 开发：默认 reload=True、单 worker（reload 与多 worker 互斥）。
- 生产：设置 RELOAD=0 或 UVICORN_WORKERS>1 时关闭 reload，worker 数由 UVICORN_WORKERS 控制（默认 1）。
对话与知识库仅使用 DeepSeek、BGE，不调用 OpenAI。
"""
import os
import sys
from pathlib import Path

# 保证以项目根为 path，可正确导入 api.main、config、src
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    import uvicorn
    use_reload = os.environ.get("RELOAD", "1").strip().lower() in ("1", "true", "yes")
    if use_reload:
        # 开发模式：reload 仅支持单 worker
        uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
    else:
        from config import get_settings
        s = get_settings()
        workers = max(1, s.uvicorn_workers)
        shutdown = max(0, getattr(s, "graceful_shutdown_timeout_seconds", 30))
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            workers=workers,
            reload=False,
            timeout_graceful_shutdown=shutdown,
        )
