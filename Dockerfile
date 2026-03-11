# 知识库智能体 API - 内网 Docker 部署
# 仅使用 DeepSeek、BGE，不调用 OpenAI
FROM python:3.11-slim

WORKDIR /app

# 系统依赖（可选，部分 embedding 依赖）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 依赖先安装，便于利用层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY config ./config
COPY api ./api
COPY src ./src
COPY run.py .

# 数据目录（挂载卷后持久化）
RUN mkdir -p /app/data/uploads /app/data

# 不依赖 .env 文件时也可通过环境变量运行
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

# 生产环境不用 reload
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
