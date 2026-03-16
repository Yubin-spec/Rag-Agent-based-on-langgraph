# 知识库智能体 API - 内网 Docker 部署
# 仅使用 DeepSeek、BGE，不调用 OpenAI
# 默认使用国内镜像（Docker Hub 不可达时）。若仍 EOF，可换：docker.xuanyuan.me/library/python:3.11-slim 或配置 Docker Desktop 镜像加速后改用 BASE_IMAGE=python:3.11-slim
ARG BASE_IMAGE=docker.m.daocloud.io/library/python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /app

# 可选：使用国内镜像避免 apt 源超时（境外构建可注释掉下一行或 build-arg APT_USE_MIRROR=0）
ARG APT_USE_MIRROR=1
RUN if [ "$APT_USE_MIRROR" = "1" ]; then \
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources; do \
      [ -f "$f" ] && sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|http://security.debian.org/debian-security|http://mirrors.aliyun.com/debian-security|g' "$f" || true; \
    done; \
    fi

# 系统依赖（部分 embedding 等 pip 包需编译）；zlib1g-dev 为 zlib-state 等依赖所需
RUN apt-get update || apt-get update \
    && apt-get install -y --no-install-recommends build-essential zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# 依赖先安装，便于利用层缓存
# 可选：使用国内 PyPI 镜像（境外构建可 build-arg PIP_USE_MIRROR=0）
# 若仍失败，可用 docker build --progress=plain 查看具体报错包
ARG PIP_USE_MIRROR=1
COPY requirements.txt .
RUN if [ "$PIP_USE_MIRROR" = "1" ]; then \
    pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --default-timeout=120; \
    pip install --no-cache-dir -r requirements.txt \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --trusted-host mirrors.aliyun.com \
        --extra-index-url https://pypi.org/simple/ \
        --default-timeout=120 --retries 5; \
    else \
    pip install --upgrade pip --default-timeout=120; \
    pip install --no-cache-dir -r requirements.txt --default-timeout=120 --retries 5; \
    fi

# 应用代码
COPY config ./config
COPY api ./api
COPY src ./src
COPY run.py .
# 前端页面：根路径 / 返回 index.html，否则返回 API 说明 JSON
COPY frontend ./frontend

# 数据目录（挂载卷后持久化；schema overrides 等可挂载或运行时生成）
RUN mkdir -p /app/data/uploads /app/data

# 非 root 运行（可选：若挂载卷权限有问题可改为 USER root 或去掉下面两行）
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
# 生产模式：关闭 reload；worker 数默认 1，可通过 UVICORN_WORKERS 覆盖
ENV RELOAD=0
ENV UVICORN_WORKERS=1

EXPOSE 8000

# 健康检查（按需调整间隔）
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# 使用 run.py 以读取 config（graceful_shutdown_timeout_seconds 等）；worker 数由 UVICORN_WORKERS 控制
CMD ["python", "run.py"]
