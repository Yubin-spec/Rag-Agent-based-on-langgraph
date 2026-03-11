# Docker 内网部署说明

## 一、前置条件

- 内网环境可访问 DeepSeek API（或内网代理），或使用内网部署的兼容 OpenAI 的模型服务。
- 可选：内网 Redis（若需问答缓存）；内网 Milvus（若需 RAG 向量检索）。

## 二、快速启动（推荐）

### 1. 准备配置

```bash
# 在项目根目录
cp .env.example .env
# 编辑 .env，至少填写：
#   OPENAI_API_BASE   DeepSeek 或内网模型 API 地址
#   OPENAI_API_KEY    API Key
# 使用本 compose 自带的 Redis 时：REDIS_URL=redis://redis:6379/0
# 若内网有独立 Redis：REDIS_URL=redis://内网Redis主机:6379/0
# 若内网有独立 Milvus：MILVUS_URI=http://内网Milvus:19530
```

### 2. 构建并启动

```bash
# 构建镜像并启动 app + Redis
docker compose up -d --build

# 仅构建镜像
docker compose build
```

### 3. 验证

- 接口文档：http://内网IP:8000/docs  
- 健康检查：http://内网IP:8000/health  

## 三、仅部署 API（不启动 Redis）

若不使用问答缓存，可关闭缓存或只启动 app：

```bash
# 方式 A：.env 中设置
ANSWER_CACHE_ENABLED=false

# 方式 B：只启动 app（需先注释 docker-compose.yml 中 app 的 depends_on）
docker compose up -d app
```

或使用单独运行容器（不依赖 compose 中的 redis）：

```bash
docker build -t kb-agent:latest .
docker run -d --name kb-agent -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e OPENAI_API_BASE=https://api.deepseek.com/v1 \
  -e OPENAI_API_KEY=your_key \
  -e ANSWER_CACHE_ENABLED=false \
  kb-agent:latest
```

## 四、数据持久化

- `./data` 挂载到容器内 `/app/data`，包含：
  - `data/kb.db`：Text2SQL 使用的 SQLite
  - `data/uploads`：上传文件
  - `data/text2sql_schema_overrides.json`：表结构人工审核
  - `data/high_freq_qa.json`：高频 QA
- Redis 数据使用 compose 中 `redis_data` 卷，删除容器不会丢数据。

## 五、内网 Milvus / Redis 已存在时

在 `.env` 中指向内网服务即可，无需启动 compose 中的 redis：

```env
# 内网 Milvus
MILVUS_URI=http://内网Milvus主机:19530

# 内网 Redis（不启动 compose 的 redis 时）
REDIS_URL=redis://内网Redis主机:6379/0
```

在 `.env` 中设置 `REDIS_URL=redis://内网Redis主机:6379/0` 即可，无需启动 compose 中的 redis 服务。

## 六、常用命令

```bash
# 查看日志
docker compose logs -f app

# 重启
docker compose restart app

# 停止并删除容器（保留 data 与 redis 卷）
docker compose down
```
