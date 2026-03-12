# 项目日志等级约定

本文档约定异常与常规输出的日志等级，便于生产环境按级别过滤与排查。

## 1. 等级含义与使用场景

| 等级 | 用途 | 典型场景 |
|------|------|----------|
| **DEBUG** | 调试与预期内流水 | 单次 Redis 非连接类异常、答案超长跳过写缓存、关闭连接时异常、LLM 选点/熔断恢复等详细流水 |
| **INFO** | 启动与关键配置 | 线程池大小、PostgreSQL 长期记忆/监控启用、探活启动、应用生命周期相关 |
| **WARNING** | 可恢复的异常与降级 | Redis/DB 读写失败（会重连或降级）、探活失败、恢复会话失败、对话历史/问答监控读写失败、LLM 端点熔断或单次失败 |
| **ERROR** | 需关注的不可用或初始化失败 | Redis 首次连接失败导致缓存功能不可用、关键依赖初始化失败 |
| **EXCEPTION** | 需排查时的完整堆栈 | 仅在需保留 traceback 时使用（如首次连接失败），等价于 error + traceback |

## 2. 异常输出原则

- **可恢复**：重试/重连/降级后业务仍可继续 → **WARNING**（如 Redis 断线后置空客户端、下次重连；DB 读写失败后本次请求降级）。
- **不可恢复或影响功能**：本次进程内功能不可用且不会自动恢复 → **ERROR**（如 Redis 首次建连失败且未配置降级）。
- **预期内或单次噪音**：如单次 get/set 非连接类异常、答案超长不写 → **DEBUG**。
- **关闭/清理阶段**的异常（如 lifespan 关闭 Redis）→ **DEBUG**，避免正常关闭时刷 WARNING。

## 3. 各模块约定

- **answer_cache**：首次连接失败 → ERROR（+ exception 带 traceback）；读写时连接类失败 → WARNING；非连接类单次异常、关闭连接异常、答案超长跳过 → DEBUG。
- **api/main**：PostgreSQL 探活失败/任务异常（会持续重试）、恢复人工介入会话失败 → WARNING；启动阶段配置与启用 → INFO。
- **chat_history / qa_monitoring**：各类 DB 读写或建表失败（业务降级）→ WARNING。
- **llm**：端点熔断、非可重试失败、可重试失败计数 → WARNING；选点/成功流水 → DEBUG（默认 INFO 下不刷屏）。

## 4. 配置建议

- 开发：`logging.basicConfig(level=logging.DEBUG)` 或通过环境变量 `LOG_LEVEL=DEBUG`。
- 生产：默认 `INFO`，需排查时临时改为 `DEBUG`；仅看严重问题可设为 `WARNING`。
