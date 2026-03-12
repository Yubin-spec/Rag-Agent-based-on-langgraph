# Redis 问答缓存说明与优化

问答缓存将「归一化后的问题 → 答案」存入 Redis，相同或高度相似问题可直接命中缓存，降低 LLM 与检索压力。

## 1. 行为概览

- **Key**：`kb:answer:v1:` + 问题归一化后的 SHA256（前 32 位）。归一化：去首尾空白、连续空白折叠、小写、截断 500 字。
- **Value**：答案纯文本（UTF-8）。
- **TTL**：由 `answer_cache_ttl_seconds` 控制，默认 86400（24 小时）。
- **开关**：`answer_cache_enabled=False` 或 `redis_url` 为空时，不连 Redis，读/写均静默跳过。

## 2. 连接与高并发

- **连接池**：使用 `redis.asyncio`，`Redis.from_url()` 内部使用连接池；`redis_max_connections` 控制池大小（默认 10），高并发时可适当调大。
- **超时**：`redis_socket_connect_timeout`（建连，默认 2 秒）、`redis_socket_timeout`（读写，默认 5 秒），避免单次请求长时间占用连接。
- **健康检查**：`redis_health_check_interval`（秒，默认 30）对池内连接做定期健康检查，0 表示不检查。可减少因长时间空闲被服务端关闭导致的半死连接。

## 3. 断线重连

- 读/写时若发生连接类或超时类异常，会置空全局客户端并打日志；下次 `get_cached_answer` / `set_cached_answer` 时重新懒加载建连，无需重启进程。
- 首次建连使用 asyncio.Lock，保证并发下只建一个池。

## 4. 单条价值长度限制

- `answer_cache_max_value_bytes`：单条答案最大缓存字节数（UTF-8），默认 0 表示不限制。
- 若 > 0 且答案编码后超过该长度，则**不写入**缓存，避免超大 value 占满内存或拖慢 Redis；读逻辑不变，仅写时做长度校验。

## 5. 热 key 与防击穿

### 5.1 热 key（进程内本地缓存）

- **问题**：同一问题被高并发重复请求时，会集中打同一个 Redis key，造成热 key 压力甚至拖垮实例。
- **做法**：可选**进程内 LRU 缓存**，在查 Redis 前先查本地；命中则直接返回、不打 Redis；未命中再查 Redis，并从 Redis 回填本地（写时同时写 Redis + 本地）。
- **配置**：`answer_cache_local_max_entries`（条数，0 表示关闭）、`answer_cache_local_ttl_seconds`（本地 TTL，默认 60）。多 worker 时每个进程独立一份本地缓存，不共享。
- **效果**：同一进程内重复相同问题可显著减少 Redis 请求，减轻热 key；TTL 较短避免长期脏数据。

### 5.2 防击穿（单飞锁）

- **问题**：缓存过期或未命中时，大量请求同时回源（走图/LLM），造成「击穿」—— 瞬时压力与重复计算。
- **做法**：对同一问题（同一缓存 key）加**单飞锁**：仅持锁的协程回源计算并写缓存，其余协程在锁外等待；持锁协程释放后，等待方再次 `get_cached_answer` 即可命中。
- **实现**：按 key 分桶锁（`answer_cache_single_flight_buckets`，默认 256），不同 key 可能同桶（轻微竞争），同一 key 必同桶，保证同一问题只有一个回源。
- **调用**：`POST /chat`、`POST /chat/stream` 在缓存未命中后使用 `async with answer_lock(question):` 包裹「再次查缓存 + 回源 + set_cached_answer」，实现防击穿。

## 6. 生命周期

- 应用关闭时（FastAPI lifespan 退出）会调用 `close_redis_connection()`，关闭连接池，避免连接泄漏与告警。

## 7. 配置项汇总（config/settings.py / 环境变量）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| redis_url | redis://localhost:6379/0 | 连接串，为空则禁用缓存 |
| answer_cache_enabled | True | 是否启用问答缓存 |
| answer_cache_ttl_seconds | 86400 | 缓存 TTL（秒） |
| answer_cache_max_value_bytes | 0 | 单条答案最大字节数，0 不限制 |
| answer_cache_local_max_entries | 0 | 热 key 进程内 LRU 条数，0 关闭 |
| answer_cache_local_ttl_seconds | 60 | 本地缓存 TTL（秒） |
| answer_cache_single_flight_buckets | 256 | 防击穿单飞锁分桶数 |
| redis_max_connections | 10 | 连接池最大连接数 |
| redis_socket_connect_timeout | 2.0 | 建连超时（秒） |
| redis_socket_timeout | 5.0 | 读写超时（秒） |
| redis_health_check_interval | 30 | 健康检查间隔（秒），0 不检查 |

## 8. 调用位置

- **读**：`POST /chat`、`POST /chat/stream` 在处理用户消息后、走图/知识库前，会先 `get_cached_answer(question)`，命中则直接返回缓存内容（先查本地热 key，再查 Redis）。
- **写**：在返回用户答案后（非流式/流式均会）调用 `set_cached_answer(question, answer)` 写入缓存（若启用本地缓存则同时写 Redis 与本地）。
- **防击穿**：缓存未命中后，使用 `answer_lock(question)` 包裹「再次查缓存 + 回源 + 写缓存」，保证同一问题仅一个协程回源。

## 9. 可选后续优化

- **按租户/用户隔离**：若多租户需隔离缓存，可在 key 中加入 `user_id` 或 `tenant_id` 前缀。
- **缓存统计**：可用 Redis INFO 或自定义指标统计命中率、key 数量，便于调 TTL 与容量。
- **集群/哨兵**：生产若使用 Redis Cluster 或 Sentinel，需使用对应 `redis_url` 与客户端配置（如 `redis-py` 对 cluster 的支持）。
