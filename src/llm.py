# src/llm.py
"""
大模型调用统一入口。

本项目仅使用 DeepSeek（如 DeepSeek R1），禁止调用 OpenAI 自有模型（如 GPT-4）。
在单一 API 基础上补充了多 Key/多 endpoint 负载均衡、失败熔断与自动切换：
- 默认仍兼容原有 `OPENAI_API_BASE` + `OPENAI_API_KEY` 配置；
- 若配置 `DEEPSEEK_API_ENDPOINTS`，则按“权重 + 当前并发负载”优先选择健康节点；
- 连续失败达到阈值后暂时熔断，避免把流量持续打到异常节点。
"""

from __future__ import annotations

import logging
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Optional

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from pydantic import ConfigDict, PrivateAttr

from config import get_settings

logger = logging.getLogger(__name__)
_LAST_ENDPOINT_NAME: ContextVar[str] = ContextVar("deepseek_last_endpoint_name", default="")


def _is_retryable_error(exc: Exception) -> bool:
    """识别适合切换到其它节点重试的错误：超时、连接错误、429 与 5xx。"""

    status_code = str(getattr(exc, "status_code", "") or "")
    msg = (str(exc) or "").lower()
    retryable_keywords = ("timeout", "timed out", "connection", "temporarily", "rate limit", "429")
    if any(keyword in msg for keyword in retryable_keywords):
        return True
    return status_code.startswith("5") or status_code == "429"


@dataclass
class _EndpointConfig:
    """单个 DeepSeek 接入点配置。"""

    base_url: str
    api_key: str
    weight: int = 1
    name: str = ""


@dataclass
class _EndpointState:
    """运行期节点状态：用于负载均衡与熔断。"""

    config: _EndpointConfig
    failures: int = 0
    opened_until: float = 0.0
    inflight: int = 0
    clients: dict[tuple[str, float], Any] = field(default_factory=dict)


def _parse_deepseek_endpoints() -> list[_EndpointConfig]:
    """
    解析 DeepSeek endpoint 配置。

    支持两种方式：
    1. 兼容旧配置：`OPENAI_API_BASE` + `OPENAI_API_KEY`
    2. 多节点配置：`DEEPSEEK_API_ENDPOINTS`
       格式：`base_url|api_key|weight|name`，多项用逗号或换行分隔
    """

    s = get_settings()
    raw = (getattr(s, "deepseek_api_endpoints", "") or "").strip()
    endpoints: list[_EndpointConfig] = []
    if raw:
        entries = [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]
        for idx, item in enumerate(entries, 1):
            parts = [part.strip() for part in item.split("|")]
            if len(parts) < 2 or not parts[0] or not parts[1]:
                continue
            base_url = parts[0]
            api_key = parts[1]
            weight_text = parts[2] if len(parts) >= 3 and parts[2] else "1"
            name = parts[3] if len(parts) >= 4 and parts[3] else f"deepseek-{idx}"
            try:
                weight = max(1, int(weight_text))
            except Exception:
                weight = 1
            endpoints.append(_EndpointConfig(base_url=base_url, api_key=api_key, weight=weight, name=name))
    if endpoints:
        return endpoints
    return [
        _EndpointConfig(
            base_url=s.openai_api_base,
            api_key=s.openai_api_key,
            weight=1,
            name="deepseek-default",
        )
    ]


class _DeepSeekEndpointPool:
    """进程内共享的 DeepSeek 节点池。"""

    def __init__(self):
        s = get_settings()
        self._states = [_EndpointState(config=item) for item in _parse_deepseek_endpoints()]
        self._failure_threshold = max(1, int(getattr(s, "deepseek_circuit_breaker_failures", 3)))
        self._open_seconds = max(5, int(getattr(s, "deepseek_circuit_breaker_open_seconds", 30)))
        self._timeout_seconds = max(1, int(getattr(s, "agent_request_timeout_seconds", 120)))
        self._lock = threading.Lock()
        self._round_robin_cursor = 0

    def _healthy_states_locked(self) -> list[_EndpointState]:
        now = time.time()
        healthy = [state for state in self._states if state.opened_until <= now]
        if healthy:
            return healthy
        if not self._states:
            return []
        # 所有节点都熔断时，放行“最早到期”的 1 个节点进入半开状态做探测。
        return [min(self._states, key=lambda state: state.opened_until)]

    @staticmethod
    def _weighted_load_score(state: _EndpointState) -> float:
        """
        计算节点当前负载分数，分数越小越应优先被选择。

        这里用 `inflight / weight` 近似表达：
        - inflight 越低，说明当前并发越小；
        - weight 越高，说明该节点理论承载能力越强；
        因此高权重节点在相同并发下会得到更低分，更容易继续接流量。
        """

        return state.inflight / max(1, state.config.weight)

    def ordered_candidates(self) -> list[_EndpointState]:
        """
        返回本次调用的候选节点顺序：优先选择「加权后当前负载更低」的健康节点。

        排序规则：
        1. `inflight / weight` 越小越优先，兼顾并发与权重；
        2. 若分数相同，则按轮询游标打散，避免长期固定命中同一节点；
        3. 若仍相同，则失败次数更少的优先。

        注意：同一次调用里每个节点最多尝试一次。
        """

        with self._lock:
            healthy = self._healthy_states_locked()
            if not healthy:
                return []

            start = self._round_robin_cursor % len(healthy)
            self._round_robin_cursor += 1
            rotated = healthy[start:] + healthy[:start]
            rotate_rank = {id(state): idx for idx, state in enumerate(rotated)}

            return sorted(
                healthy,
                key=lambda state: (
                    self._weighted_load_score(state),
                    rotate_rank[id(state)],
                    state.failures,
                ),
            )

    def status_snapshot(self) -> list[dict[str, Any]]:
        """导出当前 endpoint 状态快照，供日志与外部观测接口使用。"""

        with self._lock:
            now = time.time()
            snapshot: list[dict[str, Any]] = []
            for state in self._states:
                seconds_until_retry = max(0.0, state.opened_until - now)
                snapshot.append({
                    "name": state.config.name or "",
                    "base_url": state.config.base_url,
                    "weight": state.config.weight,
                    "inflight": state.inflight,
                    "failures": state.failures,
                    "circuit_open": state.opened_until > now,
                    "seconds_until_retry": round(seconds_until_retry, 3),
                    "model_client_count": len(state.clients),
                })
            return snapshot

    def acquire(self, state: _EndpointState) -> None:
        """标记节点正在处理请求。"""

        with self._lock:
            state.inflight += 1
            current_inflight = state.inflight
            weight = state.config.weight
            failures = state.failures
        logger.info(
            "LLM endpoint selected: name=%s inflight=%s weight=%s weighted_load=%.3f failures=%s",
            state.config.name or "unknown",
            current_inflight,
            weight,
            current_inflight / max(1, weight),
            failures,
        )

    def release_success(self, state: _EndpointState) -> None:
        """请求成功：关闭熔断并清空连续失败计数。"""

        with self._lock:
            was_open = state.opened_until > time.time()
            state.inflight = max(0, state.inflight - 1)
            state.failures = 0
            state.opened_until = 0.0
            current_inflight = state.inflight
        logger.info(
            "LLM endpoint success: name=%s inflight=%s circuit_recovered=%s",
            state.config.name or "unknown",
            current_inflight,
            was_open,
        )

    def release_failure(self, state: _EndpointState, exc: Exception) -> None:
        """请求失败：若为可切换错误则累计失败并触发熔断。"""

        with self._lock:
            state.inflight = max(0, state.inflight - 1)
            if not _is_retryable_error(exc):
                current_inflight = state.inflight
                error_text = str(exc)
                endpoint_name = state.config.name or "unknown"
                logger.warning(
                    "LLM endpoint non-retryable failure: name=%s inflight=%s error=%s",
                    endpoint_name,
                    current_inflight,
                    error_text,
                )
                return
            state.failures += 1
            current_inflight = state.inflight
            failures = state.failures
            threshold = self._failure_threshold
            error_text = str(exc)
            if state.failures >= self._failure_threshold:
                state.opened_until = time.time() + self._open_seconds
                seconds_until_retry = round(max(0.0, state.opened_until - time.time()), 3)
                endpoint_name = state.config.name or "unknown"
                logger.warning(
                    "LLM endpoint circuit opened: name=%s inflight=%s failures=%s threshold=%s retry_after_seconds=%s error=%s",
                    endpoint_name,
                    current_inflight,
                    failures,
                    threshold,
                    seconds_until_retry,
                    error_text,
                )
                return
        logger.warning(
            "LLM endpoint retryable failure: name=%s inflight=%s failures=%s threshold=%s error=%s",
            state.config.name or "unknown",
            current_inflight,
            failures,
            threshold,
            error_text,
        )

    def get_client(self, state: _EndpointState, model_name: str, temperature: float):
        """按 endpoint + 模型参数复用 ChatOpenAI 客户端。"""

        from langchain_openai import ChatOpenAI

        key = (model_name, float(temperature))
        with self._lock:
            client = state.clients.get(key)
            if client is None:
                client = ChatOpenAI(
                    model=model_name,
                    base_url=state.config.base_url,
                    api_key=state.config.api_key,
                    temperature=temperature,
                    timeout=self._timeout_seconds,
                    max_retries=0,
                )
                state.clients[key] = client
            return client


_POOL: Optional[_DeepSeekEndpointPool] = None
_POOL_LOCK = threading.Lock()


def _get_pool() -> _DeepSeekEndpointPool:
    """获取进程内共享节点池。"""

    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = _DeepSeekEndpointPool()
        return _POOL


def get_last_deepseek_endpoint_name() -> str:
    """获取当前上下文最近一次命中的 DeepSeek endpoint 名称。"""

    return (_LAST_ENDPOINT_NAME.get("") or "").strip()


def get_deepseek_router_status() -> dict[str, Any]:
    """返回当前 DeepSeek 路由器状态，供 API/诊断使用。"""

    pool = _get_pool()
    snapshot = pool.status_snapshot()
    now = time.time()
    return {
        "router": "deepseek-router",
        "endpoint_count": len(snapshot),
        "healthy_endpoint_count": sum(1 for item in snapshot if not item.get("circuit_open")),
        "all_circuits_open": bool(snapshot) and all(item.get("circuit_open") for item in snapshot),
        "timestamp": now,
        "endpoints": snapshot,
    }


class _DeepSeekChatRouter(BaseChatModel):
    """把 LangChain ChatModel 请求路由到健康的 DeepSeek endpoint。"""

    model_name: str = "deepseek-chat"
    temperature: float = 0.0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _pool: _DeepSeekEndpointPool = PrivateAttr()

    def __init__(self, **data: Any):
        pool = data.pop("pool")
        super().__init__(**data)
        self._pool = pool

    @property
    def _llm_type(self) -> str:
        return "deepseek-router"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature}

    def _generate(
        self,
        messages: list,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_error: Optional[Exception] = None
        for state in self._pool.ordered_candidates():
            client = self._pool.get_client(state, self.model_name, self.temperature)
            self._pool.acquire(state)
            try:
                _LAST_ENDPOINT_NAME.set(state.config.name or "")
                result = client._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                self._pool.release_success(state)
                return result
            except Exception as exc:
                self._pool.release_failure(state, exc)
                last_error = exc
                if not _is_retryable_error(exc):
                    raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("未配置可用的 DeepSeek endpoint。")

    async def _agenerate(
        self,
        messages: list,
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_error: Optional[Exception] = None
        for state in self._pool.ordered_candidates():
            client = self._pool.get_client(state, self.model_name, self.temperature)
            self._pool.acquire(state)
            try:
                _LAST_ENDPOINT_NAME.set(state.config.name or "")
                result = await client._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                self._pool.release_success(state)
                return result
            except Exception as exc:
                self._pool.release_failure(state, exc)
                last_error = exc
                if not _is_retryable_error(exc):
                    raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("未配置可用的 DeepSeek endpoint。")

    def _stream(
        self,
        messages: list,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        last_error: Optional[Exception] = None
        for state in self._pool.ordered_candidates():
            client = self._pool.get_client(state, self.model_name, self.temperature)
            self._pool.acquire(state)
            yielded = False
            try:
                _LAST_ENDPOINT_NAME.set(state.config.name or "")
                for chunk in client._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
                    yielded = True
                    yield chunk
                self._pool.release_success(state)
                return
            except Exception as exc:
                self._pool.release_failure(state, exc)
                last_error = exc
                # 流式场景一旦已经开始输出，再切节点会打断上下文，直接抛出。
                if yielded or not _is_retryable_error(exc):
                    raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("未配置可用的 DeepSeek endpoint。")

    async def _astream(
        self,
        messages: list,
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        last_error: Optional[Exception] = None
        for state in self._pool.ordered_candidates():
            client = self._pool.get_client(state, self.model_name, self.temperature)
            self._pool.acquire(state)
            yielded = False
            try:
                _LAST_ENDPOINT_NAME.set(state.config.name or "")
                async for chunk in client._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
                    yielded = True
                    yield chunk
                self._pool.release_success(state)
                return
            except Exception as exc:
                self._pool.release_failure(state, exc)
                last_error = exc
                if yielded or not _is_retryable_error(exc):
                    raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("未配置可用的 DeepSeek endpoint。")


def get_deepseek_llm(
    temperature: float = 0,
    model: str | None = None,
) -> BaseChatModel:
    """
    创建 DeepSeek 对话模型实例。

    - 单节点时：兼容原有 `OPENAI_API_BASE` + `OPENAI_API_KEY`
    - 多节点时：使用 `DEEPSEEK_API_ENDPOINTS` 做加权调度、异常切换与熔断
    """

    s = get_settings()
    return _DeepSeekChatRouter(
        pool=_get_pool(),
        model_name=model or s.llm_model,
        temperature=float(temperature),
    )
