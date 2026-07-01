"""Console.x.ai Responses API client for SSO-backed models."""

import asyncio
import hashlib
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List

import orjson
from curl_cffi.requests import AsyncSession as curl_AsyncSession

from app.core.config import setting
from app.core.exception import GrokApiException
from app.core.logger import logger
from app.core.proxy_pool import proxy_pool
from app.models.openai_schema import (
    OpenAIChatCompletionChoice,
    OpenAIChatCompletionChunkChoice,
    OpenAIChatCompletionChunkResponse,
    OpenAIChatCompletionMessage,
    OpenAIChatCompletionResponse,
)
from app.services.grok.token import token_manager


CONSOLE_ENDPOINT = "https://console.x.ai/v1/responses"
BROWSER = "chrome133a"
TIMEOUT = 120
MAX_RETRY = 3
RETRYABLE_HTTP_STATUS = {401, 429}

CONSOLE_MODELS: dict[str, str] = {
    "grok-4.3-console": "grok-4.3",
    "grok-4.3-low": "grok-4.3",
    "grok-4.3-medium": "grok-4.3",
    "grok-4.3-high": "grok-4.3",
    "grok-4.20-0309-reasoning-console": "grok-4.20-0309-reasoning",
    "grok-4.20-0309-console": "grok-4.20-0309",
    "grok-4.20-0309-non-reasoning-console": "grok-4.20-0309-non-reasoning",
    "grok-4.20-multi-agent-console": "grok-4.20-multi-agent-0309",
    "grok-4.20-multi-agent-low": "grok-4.20-multi-agent-0309",
    "grok-4.20-multi-agent-medium": "grok-4.20-multi-agent-0309",
    "grok-4.20-multi-agent-high": "grok-4.20-multi-agent-0309",
    "grok-4.20-multi-agent-xhigh": "grok-4.20-multi-agent-0309",
    "grok-build-console": "grok-build-0.1",
}

FIXED_EFFORT: dict[str, str] = {
    "grok-4.3-low": "low",
    "grok-4.3-medium": "medium",
    "grok-4.3-high": "high",
    "grok-4.20-multi-agent-low": "low",
    "grok-4.20-multi-agent-medium": "medium",
    "grok-4.20-multi-agent-high": "high",
    "grok-4.20-multi-agent-xhigh": "xhigh",
}

WITH_REASONING = {"grok-4.3", "grok-4.20-multi-agent-0309"}
WITH_SEARCH_TOOLS = {
    "grok-4.3",
    "grok-4.20-0309",
    "grok-4.20-0309-reasoning",
    "grok-4.20-0309-non-reasoning",
    "grok-4.20-multi-agent-0309",
    "grok-build-0.1",
}
MAX_OUTPUT_TOKENS = {
    "grok-4.20-multi-agent-0309": 2_000_000,
    "grok-build-0.1": 256_000,
}


class ConsoleClient:
    """OpenAI chat-completions adapter for console.x.ai SSO models."""

    _shared_rate_limit_until: dict[str, float] = {}

    @staticmethod
    async def openai_to_console(request: dict):
        model = request["model"]
        stream = request.get("stream", False)
        payload = ConsoleClient._build_payload(request)
        return await ConsoleClient._retry(model, payload, stream)

    @staticmethod
    def _build_payload(request: dict) -> Dict[str, Any]:
        model = request["model"]
        console_model = CONSOLE_MODELS.get(model, model)
        effort = FIXED_EFFORT.get(model) or request.get("reasoning_effort") or "medium"

        payload: Dict[str, Any] = {
            "model": console_model,
            "input": ConsoleClient._messages_to_input(request.get("messages", [])),
            "max_output_tokens": request.get("max_tokens") or MAX_OUTPUT_TOKENS.get(console_model, 1_000_000),
            "temperature": request.get("temperature", 0.7),
            "top_p": request.get("top_p", 0.95),
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "stream": bool(request.get("stream", False)),
        }

        if console_model in WITH_REASONING:
            payload["reasoning"] = {"effort": effort}

        if console_model in WITH_SEARCH_TOOLS:
            payload["tools"] = [
                {"type": "web_search", "enable_image_understanding": True},
                {"type": "x_search", "enable_video_understanding": True},
            ]
            payload["tool_choice"] = "auto"

        return payload

    @staticmethod
    def _messages_to_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            api_role = role if role in {"system", "assistant", "user"} else "user"
            content = msg.get("content", "")
            blocks: List[Dict[str, Any]] = []

            if isinstance(content, str):
                blocks.append({"type": "input_text", "text": content})
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        blocks.append({"type": "input_text", "text": block.get("text", "")})
                    elif block.get("type") == "image_url":
                        url = (block.get("image_url") or {}).get("url", "")
                        if url:
                            blocks.append({"type": "input_image", "image_url": url})
                    else:
                        blocks.append({"type": "input_text", "text": str(block)})
            else:
                blocks.append({"type": "input_text", "text": str(content)})

            if blocks:
                items.append({"role": api_role, "content": blocks})
        return items

    @staticmethod
    async def _retry(model: str, payload: dict, stream: bool):
        last_err = None
        shared_cooldown = ConsoleClient._shared_rate_limit_until.get(model, 0)
        if shared_cooldown > time.time():
            retry_after = int(shared_cooldown - time.time())
            raise GrokApiException(
                f"Console upstream rate limited for {model}, retry after {retry_after}s",
                "HTTP_ERROR",
                {"status": 429, "data": {"code": "resource-exhausted"}},
                status_code=429,
            )

        max_attempts = await ConsoleClient._max_attempts(model)
        for i in range(max_attempts):
            token_id = "unknown"
            try:
                token = await token_manager.get_token(model)
                token_id = ConsoleClient._token_fingerprint(token)
                return await ConsoleClient._request(payload, token, model, stream)
            except GrokApiException as exc:
                last_err = exc
                status = ConsoleClient._error_status(exc)
                if ConsoleClient._is_shared_rate_limit(exc):
                    ConsoleClient._shared_rate_limit_until[model] = time.time() + 70
                    logger.warning(
                        f"[Console] 上游返回 team/model 级 {status} 限速，"
                        f"停止轮询Token: {model}"
                    )
                    raise
                retryable = exc.error_code == "NO_AVAILABLE_TOKEN" or (
                    exc.error_code == "HTTP_ERROR" and status in RETRYABLE_HTTP_STATUS
                )
                if not retryable or i >= max_attempts - 1:
                    raise
                if exc.error_code == "HTTP_ERROR":
                    logger.warning(
                        f"[Console] Token {token_id} 返回 {status}，换下一个Token重试 "
                        f"({i + 1}/{max_attempts})"
                    )
                else:
                    logger.warning(
                        f"[Console] Token不可用，换Token重试 ({i + 1}/{max_attempts})"
                    )
                await asyncio.sleep(0.5)
        raise last_err or GrokApiException("Console request failed", "REQUEST_ERROR")

    @staticmethod
    async def _max_attempts(model: str) -> int:
        try:
            available = await token_manager.available_token_count(model)
        except Exception as exc:
            logger.warning(f"[Console] 获取可用Token数量失败，使用默认重试次数: {exc}")
            available = 0
        return max(MAX_RETRY, available or 0)

    @staticmethod
    def _error_status(exc: GrokApiException) -> int | None:
        if isinstance(exc.details, dict):
            return exc.details.get("status")
        return None

    @staticmethod
    def _is_shared_rate_limit(exc: GrokApiException) -> bool:
        if ConsoleClient._error_status(exc) != 429:
            return False
        data = exc.details.get("data") if isinstance(exc.details, dict) else None
        if isinstance(data, dict):
            message = f"{data.get('code', '')} {data.get('error', '')}"
        else:
            message = str(data or exc)
        message = message.lower()
        return "resource-exhausted" in message and "for team" in message and "and model" in message

    @staticmethod
    def _token_fingerprint(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()[:10]

    @staticmethod
    async def _request(payload: dict, token: str, model: str, stream: bool):
        proxy = await setting.get_proxy_async("service")
        proxies = {"http": proxy, "https": proxy} if proxy else None
        headers = ConsoleClient._headers(token)
        session = curl_AsyncSession(impersonate=BROWSER)
        try:
            response = await session.post(
                CONSOLE_ENDPOINT,
                headers=headers,
                data=orjson.dumps(payload),
                timeout=TIMEOUT,
                stream=stream,
                proxies=proxies,
            )
            if response.status_code != 200:
                await ConsoleClient._handle_error(response, token)

            asyncio.create_task(token_manager.reset_failure(token))
            if stream:
                return ConsoleClient._process_stream(response, session, model)

            try:
                return await ConsoleClient._process_normal(response, model)
            finally:
                await session.close()
        except Exception:
            await session.close()
            raise

    @staticmethod
    def _headers(token: str) -> Dict[str, str]:
        return {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "Authorization": "Bearer anonymous",
            "Origin": "https://console.x.ai",
            "Referer": "https://console.x.ai/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
            "Cookie": token,
        }

    @staticmethod
    async def _handle_error(response, token: str):
        try:
            data = response.json()
            msg = str(data)
        except Exception:
            data = response.text
            msg = data[:200] if data else "unknown error"
        await token_manager.record_failure(token, response.status_code, msg)
        await token_manager.apply_cooldown(token, response.status_code)
        raise GrokApiException(
            f"Console request failed: {response.status_code} - {msg}",
            "HTTP_ERROR",
            {"status": response.status_code, "data": data},
            status_code=response.status_code if response.status_code == 429 else None,
        )

    @staticmethod
    async def _process_normal(response, model: str) -> OpenAIChatCompletionResponse:
        data = response.json()
        content = ConsoleClient._extract_text(data)
        usage = data.get("usage")
        return ConsoleClient._build_response(content, model, usage)

    @staticmethod
    async def _process_stream(response, session, model: str) -> AsyncGenerator[str, None]:
        stream_id = f"chatcmpl-{uuid.uuid4()}"
        event_type = None
        try:
            async for line in response.aiter_lines():
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="ignore")
                if not line:
                    continue
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    data = orjson.loads(raw)
                except Exception:
                    continue
                if event_type == "response.output_text.delta":
                    delta = data.get("delta", "")
                    if delta:
                        yield ConsoleClient._chunk(stream_id, model, delta)
                elif event_type == "response.completed":
                    break
            yield ConsoleClient._chunk(stream_id, model, "", finish="stop")
            yield "data: [DONE]\n\n"
        finally:
            await session.close()

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        parts: List[str] = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    parts.append(content.get("text", ""))
        return "".join(parts)

    @staticmethod
    def _build_response(content: str, model: str, usage: dict | None = None) -> OpenAIChatCompletionResponse:
        return OpenAIChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                OpenAIChatCompletionChoice(
                    index=0,
                    message=OpenAIChatCompletionMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=ConsoleClient._usage(usage),
        )

    @staticmethod
    def _usage(usage: dict | None) -> dict | None:
        if not usage:
            return None
        return {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "completion_tokens_details": usage.get("output_tokens_details", {}),
        }

    @staticmethod
    def _chunk(stream_id: str, model: str, content: str, finish: str | None = None) -> str:
        chunk = OpenAIChatCompletionChunkResponse(
            id=stream_id,
            created=int(time.time()),
            model=model,
            choices=[
                OpenAIChatCompletionChunkChoice(
                    index=0,
                    delta={"role": "assistant", "content": content} if content else {},
                    finish_reason=finish,
                )
            ],
        )
        return f"data: {chunk.model_dump_json()}\n\n"
