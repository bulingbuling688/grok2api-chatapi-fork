import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path


class _Missing:
    pass


MISSING = _Missing()


class FakeGrokApiException(Exception):
    def __init__(self, message, error_code=None, details=None, status_code=None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code


class FakeLogger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class FakeSchema:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _stub_module(name, module, originals):
    originals[name] = sys.modules.get(name, MISSING)
    sys.modules[name] = module


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def load_console_client_module():
    originals = {}

    _stub_module("app", _package("app"), originals)
    _stub_module("app.core", _package("app.core"), originals)
    _stub_module("app.models", _package("app.models"), originals)
    _stub_module("app.services", _package("app.services"), originals)
    _stub_module("app.services.grok", _package("app.services.grok"), originals)

    config = types.ModuleType("app.core.config")
    config.setting = types.SimpleNamespace(get_proxy_async=lambda _scope: None)
    _stub_module("app.core.config", config, originals)

    exception = types.ModuleType("app.core.exception")
    exception.GrokApiException = FakeGrokApiException
    _stub_module("app.core.exception", exception, originals)

    logger = types.ModuleType("app.core.logger")
    logger.logger = FakeLogger()
    _stub_module("app.core.logger", logger, originals)

    proxy_pool = types.ModuleType("app.core.proxy_pool")
    proxy_pool.proxy_pool = types.SimpleNamespace(enabled=False)
    _stub_module("app.core.proxy_pool", proxy_pool, originals)

    schema = types.ModuleType("app.models.openai_schema")
    for name in [
        "OpenAIChatCompletionChoice",
        "OpenAIChatCompletionChunkChoice",
        "OpenAIChatCompletionChunkResponse",
        "OpenAIChatCompletionMessage",
        "OpenAIChatCompletionResponse",
    ]:
        setattr(schema, name, FakeSchema)
    _stub_module("app.models.openai_schema", schema, originals)

    token = types.ModuleType("app.services.grok.token")
    token.token_manager = types.SimpleNamespace()
    _stub_module("app.services.grok.token", token, originals)

    curl_cffi = _package("curl_cffi")
    curl_requests = types.ModuleType("curl_cffi.requests")
    curl_requests.AsyncSession = object
    _stub_module("curl_cffi", curl_cffi, originals)
    _stub_module("curl_cffi.requests", curl_requests, originals)

    orjson = types.ModuleType("orjson")
    orjson.dumps = lambda value: json.dumps(value).encode()
    orjson.loads = json.loads
    _stub_module("orjson", orjson, originals)

    try:
        path = Path(__file__).resolve().parents[1] / "app" / "services" / "grok" / "console_client.py"
        spec = importlib.util.spec_from_file_location("console_client_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in originals.items():
            if original is MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_console_retries_next_token_after_429_http_error():
    async def exercise():
        module = load_console_client_module()

        class FakeTokenManager:
            def __init__(self):
                self.calls = 0

            async def get_token(self, model):
                self.calls += 1
                return f"token-{self.calls}"

        tokens_used = []

        async def fake_request(payload, token, model, stream):
            tokens_used.append(token)
            if len(tokens_used) == 1:
                raise module.GrokApiException(
                    "rate limited",
                    "HTTP_ERROR",
                    {"status": 429, "data": {"code": "resource-exhausted"}},
                )
            return {"ok": True, "token": token}

        async def no_sleep(_seconds):
            pass

        module.token_manager = FakeTokenManager()
        module.ConsoleClient._request = staticmethod(fake_request)
        module.asyncio.sleep = no_sleep

        result = await module.ConsoleClient._retry(
            "grok-4.20-multi-agent-xhigh",
            {"input": []},
            False,
        )

        assert result == {"ok": True, "token": "token-2"}
        assert tokens_used == ["token-1", "token-2"]

    asyncio.run(exercise())


def test_console_retries_all_available_tokens_after_429_http_errors():
    async def exercise():
        module = load_console_client_module()

        class FakeTokenManager:
            def __init__(self):
                self.tokens = ["token-1", "token-2", "token-3", "token-4"]
                self.calls = 0

            async def available_token_count(self, model):
                return len(self.tokens)

            async def get_token(self, model):
                token = self.tokens[self.calls]
                self.calls += 1
                return token

        tokens_used = []

        async def fake_request(payload, token, model, stream):
            tokens_used.append(token)
            if token != "token-4":
                raise module.GrokApiException(
                    "rate limited",
                    "HTTP_ERROR",
                    {"status": 429, "data": {"code": "resource-exhausted"}},
                )
            return {"ok": True, "token": token}

        async def no_sleep(_seconds):
            pass

        module.token_manager = FakeTokenManager()
        module.ConsoleClient._request = staticmethod(fake_request)
        module.asyncio.sleep = no_sleep

        result = await module.ConsoleClient._retry(
            "grok-4.20-multi-agent-xhigh",
            {"input": []},
            False,
        )

        assert result == {"ok": True, "token": "token-4"}
        assert tokens_used == ["token-1", "token-2", "token-3", "token-4"]

    asyncio.run(exercise())


def test_console_does_not_retry_team_level_429_across_token_pool():
    async def exercise():
        module = load_console_client_module()

        class FakeTokenManager:
            def __init__(self):
                self.tokens = ["token-1", "token-2", "token-3", "token-4"]
                self.calls = 0

            async def available_token_count(self, model):
                return len(self.tokens)

            async def get_token(self, model):
                token = self.tokens[self.calls]
                self.calls += 1
                return token

        tokens_used = []

        async def fake_request(payload, token, model, stream):
            tokens_used.append(token)
            raise module.GrokApiException(
                "rate limited",
                "HTTP_ERROR",
                {
                    "status": 429,
                    "data": {
                        "code": "resource-exhausted",
                        "error": (
                            "Too many requests for team 00000000-0000-0000-0000-000000000013 "
                            "and model grok-4.3. Requests per Minute (actual/limit): 313/60."
                        ),
                    },
                },
            )

        async def no_sleep(_seconds):
            pass

        module.token_manager = FakeTokenManager()
        module.ConsoleClient._request = staticmethod(fake_request)
        module.asyncio.sleep = no_sleep

        try:
            await module.ConsoleClient._retry(
                "grok-4.3-low",
                {"input": []},
                False,
            )
        except module.GrokApiException as exc:
            assert exc.details["status"] == 429
        else:
            raise AssertionError("expected team-level 429 to be raised")

        assert tokens_used == ["token-1"]

    asyncio.run(exercise())


def test_console_handle_error_preserves_upstream_429_status_code():
    async def exercise():
        module = load_console_client_module()

        class FakeTokenManager:
            async def record_failure(self, token, status, msg):
                pass

            async def apply_cooldown(self, token, status):
                pass

        class FakeResponse:
            status_code = 429

            def json(self):
                return {
                    "code": "resource-exhausted",
                    "error": "Too many requests for team t and model grok-4.3.",
                }

        module.token_manager = FakeTokenManager()

        try:
            await module.ConsoleClient._handle_error(FakeResponse(), "token")
        except module.GrokApiException as exc:
            assert exc.error_code == "HTTP_ERROR"
            assert exc.status_code == 429
        else:
            raise AssertionError("expected 429 to raise")

    asyncio.run(exercise())
