import importlib.util
import sys
import types
from pathlib import Path


class _Missing:
    pass


MISSING = _Missing()


class FakeGrokApiException(Exception):
    pass


class FakeLogger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _stub_module(name, module, originals):
    originals[name] = sys.modules.get(name, MISSING)
    sys.modules[name] = module


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def load_token_module():
    originals = {}

    _stub_module("app", _package("app"), originals)
    _stub_module("app.models", _package("app.models"), originals)
    _stub_module("app.core", _package("app.core"), originals)
    _stub_module("app.services", _package("app.services"), originals)
    _stub_module("app.services.grok", _package("app.services.grok"), originals)

    models = types.ModuleType("app.models.grok_models")
    models.TokenType = types.SimpleNamespace(
        NORMAL=types.SimpleNamespace(value="ssoNormal"),
        SUPER=types.SimpleNamespace(value="ssoSuper"),
    )
    models.Models = types.SimpleNamespace(to_rate_limit=lambda model: model)
    _stub_module("app.models.grok_models", models, originals)

    exception = types.ModuleType("app.core.exception")
    exception.GrokApiException = FakeGrokApiException
    _stub_module("app.core.exception", exception, originals)

    logger = types.ModuleType("app.core.logger")
    logger.logger = FakeLogger()
    _stub_module("app.core.logger", logger, originals)

    config = types.ModuleType("app.core.config")
    config.setting = types.SimpleNamespace(global_config={}, grok_config={})
    _stub_module("app.core.config", config, originals)

    proxy_pool = types.ModuleType("app.core.proxy_pool")
    proxy_pool.proxy_pool = types.SimpleNamespace(enabled=False)
    _stub_module("app.core.proxy_pool", proxy_pool, originals)

    statsig = types.ModuleType("app.services.grok.statsig")
    statsig.get_dynamic_headers = lambda _path: {}
    _stub_module("app.services.grok.statsig", statsig, originals)

    portalocker = types.ModuleType("portalocker")
    portalocker.LOCK_SH = 1
    portalocker.LOCK_EX = 2
    portalocker.lock = lambda *args, **kwargs: None
    portalocker.unlock = lambda *args, **kwargs: None
    _stub_module("portalocker", portalocker, originals)

    curl_cffi = _package("curl_cffi")
    curl_requests = types.ModuleType("curl_cffi.requests")
    curl_requests.AsyncSession = object
    _stub_module("curl_cffi", curl_cffi, originals)
    _stub_module("curl_cffi.requests", curl_requests, originals)

    orjson = types.ModuleType("orjson")
    _stub_module("orjson", orjson, originals)

    try:
        path = Path(__file__).resolve().parents[1] / "app" / "services" / "grok" / "token.py"
        spec = importlib.util.spec_from_file_location("token_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in originals.items():
            if original is MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_extract_sso_reads_cookie_field_even_when_token_contains_sso_text():
    module = load_token_module()
    jwt = "header.payload-with-sso=inside.signature"
    cookie = f"sso-rw={jwt};sso={jwt}"

    assert module.GrokTokenManager._extract_sso(cookie) == jwt


def test_get_token_keeps_existing_cookie_token_without_double_wrapping():
    async def exercise():
        module = load_token_module()
        manager = module.GrokTokenManager.__new__(module.GrokTokenManager)
        raw_cookie = "sso-rw=header.payload.signature;sso=header.payload.signature;other=value"

        async def select_token(_model):
            return raw_cookie

        manager.select_token = select_token

        assert await manager.get_token("grok-4.20-multi-agent-xhigh") == raw_cookie

    import asyncio

    asyncio.run(exercise())


def test_get_token_wraps_bare_sso_jwt_cookie():
    async def exercise():
        module = load_token_module()
        manager = module.GrokTokenManager.__new__(module.GrokTokenManager)
        jwt = "header.payload.signature"

        async def select_token(_model):
            return jwt

        manager.select_token = select_token

        assert await manager.get_token("grok-4.20-multi-agent-xhigh") == f"sso-rw={jwt};sso={jwt}"

    import asyncio

    asyncio.run(exercise())


def test_record_failure_does_not_increment_failed_count_for_rate_limits():
    async def exercise():
        module = load_token_module()
        manager = module.GrokTokenManager.__new__(module.GrokTokenManager)
        token = "header.payload.signature"
        manager.token_data = {
            "ssoNormal": {
                token: {
                    "status": "active",
                    "failedCount": 0,
                    "remainingQueries": -1,
                }
            },
            "ssoSuper": {},
        }
        manager._mark_dirty = lambda: None

        await manager.record_failure(
            f"sso-rw={token};sso={token}",
            429,
            "Requests per Minute (actual/limit): 90/60",
        )

        data = manager.token_data["ssoNormal"][token]
        assert data["failedCount"] == 0
        assert "lastRateLimitReason" in data

    import asyncio

    asyncio.run(exercise())


def test_rate_limit_cooldown_uses_short_window_from_error_message():
    async def exercise():
        module = load_token_module()
        manager = module.GrokTokenManager.__new__(module.GrokTokenManager)
        token = "header.payload.signature"
        manager.token_data = {
            "ssoNormal": {
                token: {
                    "status": "active",
                    "failedCount": 0,
                    "remainingQueries": -1,
                    "lastFailureReason": "429: Requests per Minute (actual/limit): 90/60",
                }
            },
            "ssoSuper": {},
        }
        manager._mark_dirty = lambda: None

        before = module.time.time()
        await manager.apply_cooldown(f"sso-rw={token};sso={token}", 429)
        cooldown_seconds = (manager.token_data["ssoNormal"][token]["cooldownUntil"] / 1000) - before

        assert 60 <= cooldown_seconds <= 180

    import asyncio

    asyncio.run(exercise())
