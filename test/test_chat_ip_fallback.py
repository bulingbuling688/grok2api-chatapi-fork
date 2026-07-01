from pathlib import Path


def test_chat_ip_has_client_fallback_guard():
    chat_file = Path(__file__).resolve().parents[1] / "app" / "api" / "v1" / "chat.py"
    source = chat_file.read_text(encoding="utf-8")

    assert 'request.client.host if request.client else "unknown"' in source
