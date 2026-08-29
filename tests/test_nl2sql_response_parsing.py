"""generate_sql/summarize_result must find the TEXT block in the response,
not assume it's at content[0]. claude-sonnet-5 puts a `thinking` block
(text=None) first when thinking is enabled by default — this crashed every
real /ask call with AttributeError, uncaught by any existing test because
every other test mocks generate_sql/summarize_result themselves rather than
the underlying Anthropic response shape. First reproduced on the first real
end-to-end call after the first Render deploy."""
from types import SimpleNamespace

from app.nl2sql import generate, summarize


def _response(*blocks):
    return SimpleNamespace(content=[SimpleNamespace(**b) for b in blocks])


def test_generate_sql_skips_a_leading_thinking_block(monkeypatch):
    thinking_then_text = _response(
        {"type": "thinking", "text": None, "thinking": "reasoning..."},
        {"type": "text", "text": "SELECT * FROM events"},
    )
    monkeypatch.setattr(generate, "_get_client", lambda: SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kw: thinking_then_text)
    ))
    assert generate.generate_sql("any question") == "SELECT * FROM events"


def test_generate_sql_still_works_with_text_only_response(monkeypatch):
    # The pre-thinking-model shape must keep working too.
    text_only = _response({"type": "text", "text": "SELECT 1"})
    monkeypatch.setattr(generate, "_get_client", lambda: SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kw: text_only)
    ))
    assert generate.generate_sql("any question") == "SELECT 1"


def test_summarize_result_skips_a_leading_thinking_block(monkeypatch):
    thinking_then_text = _response(
        {"type": "thinking", "text": None, "thinking": "reasoning..."},
        {"type": "text", "text": "Yes, there was a spike."},
    )
    monkeypatch.setattr(summarize, "_get_client", lambda: SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kw: thinking_then_text)
    ))
    assert summarize.summarize_result("q", [{"a": 1}]) == "Yes, there was a spike."
