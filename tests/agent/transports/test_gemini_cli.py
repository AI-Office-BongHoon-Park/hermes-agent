from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

from agent.transports.gemini_cli import GeminiCliTransport


def test_gemini_cli_transport_invokes_headless_json(monkeypatch):
    calls = []

    def _fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout='{"response":"hello","stats":{"models":{"gemini-3-flash-preview":{"tokens":{"prompt":3,"candidates":4,"total":7}}}}}',
            stderr="",
        )

    monkeypatch.setattr("agent.transports.gemini_cli.subprocess.run", _fake_run)

    transport = GeminiCliTransport()
    api_kwargs = transport.build_kwargs(
        model="gemini-3-flash-preview",
        messages=[{"role": "user", "content": "Say hello"}],
        command="/usr/local/bin/gemini",
        args=["--sandbox"],
        timeout=5,
    )
    raw = transport.invoke(api_kwargs)
    normalized = transport.normalize_response(raw)

    argv, kwargs = calls[0]
    assert argv == [
        "/usr/local/bin/gemini",
        "--sandbox",
        "--model",
        "gemini-3-flash-preview",
        "--output-format",
        "json",
        "--prompt",
        "",
    ]
    assert "Say hello" in kwargs["input"]
    assert kwargs["text"] is True
    assert kwargs["timeout"] == 5
    assert normalized.content == "hello"
    assert normalized.usage.prompt_tokens == 3
    assert normalized.usage.completion_tokens == 4
    assert normalized.usage.total_tokens == 7


def test_gemini_cli_transport_includes_tool_protocol_in_prompt():
    transport = GeminiCliTransport()
    api_kwargs = transport.build_kwargs(
        model="gemini-3-flash-preview",
        messages=[{"role": "user", "content": "Read the file"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
    )

    prompt = api_kwargs["prompt"]
    assert "Hermes tools are available" in prompt
    assert '"tool_calls"' in prompt
    assert '"name":"read_file"' in prompt
    assert api_kwargs["tools"][0]["name"] == "read_file"


def test_gemini_cli_transport_parses_tool_call_protocol():
    transport = GeminiCliTransport()

    normalized = transport.normalize_response(
        {
            "response": (
                '{"tool_calls":[{"name":"read_file",'
                '"arguments":{"path":"README.md","limit":2}}]}'
            )
        }
    )

    assert normalized.finish_reason == "tool_calls"
    assert normalized.content == ""
    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].function.name == "read_file"
    assert normalized.tool_calls[0].function.arguments == '{"path": "README.md", "limit": 2}'


def test_gemini_cli_transport_unwraps_final_response_protocol():
    transport = GeminiCliTransport()

    normalized = transport.normalize_response(
        {"response": '{"response":"done from protocol"}'}
    )

    assert normalized.finish_reason == "stop"
    assert normalized.tool_calls is None
    assert normalized.content == "done from protocol"


def test_gemini_cli_transport_accepts_plain_text(monkeypatch):
    monkeypatch.setattr(
        "agent.transports.gemini_cli.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="plain answer",
            stderr="",
        ),
    )

    transport = GeminiCliTransport()
    raw = transport.invoke(
        transport.build_kwargs(
            model="gemini-3-flash-preview",
            messages=[{"role": "user", "content": "Question"}],
        )
    )
    normalized = transport.normalize_response(raw)

    assert normalized.content == "plain answer"


def test_gemini_cli_transport_resumes_or_creates_session(monkeypatch):
    calls = []

    def _fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if "--resume" in argv:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr='Invalid session identifier "missing"',
            )
        return SimpleNamespace(
            returncode=0,
            stdout='{"response":"created"}',
            stderr="",
        )

    monkeypatch.setattr("agent.transports.gemini_cli.subprocess.run", _fake_run)

    transport = GeminiCliTransport()
    api_kwargs = transport.build_kwargs(
        model="gemini-3-flash-preview",
        messages=[
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "First turn"},
        ],
        command="/usr/local/bin/gemini",
        session_id="hermes-session-a",
    )
    raw = transport.invoke(api_kwargs)
    normalized = transport.normalize_response(raw)

    assert normalized.content == "created"
    assert len(calls) == 2
    assert "--resume" in calls[0][0]
    assert "--session-id" in calls[1][0]
    assert UUID(calls[0][0][-1]) == UUID(calls[1][0][-1])
    assert "Conversation transcript:" in calls[1][1]["input"]
    assert "First turn" in calls[1][1]["input"]


def test_gemini_cli_transport_resume_prompt_sends_only_new_items_after_first_call():
    transport = GeminiCliTransport()
    session_id = "hermes-session-b"
    first_kwargs = transport.build_kwargs(
        model="gemini-3-flash-preview",
        messages=[
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "First turn"},
        ],
        session_id=session_id,
    )
    transport.normalize_response(
        {
            "response": "First answer",
            "_hermes_gemini_cli_session_id": first_kwargs["session_id"],
            "_hermes_message_count": first_kwargs["message_count"],
        }
    )

    second_kwargs = transport.build_kwargs(
        model="gemini-3-flash-preview",
        messages=[
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "First turn"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second turn"},
        ],
        session_id=session_id,
    )

    assert "New transcript items:" in second_kwargs["resume_prompt"]
    assert "Second turn" in second_kwargs["resume_prompt"]
    assert "First turn" not in second_kwargs["resume_prompt"]
    assert "First answer" not in second_kwargs["resume_prompt"]
