"""Transport for invoking the local Gemini CLI as an inference backend.

Hermes does not resolve Google OAuth tokens or API keys for this backend.  It
shells out to the user-installed ``gemini`` command and relies on that CLI's
own local authentication/session state.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from typing import Any, Dict, List, Optional

from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, ToolCall, Usage


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") in {"text", "input_text"}:
                    parts.append(str(part.get("text") or ""))
                elif part.get("type") in {"image_url", "input_image"}:
                    image = part.get("image_url") or part.get("image") or {}
                    url = image.get("url") if isinstance(image, dict) else image
                    if url:
                        parts.append(f"[image: {url}]")
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "")
        return json.dumps(content, ensure_ascii=True)
    return str(content)


def _tool_to_prompt_entry(tool: Dict[str, Any]) -> Dict[str, Any] | None:
    if not isinstance(tool, dict):
        return None
    fn = tool.get("function") if tool.get("type") == "function" else tool
    if not isinstance(fn, dict):
        return None
    name = fn.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    return {
        "name": name.strip(),
        "description": fn.get("description") or "",
        "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
    }


def _messages_to_prompt(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    *,
    resumed_session: bool = False,
) -> str:
    if resumed_session:
        lines: list[str] = [
            "You are running as the model backend for Hermes Agent.",
            "Continue the existing Gemini CLI session for Hermes Agent.",
            (
                "Use only the new transcript items below; previous context is "
                "already in this CLI session."
            ),
            "",
        ]
    else:
        lines = [
            "You are running as the model backend for Hermes Agent.",
            "Answer the latest user request using the conversation transcript below.",
            "",
        ]
    if tools:
        lines.extend(
            [
                "Hermes tools are available. To call tools, respond with only a JSON object in this exact protocol:",
                '{"tool_calls":[{"name":"tool_name","arguments":{"arg":"value"}}]}',
                "When you have the final answer, respond with only:",
                '{"response":"final answer text"}',
                "Do not wrap protocol JSON in Markdown fences.",
                "",
                "Available Hermes tools:",
                json.dumps(tools, ensure_ascii=True, separators=(",", ":")),
                "",
            ]
        )
    lines.append("New transcript items:" if resumed_session else "Conversation transcript:")
    for msg in messages:
        role = str(msg.get("role") or "user")
        content = _content_to_text(msg.get("content"))
        if role == "assistant" and msg.get("tool_calls"):
            tool_lines = []
            for call in msg.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                tool_lines.append(
                    f"- {fn.get('name', 'unknown')}({fn.get('arguments', '')})"
                )
            if tool_lines:
                content = "\n".join([content, "Tool calls requested:", *tool_lines]).strip()
        elif role == "tool":
            name = msg.get("name") or msg.get("tool_call_id") or "tool"
            role = f"tool:{name}"
        lines.append(f"\n[{role}]\n{content}")
    lines.append("\n[assistant]\n")
    return "\n".join(lines)


def _stable_cli_session_id(session_id: Any) -> str:
    raw = str(session_id or "").strip()
    if not raw:
        return ""
    try:
        return str(uuid.UUID(raw))
    except (TypeError, ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"hermes-agent:gemini-cli:{raw}"))


def _latest_replay_start(messages: List[Dict[str, Any]]) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], dict) and messages[idx].get("role") == "user":
            return idx
    return max(0, len(messages) - 1)


def _has_cli_session_arg(argv: list[str]) -> bool:
    session_flags = {"--resume", "-r", "--session-id", "--session-file"}
    for arg in argv:
        if arg in session_flags:
            return True
        if any(arg.startswith(flag + "=") for flag in session_flags):
            return True
    return False


def _is_missing_session_error(text: str) -> bool:
    lowered = (text or "").lower()
    return (
        "invalid session identifier" in lowered
        or "no previous sessions found" in lowered
        or "failed to find session" in lowered
    )


def _json_from_stdout(stdout: str) -> Any:
    text = (stdout or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {"response": text}
    return {"response": text}


def _usage_from_stats(stats: Any) -> Usage | None:
    if not isinstance(stats, dict):
        return None
    models = stats.get("models")
    if not isinstance(models, dict):
        return None
    prompt = completion = total = cached = 0
    seen = False
    for model_stats in models.values():
        if not isinstance(model_stats, dict):
            continue
        tokens = model_stats.get("tokens")
        if not isinstance(tokens, dict):
            continue
        seen = True
        prompt += int(tokens.get("prompt") or tokens.get("input") or 0)
        completion += int(tokens.get("candidates") or tokens.get("output") or 0)
        total += int(tokens.get("total") or 0)
        cached += int(tokens.get("cached") or 0)
    if not seen:
        return None
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total or prompt + completion,
        cached_tokens=cached,
    )


def _parse_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _tool_calls_from_protocol(protocol: dict[str, Any]) -> list[ToolCall] | None:
    raw_calls = protocol.get("tool_calls")
    if raw_calls is None:
        raw_calls = protocol.get("tools")
    if not isinstance(raw_calls, list):
        return None

    calls: list[ToolCall] = []
    for idx, raw in enumerate(raw_calls):
        if not isinstance(raw, dict):
            continue
        fn = raw.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            arguments = fn.get("arguments", {})
        else:
            name = raw.get("name") or raw.get("tool")
            arguments = raw.get("arguments", raw.get("args", raw.get("input", {})))
        if not isinstance(name, str) or not name.strip():
            continue
        if isinstance(arguments, str):
            args_text = arguments
        else:
            args_text = json.dumps(arguments or {}, ensure_ascii=True)
        call_id = raw.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"call_gemini_cli_{uuid.uuid4().hex[:12]}_{idx}"
        calls.append(
            ToolCall(
                id=call_id,
                name=name.strip(),
                arguments=args_text,
            )
        )
    return calls or None


class GeminiCliTransport(ProviderTransport):
    """Run ``gemini --prompt`` in headless mode and normalize its response."""

    def __init__(self) -> None:
        self._session_state: dict[str, dict[str, Any]] = {}

    @property
    def api_mode(self) -> str:
        return "gemini_cli"

    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        return _messages_to_prompt(
            messages,
            kwargs.get("tools"),
            resumed_session=bool(kwargs.get("resumed_session")),
        )

    def convert_tools(self, tools: List[Dict[str, Any]]) -> list:
        converted = [_tool_to_prompt_entry(tool) for tool in tools or []]
        return [tool for tool in converted if tool]

    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params,
    ) -> Dict[str, Any]:
        converted_tools = self.convert_tools(tools or [])
        cli_session_id = _stable_cli_session_id(params.get("session_id"))
        message_count = len(messages or [])
        resume_start = 0
        if cli_session_id:
            state = self._session_state.setdefault(cli_session_id, {})
            resume_start = int(state.get("sent_count") or 0)
            if resume_start <= 0 or resume_start > message_count:
                resume_start = _latest_replay_start(messages or [])
        resume_messages = list(messages or [])[resume_start:]
        return {
            "model": model,
            "prompt": self.convert_messages(messages, tools=converted_tools),
            "resume_prompt": self.convert_messages(
                resume_messages,
                tools=converted_tools,
                resumed_session=True,
            ) if cli_session_id else "",
            "tools": converted_tools,
            "command": params.get("command") or "gemini",
            "args": list(params.get("args") or ()),
            "cwd": params.get("cwd") or os.getcwd(),
            "timeout": params.get("timeout"),
            "output_format": params.get("output_format") or "json",
            "session_id": cli_session_id,
            "message_count": message_count,
        }

    def _run_cli(
        self,
        argv: list[str],
        api_kwargs: Dict[str, Any],
        prompt: str,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            argv,
            input=prompt,
            text=True,
            capture_output=True,
            cwd=api_kwargs.get("cwd") or None,
            timeout=api_kwargs.get("timeout") or None,
            check=False,
        )

    def invoke(self, api_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        command = api_kwargs.get("command") or "gemini"
        args = list(api_kwargs.get("args") or [])
        model = str(api_kwargs.get("model") or "").strip()
        output_format = str(api_kwargs.get("output_format") or "json").strip()
        cli_session_id = str(api_kwargs.get("session_id") or "").strip()

        argv = [command, *args]
        if model and "--model" not in argv and "-m" not in argv:
            argv.extend(["--model", model])
        if output_format and "--output-format" not in argv:
            argv.extend(["--output-format", output_format])
        if "--prompt" not in argv and "-p" not in argv:
            argv.extend(["--prompt", ""])

        try:
            auto_session = bool(cli_session_id) and not _has_cli_session_arg(argv)
            used_resume = False
            if auto_session:
                resume_argv = [*argv, "--resume", cli_session_id]
                completed = self._run_cli(
                    resume_argv,
                    api_kwargs,
                    api_kwargs.get("resume_prompt") or api_kwargs.get("prompt") or "",
                )
                used_resume = completed.returncode == 0
                if completed.returncode != 0 and _is_missing_session_error(
                    "\n".join([completed.stderr or "", completed.stdout or ""])
                ):
                    create_argv = [*argv, "--session-id", cli_session_id]
                    completed = self._run_cli(
                        create_argv,
                        api_kwargs,
                        api_kwargs.get("prompt") or "",
                    )
                    used_resume = False
            else:
                completed = self._run_cli(argv, api_kwargs, api_kwargs.get("prompt") or "")
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Gemini CLI command not found: {command!r}. Install @google/gemini-cli "
                "or set model.gemini_cli.command in config.yaml."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Gemini CLI timed out after {exc.timeout}s") from exc

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr or stdout or f"exit code {completed.returncode}"
            raise RuntimeError(f"Gemini CLI failed: {detail}")

        data = _json_from_stdout(completed.stdout)
        if isinstance(data, dict):
            data.setdefault("_stderr", completed.stderr or "")
            if cli_session_id:
                data.setdefault("_hermes_gemini_cli_session_id", cli_session_id)
                data.setdefault("_hermes_message_count", api_kwargs.get("message_count") or 0)
                data.setdefault("_hermes_used_resume", used_resume)
            return data
        response = {"response": str(data), "_stderr": completed.stderr or ""}
        if cli_session_id:
            response["_hermes_gemini_cli_session_id"] = cli_session_id
            response["_hermes_message_count"] = api_kwargs.get("message_count") or 0
            response["_hermes_used_resume"] = used_resume
        return response

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        session_id = ""
        message_count = 0
        if isinstance(response, dict):
            session_id = str(response.get("_hermes_gemini_cli_session_id") or "")
            try:
                message_count = int(response.get("_hermes_message_count") or 0)
            except (TypeError, ValueError):
                message_count = 0
            content = response.get("response")
            if content is None:
                content = response.get("text") or response.get("content") or ""
            protocol = response if isinstance(response.get("tool_calls"), list) else None
            if protocol is None and isinstance(content, str):
                protocol = _parse_json_object(content)
            tool_calls = _tool_calls_from_protocol(protocol or {})
            usage = _usage_from_stats(response.get("stats"))
            provider_data = {
                k: v for k, v in response.items()
                if k not in {
                    "response",
                    "text",
                    "content",
                    "stats",
                    "tool_calls",
                    "_hermes_gemini_cli_session_id",
                    "_hermes_message_count",
                }
            } or None
            if protocol and tool_calls:
                protocol_content = (
                    protocol.get("response")
                    or protocol.get("content")
                    or protocol.get("message")
                    or ""
                )
                content = "" if isinstance(protocol_content, list) else str(protocol_content or "")
                finish_reason = "tool_calls"
            else:
                if protocol and not tool_calls:
                    content = (
                        protocol.get("response")
                        or protocol.get("content")
                        or protocol.get("message")
                        or content
                    )
                finish_reason = "stop"
        else:
            content = str(response or "")
            protocol = _parse_json_object(content)
            tool_calls = _tool_calls_from_protocol(protocol or {})
            if protocol and tool_calls:
                content = str(protocol.get("response") or protocol.get("content") or "")
                finish_reason = "tool_calls"
            else:
                if protocol and not tool_calls:
                    content = str(
                        protocol.get("response")
                        or protocol.get("content")
                        or protocol.get("message")
                        or content
                    )
                finish_reason = "stop"
            usage = None
            provider_data = None
        if session_id and message_count > 0:
            state = self._session_state.setdefault(session_id, {})
            state["sent_count"] = max(int(state.get("sent_count") or 0), message_count + 1)
            state["created"] = True
        return NormalizedResponse(
            content=str(content or ""),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            provider_data=provider_data,
        )

    def validate_response(self, response: Any) -> bool:
        if response is None:
            return False
        if isinstance(response, dict):
            return bool(
                response.get("tool_calls")
                or str(
                    response.get("response")
                    or response.get("text")
                    or response.get("content")
                    or ""
                ).strip()
            )
        return bool(str(response).strip())


from agent.transports import register_transport  # noqa: E402

register_transport("gemini_cli", GeminiCliTransport)
