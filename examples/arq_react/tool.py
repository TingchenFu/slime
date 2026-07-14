from dataclasses import dataclass
from typing import Any
from collection.abc import Callable, Awaitable
from sandbox import SandboxHandle
import json
import time
import uuid
import inspect


@dataclass
class ToolDef:
    """One tool: JSON schema for the model + the async handler we dispatch to."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for {properties, required, type=object}
    handler: Callable[..., Awaitable[Any]]

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _bash_tool(handle: SandboxHandle, timeout: int | Callable[[], int | None] | None = None) -> ToolDef:
    async def execute(cmd: str) -> str:
        resolved_timeout = timeout() if callable(timeout) else timeout
        cmd_specific_timeout = resolved_timeout
        if "apply_patch" in cmd and "<<" in cmd:
            cmd_specific_timeout = 30
        try:
            result = await handle.exec(
                cmd=["bash", "--login", "-c", cmd],
                # Send an empty stdin so commands that read from stdin
                # (``cat > file``, ``python -``, etc.) immediately see EOF
                # instead of hanging the entire rollout. The model issues
                # ``cmd`` as a self-contained shell snippet; if it needs
                # an interactive heredoc it'll inline the heredoc body
                # in ``cmd`` itself, which bash interprets without
                # touching our pipe.
                input="",
                timeout=cmd_specific_timeout,
                env={
                    "DEBIAN_FRONTEND": "noninteractive",
                    "GIT_TERMINAL_PROMPT": "0",
                },
            )
            output = ""
            if getattr(result, "stderr", ""):
                output = f"{result.stderr}\n"
            return f"{output}{getattr(result, 'stdout', '')}"
        except Exception as exc:
            return f"Error executing bash command: {exc}"

    return ToolDef(
        name="bash",
        description="Execute a bash shell command in the sandbox.",
        parameters={
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "The bash command to execute.",
                }
            },
            "required": ["cmd"],
        },
        handler=execute,
    )


def _python_tool(handle: SandboxHandle, timeout: int | Callable[[], int | None] | None = None) -> ToolDef:
    async def execute(code: str) -> str:
        resolved_timeout = timeout() if callable(timeout) else timeout
        try:
            result = await handle.exec(
                cmd=["python3", "-c", code],
                input="",  # close stdin so input()/sys.stdin reads see EOF
                timeout=resolved_timeout,
            )
            output = ""
            if getattr(result, "stderr", ""):
                output = f"{result.stderr}\n"
            return f"{output}{getattr(result, 'stdout', '')}"
        except Exception as exc:
            return f"Error executing python code: {exc}"

    return ToolDef(
        name="python",
        description="Execute a python script in the sandbox.",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The python code to execute.",
                }
            },
            "required": ["code"],
        },
        handler=execute,
    )


def _submit_solution_tool() -> ToolDef:
    """Special-cased: returns a JSON placeholder. The ARQ on_continue
    swaps the placeholder for the real feedback message once it runs.
    """

    async def execute(summary: str) -> str:
        return json.dumps(
            {
                "attention": ("This is a placeholder that is going to be replaced. If you see this, something went wrong."),
                "dump": summary,
            }
        )

    return ToolDef(
        name="submit_solution",
        description=("Submit your summary of your experiment design rationale, experimental results and findings in plain text as evidence for the prediction."),
        parameters={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": ("The summary of your experimental design idea, execution results and findings."),
                }
            },
            "required": ["summary"],
        },
        handler=execute,
    )


def build_default_tools(
    handle: SandboxHandle,
    *,
    include_python: bool = True,
    include_bash: bool = True,
    include_submit: bool = True,
    timeout: int | Callable[[], int | None] | None = None,
) -> list[ToolDef]:
    """Build the tool set for one rollout.

    Two parameter styles are supported (backward compat + new toolset
    selector):

    - Legacy boolean flags (include_python / include_bash / include_submit):
      used when `toolset="default"` (the historical behavior).
    - `toolset="bash_only"` → bash + submit_solution (no python).
    - `toolset="opencode"` → bash + read/glob/grep/edit/write + submit_solution
      (opencode-style file-editing toolset; implementations in
      tools_opencode.py).

    ``timeout`` applies to every code-executing tool. The
    ``submit_solution`` tool is deliberately *excluded* from this cap so
    that when the wall-time budget is exhausted and ``on_continue``
    forces a final submission, the submit call always has time to run
    (otherwise ``min(tool_timeout, remaining_budget)`` could clamp it to
    0 and crash the final-prediction context).
    """
    tools: list[ToolDef] = []
    if include_python:
        tools.append(_python_tool(handle, timeout=timeout))
    if include_bash:
        tools.append(_bash_tool(handle, timeout=timeout))
    if include_submit:
        tools.append(_submit_solution_tool())
    return tools


# Tool-call argument aliases. Qwen3-coder emits ``bash({"command":...})``
# even when the schema declares ``cmd``. Rewrite client-side so the model
# can use either; same logic as v1's install_tool_arg_alias_patch.
_TOOL_ARG_ALIASES: dict[str, list[tuple[str, str]]] = {
    "bash": [("command", "cmd")],
}


def _maybe_alias_arguments(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    for incoming, canonical in _TOOL_ARG_ALIASES.get(tool_name, []):
        if incoming in args and canonical not in args:
            args = dict(args)
            args[canonical] = args.pop(incoming)
    return args


async def _dispatch_tool_call(
    tool_call: dict[str, Any],
    tools_by_name: dict[str, ToolDef],
) -> tuple[dict[str, Any], float]:
    """Run one tool call. Returns (tool_role_message, elapsed_sec).

    elapsed_sec is the wall duration of the handler invocation —
    used as our sole exec-time accounting (accumulated into
    AgentState.exec_time_accumulated by the caller).
    """
    call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:8]}"
    function = tool_call.get("function") or {}
    name = function.get("name", "")
    raw_args = function.get("arguments", "{}")
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
    except json.JSONDecodeError:
        args = {}
    args = _maybe_alias_arguments(name, args)
    tool = tools_by_name.get(name)
    t0 = time.monotonic()
    if tool is None:
        content = f"Error: unknown tool {name!r}"
    else:
        try:
            sig = inspect.signature(tool.handler)
            kwargs = {k: v for k, v in args.items() if k in sig.parameters}
            content = await tool.handler(**kwargs)
        except Exception as exc:
            content = f"Error invoking tool {name}: {exc}"
    elapsed = time.monotonic() - t0
    return (
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": str(content),
        },
        elapsed,
    )
