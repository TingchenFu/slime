"""ReAct rollout helper for external RL training.

This module is a thin adapter over inspect_ai's tool / sandbox / message
machinery for use in RL training loops where the policy is served by an
external system (e.g. SGLang) and the rollout function — not inspect — drives
generation.

It is intentionally *not* an `Agent` and does not depend on
`inspect_ai.model` providers. The caller owns:

  * calling the LLM (inspect provides no LLM call here),
  * deciding when the rollout is done (no submit-tool detection, no
    `on_continue` loop — return values describe what just happened),
  * preparing the `SandboxEnvironment` used to execute tools.

See `_react.py` for the full ReAct loop this is patterned after.
"""

from contextvars import Token
from dataclasses import dataclass, field
import dataclasses
from collections.abc import Sequence, Iterable, Callable, Optional
from typing import Any
import json
import re
import os
import yaml
import time
import importlib
import logging
import asyncio


from inspect_ai._util.content import Content, ContentReasoning, ContentText
from inspect_ai.agent._types import DEFAULT_ASSISTANT_PROMPT
from inspect_ai.model._call_tools import execute_tools
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from atlas.core.agent import Result
from atlas.agents.llm_agents.react_agent import TOOL_FACTORY
from inspect_ai.model._openai import messages_to_openai, openai_chat_tools
from inspect_ai.model._providers.util.hf_handler import HFHandler
from inspect_ai.model._reasoning import parse_content_with_reasoning
from inspect_ai.tool._tool import Tool, ToolSource, tool
from inspect_ai.tool._tool_def import ToolDef, tool_defs
from inspect_ai.util._sandbox.context import sandbox_environments_context_var
from inspect_ai.util._sandbox.environment import SandboxEnvironment, SandboxEnvironmentSpec
from inspect_ai.util._sandbox.registry import sandboxenv, registry_find_sandboxenv
from atlas.agents.llm_agents.tools.apply_patch import apply_patch
from atlas.agents.llm_agents.tools.basic_tools import (
    bash,
    python,
    restart_stateful_python_session,
    stateful_python,
)
from atlas.utils.dojo_sandbox import DojoSandboxEnvironment, DojoSandboxEnvironmentConfig
from atlas.agents.llm_agents.react_agent import ON_CONTINUE_STR
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from examples.arq_react.arq_react_env import DojoARQSandboxEnvironment
from examples.arq_react.utils import extract_json_data
from .tool import build_default_tools, _dispatch_tool_call
from examples.arq_react.sandbox import boot_sandbox
from examples.arq_react.task import TaskMetadata, load_task_metadata
from agentbox.slurm_allocator import SlurmServiceAllocatorArguments
from agentbox import AgentBoxManager, ContainerConfig


logger = logging.getLogger(__name__)

from .prompts.library import arq_code_generate_execution_prompt, arq_code_generate_execution_prompt_for_tuple, datapoint_to_final_prediction_prompt, datapoint_to_final_prediction_prompt_for_tuple, datapoint_to_feedback_prompt_for_tuple, datapoint_to_feedback_prompt


PROMPT_FUNCTION_MAP = {
    "arq_code_generate_execution_prompt": arq_code_generate_execution_prompt,
    "arq_code_generate_execution_prompt_for_tuple": arq_code_generate_execution_prompt_for_tuple,
    "datapoint_to_final_prediction_prompt": datapoint_to_final_prediction_prompt,
    "datapoint_to_final_prediction_prompt_for_tuple": datapoint_to_final_prediction_prompt_for_tuple,
    "datapoint_to_feedback_prompt_for_tuple": datapoint_to_feedback_prompt_for_tuple,
    "datapoint_to_feedback_prompt": datapoint_to_feedback_prompt,
}


# @dataclass
# class AssistantMessage:
#     """View of the last assistant message we received. Mirrors the
#     ``ChatMessageAssistant`` attributes ARQ reads.
#     """

#     content: str = ""
#     tool_calls: list[dict[str, Any]] | None = None
#     raw: dict[str, Any] = field(default_factory=dict)


# @dataclass
# class ModelOutput:
#     """Mirrors ``inspect_ai.model.ModelOutput`` to the degree ARQ reads it."""

#     message: AssistantMessage = field(default_factory=AssistantMessage)
#     stop_reason: str | None = None
#     time: float = 0.0


# @dataclass
# class AgentState:
#     messages: list[dict[str, Any]] = field(default_factory=list)
#     output: ModelOutput | None = None
#     exec_time_accumulated: float = 0.0


# @dataclass
# class ToolExecution:
#     """One tool call and its observable result."""

#     call_id: str
#     function: str
#     arguments: dict[str, Any]
#     result_text: str
#     error: str | None = None
#     parse_error: str | None = None


# @dataclass
# class StepResult:
#     """Outcome of advancing the rollout by one assistant turn."""

#     next_inspect_messages: list[ChatMessage]
#     tool_executions: list[ToolExecution] = field(default_factory=list)
#     """Non-fatal warning when the raw text looks broken (unclosed <tool_call>,
#     too many tool calls, suspiciously long output). The message has still been
#     appended and any successfully-parsed tool calls have still been executed;
#     this just flags the situation so the caller's on_continue can decide
#     whether to terminate, penalise, or push back."""
#     parse_warning: str | None = None
#     execution_time: float = 0.0


async def inspect_initial_messages(
    *,
    user_prompt: str,
    instructions: str | None = None,
    assistant_prompt: str | None = DEFAULT_ASSISTANT_PROMPT,
) -> list[dict[str, Any]]:
    """Build the starting messages for a ReAct rollout.

    Returns (openai_format, inspect_format). The caller feeds openai_format
    into SGLang and keeps inspect_format around to pass back to `step()`.
    """
    prompt_lines = [p for p in (instructions, assistant_prompt) if p]
    system_text = "\n\n".join(prompt_lines).strip()

    inspect_messages: list[ChatMessage] = []
    if system_text:
        inspect_messages.append(ChatMessageSystem(content=system_text))
    inspect_messages.append(ChatMessageUser(content=user_prompt))

    openai_messages = await messages_to_openai(inspect_messages)
    return [dict(m) for m in openai_messages]


async def message_parsing(
    *,
    assistant_raw: str,
    tools: Sequence[Tool | ToolDef | ToolSource],
    model_name: str = "qwen-instruct",
    max_tool_calls_per_turn: int = 4,
    max_assistant_words: int = 2048,
) -> list[dict[str, Any]]:
    """Parsing assistant raw -> openai-format assistant message"""

    def _tool_def_to_info(td: ToolDef) -> Any:
        from inspect_ai.tool._tool_info import ToolInfo

        return ToolInfo(
            name=td.name,
            description=td.description,
            parameters=td.parameters,
            options=td.options,
        )

    def _has_unclosed_tool_call(raw: str) -> bool:
        """True iff `raw` has more `<tool_call>` opens than `</tool_call>` closes."""
        opens = raw.count("<tool_call>")
        closes = raw.count("</tool_call>")
        return opens > closes

    # resolve tools to ToolInfo for the parser, and to ToolDef list for execute_tools
    tdefs = await tool_defs(list(tools))
    tool_infos = [_tool_def_to_info(td) for td in tdefs]

    # parse raw text into a ChatMessageAssistant (Qwen <tool_call> handled here)
    assistant = HFHandler(model_name).parse_assistant_response(assistant_raw, tool_infos)

    warnings: list[str] = []

    # case1:
    # detect unclosed <tool_call> — xml_extract only matches well-formed pairs,
    # so an unclosed opener would silently look like "no tool call"
    if assistant.tool_calls is None and _has_unclosed_tool_call(assistant_raw):
        warnings.append("assistant raw text contains an unclosed <tool_call> tag (likely truncated by max_tokens)")

    # case2:
    # detect well-formed but unparseable <tool_call> tags. HFHandler's pre-filter
    # requires the literal words "name" and "arguments" to be present before it
    # tries to parse, so a tag like `<tool_call>garbage</tool_call>` is silently
    # treated as prose. For RL we want to surface this so the model can learn.
    if assistant.tool_calls is None and "<tool_call>" in assistant_raw and "</tool_call>" in assistant_raw:
        warnings.append("assistant emitted <tool_call> tags whose content is not a valid name+arguments JSON object")

    # case3:
    # cap parallel tool calls — collapsed models will spam the same call
    if assistant.tool_calls and len(assistant.tool_calls) > max_tool_calls_per_turn:
        warnings.append(f"assistant emitted {len(assistant.tool_calls)} tool calls; keeping only the first {max_tool_calls_per_turn}")
        assistant = assistant.model_copy(update={"tool_calls": assistant.tool_calls[:max_tool_calls_per_turn]})

    # case4:
    # flag suspiciously long outputs (don't truncate — RL needs the full token
    # sequence for logprob computation outside this module)
    if len(assistant_raw.split(" ")) > max_assistant_words:
        warnings.append(f"assistant raw text is {len(assistant_raw.split(' '))} chars (> {max_assistant_words}); possible repetition collapse")

    # split <think>...</think> out into a ContentReasoning, matching the
    # OpenAI provider's behaviour (`_openai.py:710`).
    if isinstance(assistant.content, str):
        remaining, capsule = parse_content_with_reasoning(assistant.content)
        if capsule is not None:
            reasoning = ContentReasoning(
                reasoning=capsule.reasoning,
                signature=capsule.signature,
                redacted=capsule.redacted,
                internal=capsule.internal,
            )
            content_list: list[Content] = [reasoning]
            if remaining:
                content_list.append(ContentText(text=remaining))
            assistant = assistant.model_copy(update={"content": content_list})

    openai_assistant = await messages_to_openai(assistant)
    return [dict(x) for x in openai_assistant]


@tool(name="submit_solution")
def _create_custom_submit_tool() -> Tool:
    """
    Submit tool is a special tool in our case. The tool abstraction in
    inspect-ai gets only the tool call arguments as input. For us it's
    not enough, we need the whole agent state to be able to
    construct the final result, including the trajectory, logs, and
    other metadata.

    For that, the logic is to reply with a dump of the submission, then
    the `on_continue` callable (which, unlike tools, gets the full agent state
    and can modify it) will construct the full Result object, include the
    needed metadata, and hand over to the parent agent. The feedback will
    manually replace the last message. For non submit tools `on_continue`
    will behave as usual.
    """

    async def execute(summary: str) -> ContentText:
        """Submit your summary of your experiment design rationale, experimental results and findings in plain text as evidence for the prediction.

        Args:
            summary (str): The summary of your experimental design idea, execution results and findings.
        """

        # we're missing so much information here, so we'll just
        # dump what we have and let the on_continue handle the rest
        json_str = json.dumps(
            {
                "attention": "This is a placeholder that is going to be replaced. If you see this, something went wrong.",
                "dump": summary,
            }
        )

        # instead of returning a string, we return ContentText so inspect ai will not
        # do extra modifications such as truncation, formatting, etc.
        return ContentText(text=json_str)

    return execute


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Custom generation function supporting tool calls"""
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    # Retried samples (previously aborted / partial) arrive here with stale
    # rollout state from the first attempt. Clear it so this generation starts
    # clean; otherwise the concatenation below appends new tokens to old ones
    # and downstream `slice_log_prob_with_cp` sees a length mismatch.
    sample.rollout_log_probs = None
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = None

    state = GenerateState(args)
    agent_id = f"rollout{sample.metadata['current_rollout_id']}_example{sample.metadata['example_id']}"
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    datapoint = None
    for i, line in enumerate(open(sample.metadata["data_filepath"]).readlines()):
        if i == int(sample.metadata["example_id"]):
            datapoint = json.loads(line)
            break
    assert datapoint is not None

    # openai_tools = await openai_tools_schema(tools=inspect_tools)
    # initial_prompt = state.tokenizer.apply_chat_template(conversation=initial_openai_messages, tools=openai_tools, tokenize=False)
    # prompt = initial_prompt
    # prompt_tokens_ids = state.tokenizer(initial_prompt, add_special_tokens=False)["input_ids"]
    # inspect_messages = initial_inspect_messages
    # openai_messages = initial_openai_messages

    response = ""
    response_token_ids = []
    loss_masks = []
    tool_call_count = 0  # Track actual tool call rounds
    exe_time = 0
    pending_experiment = None
    has_unsubmitted_result = None
    is_last_submit = False
    intermediate_results: list[Result] = []  # to store all valid submission result.

    if args.rollout_max_context_len is not None:
        max_context_length = args.rollout_max_context_len
    else:
        max_context_length = args.context_parallel_size * args.max_tokens_per_gpu

    def _clamped_tool_timeout() -> int | None:
        """`min(tool_timeout, remaining_budget)` per tool call.

        When clamping is disabled (config opt-out), behaves
        like upstream ARQ: returns the static tool_timeout,
        no budget awareness. Submit tool is exempt regardless
        — see build_default_tools docstring.
        """
        remaining = max(
            1.0,
            float(args.exe_timeout) - exe_time,
        )
        return int(min(args.exe_timeout, remaining))

    container_config = ContainerConfig(
        image_handle=args.container_image,
        working_dir=args.working_dir,
        bind_inputs_dir=os.path.join(args.env_cache_dir, sample.metadata["task_name"]),
        container_runtime="apptainer",
        env={
            # "HF_HUB_OFFLINE": "1",
            "HF_ENDPOINT": "https://hf-mirror.com",
            "NLTK_DATA": "/root/.nltk_data",
            "WANDB_DISABLED": "true",
        },
        read_only_overlays=[
            "/engshare/maui_dojo/shared/agent/2025-12-30.cache.overlay.ext3.img",
            # "/engshare/maui_dojo/shared/agent/patch-gpu-memory-limit.overlay.ext3.img",
        ],
    )
    task_meta = load_task_metadata(args.airsbench_tasks_root, sample.metadata["task_name"])
    rpc_manager = AgentBoxManager(args.remote_manager_uri)
    DojoARQSandboxEnvironment.set_manager(rpc_manager)

    sandbox_config = DojoSandboxEnvironmentConfig(
        machine=None,
        container_config=container_config,
    )

    handle = await boot_sandbox(
        task_name=str(agent_id),
        sandbox_config=sandbox_config,
        metadata=dataclasses.asdict(task_meta) if dataclasses.is_dataclass(task_meta) else (task_meta or {}),
        sandbox_cls=DojoARQSandboxEnvironment,
    )

    tools = build_default_tools(
        handle,
        include_python=True,
        include_bash=True,
        include_submit=True,
        timeout=_clamped_tool_timeout,
    )
    tools_list = list(tools)
    tools_by_name = {t.name: t for t in tools_list}

    custom_system_prompt = "You are an advanced AI agent that uses the react framework to solve tasks by reasoning and using tools as needed."
    user_prompt, _ = PROMPT_FUNCTION_MAP[args.template_function](datapoint, args.template_version, args.device, args.prompt_timeout)
    messages = await inspect_initial_messages(user_prompt=user_prompt, instructions=custom_system_prompt)
    prompt = state.tokenizer.apply_chat_template(conversation=messages, tools=[t.openai_schema() for t in tools], tokenize=False)
    # state = AgentState(messages=messages)
    prompt_token_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    # await run_react_loop(
    #     state,
    #     tools,
    #     on_continue=self._on_continue,
    #     max_iterations=None,
    #     temperature=None,
    # )

    # _slurm_env_keys = [k for k in os.environ if k.startswith("SLURM")]
    # _saved_slurm_env = {k: os.environ.pop(k) for k in _slurm_env_keys}
    # slurm_args = SlurmServiceAllocatorArguments(
    #     name=f"arq_react_rl_{agent_id}",  # TODO
    #     timeout_min=args.slurm_time_min,
    #     cpus_per_node=args.slurm_cpus_per_node,
    #     gpus_per_node=args.slurm_gpus_per_node,
    #     account=args.slurm_account,
    #     qos=args.slurm_qos,
    #     mem=args.slurm_mem,
    #     partition=args.slurm_partition,
    #     use_arrays=args.slurm_use_arrays,
    #     array_max_parallelism=2,
    # )
    # allocator = AgentBoxManager.create_slurm_cluster(
    #     num_workers=1,
    #     slurm_args=slurm_args,
    #     worker_startup_timeout=args.worker_startup_timeout,
    #     allocate_manager_remotely=False,
    # )
    # rpc_manager = allocator.manager
    # os.environ.update(_saved_slurm_env)

    async def generate_plan_feedback(remaining_prompt_time: float | int, prev_estimations: list[str | None]):
        # template_module = importlib.import_module("examples.arq_react.feedback_template")
        # template = getattr(template_module, args.feedback_template)
        prompt = datapoint_to_feedback_prompt(datapoint, args.feedback_template_version)  # PROMPT_FUNCTION_MAP[args.feedback_template_function]

        prompt_token_ids = state.tokenizer.apply_chat_template(conversation=[{"role": "system", "content": prompt}], tokenize=True)
        payload = {
            "input_ids": prompt_token_ids,
            "sampling_params": dict(sampling_params),
            "return_logprob": False,  # Not used for training
        }
        plan_response = None

        output = await post(url, payload)

        if output["meta_info"]["finish_reason"]["type"] == "abort":
            logger.info("plan response abort")
            return None

        try:
            return extract_json_data(plan_response)
        except Exception as e:
            logger.info(f"Exception {e} when extracting json format data from plan response {plan_response}")
            return None

    async def on_continue():
        nonlocal messages
        nonlocal is_last_submit
        nonlocal intermediate_results
        nonlocal pending_experiment
        nonlocal has_unsubmitted_result

        def is_assistant(m):
            return m.get("role") == "assistant"

        def is_tool(m):
            return m.get("role") == "tool"

        def tool_name(m):
            return m.get("name")

        last_assistant_idx = max([i for i in range(len(messages)) if is_assistant(messages[i])])
        last_submission_idx = max([i for i in range(len(messages)) if is_tool(messages[i]) and tool_name(messages[i]) == "submit_solution"], default=0)
        num_submit_messages = sum([is_tool(msg) and tool_name(msg) == "submit_solution" for msg in messages])

        for i in range(last_assistant_idx, len(messages)):
            if is_tool(messages[i]) and tool_name(messages[i]) == "python":
                pending_experiment = None
        for i in range(last_submission_idx, len(messages)):
            if is_tool(messages[i]) and tool_name(messages[i]) == "python":
                has_unsubmitted_result = True

        if exe_time > args.exe_timeout and not is_last_submit:
            if has_unsubmitted_result and args.allow_last_submit:
                # case: last submit
                is_last_submit = True
                feedback = "TIMEOUT. You have run out of time. Please immediately submit whatever experimental findings and numeric results you have gathered so far."
                return feedback  # TODO: make decison based on the actual content of feedback text.
            # case: time up with no last submit
            return False

        if is_tool(messages[-1]) and tool_name(messages[-1]) == "submit_solution":
            # case: submit with pending pilot experiment
            if pending_experiment is not None and not args.is_last_submit:
                feedback = f"INVALID submit_solution() call. Before calling submit_solution(), you MUST use python() tool to run the pending pilot experiment: {pending_experiment['title']}\nImplementation Steps:\n{pending_experiment['steps']}\n"
                messages[-1].content = feedback
                return True

            # case: submit without pending pilot experiment
            dump = json.loads(messages[-1].text)["dump"]
            agent_result = Result(
                output={"response": dump},
                success=True,
                author=agent_id,
                metadata={"agent_trajectory": [msg.model_dump() for msg in messages], "exec_time_accumulated": exe_time},
            )
            intermediate_results.append(agent_result)

            feedback_raw = await generate_plan_feedback(args.prompt_timeout - exe_time * (args.prompt_timeout / args.exe_timeout), [ar.output["response"] for ar in intermediate_results])

            if feedback_raw is None:
                logger.info(f"Agent {agent_id} received no feedback for submission, stopping further actions.")
                return False
            logger.info(f"Agent {agent_id} received feedback for submission: {feedback_raw}")
            feedback_dict = json.loads(feedback_raw)
            if feedback_dict["rationale"] == "stop":
                return False
            if feedback_dict["next_experiment"] is not None:
                pending_experiment = feedback_dict["next_experiment"]
                feedback = (
                    f"Submission recorded. You still have over {feedback_dict['remaining_time']} seconds of GPU access remaining.\n"
                    f"DO NOT call `submit_solution()` again immediately. You MUST use python() tool to run next pilot experiment: {feedback_dict['next_experiment']['title']}\n"
                    f"Implementation Steps:\n"
                    f"{feedback_dict['next_experiment']['steps']}\n"
                )
            else:
                feedback = (
                    f"Submission recorded. You still have over {feedback_dict['remaining_time']} seconds of GPU access remaining.\n"
                    f"DO NOT call `submit_solution()` again immediately. You MUST run at least one new experiment using `python()` before your next submission. For example:\n"
                    f"- Try a different model or algorithm\n"
                    f"- Adjust hyperparameters (learning rate, regularization, number of folds, etc.)\n"
                    f"- Engineer new features or apply different preprocessing\n"
                    f"- Run the same approach with a different random seed to check stability\n"
                    f"Only call `submit_solution()` again after you have obtained new experimental evidence that either confirms or changes your current answer. Submitting the same answer without new evidence is wasteful and not allowed.\n"
                )
            messages[-1].content = feedback

            # case: last submit
            if is_last_submit:
                return False

            if num_submit_messages >= args.num_attempts:
                logger.info(f"Agent {agent_id} reached max submissions ({args.num_attempts}), stopping further actions.")
                return False

        if is_last_submit:
            # case: last_submit but din't call submit_solution()
            logger.warning(f"agent {agent_id} fails to call submit_solution() at last submission")
            return False

        if not state.output.message.tool_calls:
            # case: pure text with pending experiment
            if pending_experiment:
                feedback = f"Remember that you MUST use python() tool to run the pilot experiment: {pending_experiment['title']}\nImplementation Steps:\n{pending_experiment['steps']}\n"
                return feedback
            else:
                # case: pure text without pending experiment
                return ON_CONTINUE_STR

        return True

    # try:

    # finally:TODO: add finally logic to cleanup
    #     await environment_cls.sample_cleanup(
    #         task_name="agent_x", config=spec.config,
    #         environments=environments, interrupted=False,
    #     )
    for turn in range(args.react_max_turns):
        # trun = model message + (one or more) tool calls

        # Check if total length exceeds max context length
        total_length = len(prompt_token_ids) + len(response_token_ids)
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break

        # Clamp per-turn max_new_tokens to the remaining context budget so a
        # single turn cannot push total_length past max_context_length. Without
        # this, a turn can append up to rollout_max_response_len tokens on top
        # of a total that was just barely under the cap, producing samples
        # that exceed the training-side max_tokens_per_gpu * cp_size budget
        # and crash the partition/batch code (asserts or OOMs on an oversized
        # partition).
        remaining_budget = max_context_length - total_length
        per_turn_sampling_params = dict(sampling_params)
        per_turn_sampling_params["max_new_tokens"] = min(
            sampling_params.get("max_new_tokens", remaining_budget),
            remaining_budget,
        )

        # Use token IDs instead of text
        current_token_ids = prompt_token_ids + response_token_ids
        payload = {
            "input_ids": current_token_ids,
            "sampling_params": per_turn_sampling_params,
            "return_logprob": True,  # Request log probabilities for training
        }

        # Log payload to wandb for debugging
        try:
            import wandb

            if wandb.run is not None:
                # Count available tools (from tool_specs)
                available_tools = len(tools)
                # Count tools used in the current response

                wandb.log(
                    {
                        "debug/payload_length": len(prompt.split(" ") + response.split(" ")),
                        "debug/available_tools": available_tools,
                        "debug/tools_used": tool_call_count,
                        "debug/turn": turn,
                    }
                )
        except ImportError:
            pass  # wandb not available

        output = await post(url, payload)

        # Handle abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        if "output_token_logprobs" in output["meta_info"]:
            cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response = state.tokenizer.decode(cur_response_token_ids)
            cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            if sample.rollout_log_probs is None:
                sample.rollout_log_probs = []
            sample.rollout_log_probs += cur_log_probs

        else:
            # sglang returned text but no output_token_logprobs — we cannot
            # recover per-token logprobs for this turn, which would desync
            # rollout_log_probs from response_token_ids and blow up
            # `slice_log_prob_with_cp` downstream. Abort the sample so the
            # fully_async rollout manager returns the whole group to the
            # buffer for retry instead of poisoning the trainer.
            sample.status = Sample.Status.ABORTED
            return sample

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)

        # Check length limit
        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        cur_message = await message_parsing(assistant_raw=cur_response, tools=tools)
        messages.append(cur_message)

        tool_calls = cur_message.get("tool_calls", [])
        num_message_before_tool = len(messages)  # number of messages before tool execution
        if tool_calls:
            tool_results = await asyncio.gather(*(_dispatch_tool_call(tc, tools_by_name) for tc in tool_calls))
            for msg, elapsed in tool_results:
                messages.append(msg)
                exe_time += elapsed

        to_continue = await on_continue()

        if num_message_before_tool < len(messages):
            for msg in messages[num_message_before_tool:]:
                tool_str = state.tokenizer.apply_chat_template(msg, tokenize=False)
                tool_token_ids = state.tokenizer(tool_str, add_special_tokens=False)["input_ids"]
                response += tool_str
                response_token_ids += tool_token_ids
                loss_masks += [0] * len(tool_token_ids)
                sample.rollout_log_probs += [0.0] * len(tool_token_ids)

        if isinstance(to_continue, str):
            messages.append({"role": "user", "content": to_continue})
            user_str = state.tokenizer.apply_chat_template(messages[-1], tokenize=False)
            user_token_ids = state.tokenizer(user_str, add_special_tokens=False)["input_ids"]
            response += user_str
            response_token_ids += user_token_ids
            loss_masks += [0] * len(user_token_ids)
            sample.rollout_log_probs += [0.0] * len(user_token_ids)

        elif isinstance(to_continue, bool) and not to_continue:
            break

        # Tool output is appended verbatim and can push total_length past
        # max_context_length (the per-turn generation was clamped to the
        # remaining budget, but tool output is unconstrained). Trim tail
        # tokens so the final sample fits the training budget exactly.
        overflow = len(prompt_token_ids) + len(response_token_ids) - max_context_length
        if overflow > 0:
            response_token_ids = response_token_ids[:-overflow]
            loss_masks = loss_masks[:-overflow]
            if sample.rollout_log_probs is not None:
                sample.rollout_log_probs = sample.rollout_log_probs[:-overflow]
            # Resync the text field from the trimmed token list so
            # reward_func's `sample.prompt + sample.response` matches what
            # the model was actually trained on. decode(tokenize(text)) can
            # be lossy on some tokenizers (whitespace / special-token
            # collapse), but reward_func's regex is whitespace-robust and
            # the trainer sees tokens, not text — so the drift is safe.
            response = state.tokenizer.decode(response_token_ids)
            sample.status = Sample.Status.TRUNCATED
            break

    # Set sample attributes
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks

    # Store payload information for wandb logging
    sample.payload_text = prompt + response
    sample.payload_has_system = "<|im_start|>system" in prompt + response
    sample.payload_has_tools = "# Tools" in prompt + response

    # Store tool call count for reward calculation
    sample.tool_call_count = tool_call_count

    # Set status
    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


# async def run_react_loop(
#     state: AgentState,
#     tools: Iterable[ToolDef],
#     *,
#     on_continue: Callable | None,
#     max_iterations: int | None = None,
#     temperature: float | None = None,
# ) -> AgentState:
#     """ARQ-shaped ReAct loop.

#     Per iteration:
#       1. Generate (with prompt_too_long / model_length compaction)
#       2. If tool_calls: dispatch each, append tool result messages
#       3. Call on_continue(state); use its return value:
#          - False → stop
#          - True → continue (if no tool_calls, append default user msg)
#          - str → continue, append as user message
#          - AgentState → replace state, continue
#     """
#     tools_list = list(tools)
#     tools_by_name = {t.name: t for t in tools_list}

#     iteration = 0
#     while max_iterations is None or iteration < max_iterations:
#         iteration += 1

#         output = await _generate_with_recovery(state, tools_list, temperature=temperature)
#         state.output = output

#         tool_calls = output.message.tool_calls or []
#         if tool_calls:
#             tool_results = await asyncio.gather(*(_dispatch_tool_call(tc, tools_by_name) for tc in tool_calls))
#             for msg, elapsed in tool_results:
#                 state.messages.append(msg)
#                 state.exec_time_accumulated += elapsed

#         if on_continue is not None:
#             decision = await on_continue(state)
#             if decision is False:
#                 break
#             if isinstance(decision, AgentState):
#                 state = decision
#                 continue
#             if isinstance(decision, str):
#                 state.messages.append({"role": "user", "content": decision})
#                 continue
#             # decision is True (or None): continue. If no tool_calls were
#             # made this turn, add a default user prompt so the model knows
#             # to act.
#             if not tool_calls:
#                 state.messages.append({"role": "user", "content": DEFAULT_CONTINUE_PROMPT})
#             continue

#         # No on_continue: stop after first tool-less turn.
#         if not tool_calls:
#             break

#     return state
