import logging
import os
from pathlib import Path
from typing import Optional

from .task import AIRSBENCH_LITE_TASKS, AIRSBENCH_MM_TASKS, WMRL_TRAIN_TASKS

logger = logging.getLogger(__name__)


GENERIC_RANKING_PREAMBLE = """An AI researcher is attempting to solve a machine learning problem.
Here are two different solutions that they produced at different times in the research process.
Both solutions should train on a train set and ultimately produce an artifact for testing.
For example, a submission.csv of predictions on the test set, or a set of model weights that would be tested outside of this script.
Ideally, both solutions also produce a validation metric - e.g by splitting some data off the training set to use as a validation set.
This validation metric is then used by the researcher to decide which solution is more promising, and either submit the artifact for testing,
or iterate further on the solution.
\n
"""

GENERIC_CLASSIFICATION_PREAMBLE = """An AI researcher is attempting to solve a machine learning problem.
Here is a solution that they produced during the research process.
The solution should train on a train set and ultimately produce an artifact for testing.
For example, a submission.csv of predictions on the test set, or a set of model weights that would be tested outside of this script.
\n
"""

prompt_library = {
    "ranking": {
        "validation_metric": {
            "prefix": GENERIC_RANKING_PREAMBLE,
            "suffix": "Which of the two solutions will yield a better validation metric? Think step by step and provide your reasoning before giving a final answer of A for option A, and B for the option B, inside a \\boxed{}, ie \\boxed{A} or \\boxed{B}.",
        },
        "test_metric": {
            "prefix": GENERIC_RANKING_PREAMBLE,
            "suffix": "Which of the two solutions will yield better performance when it is tested on the test set? Think step by step and provide your reasoning before giving a final answer of A for option A, and B for the option B, inside a \\boxed{}, ie \\boxed{A} or \\boxed{B}.",
        },
        "exec_time": {
            "prefix": GENERIC_RANKING_PREAMBLE,
            "suffix": "Which of the two solutions will execute faster? Think step by step and provide your reasoning before giving a final answer of A for option A, and B for the option B, inside a \\boxed{}, ie \\boxed{A} or \\boxed{B}.",
        },
    },
    "classification": {
        "is_buggy": {
            "prefix": GENERIC_CLASSIFICATION_PREAMBLE,
            "suffix": "Will this solution crash, produce an error, or fail to execute successfully? Think step by step and provide your reasoning before giving a final answer. Answer True if the solution is buggy (will crash or error), or False if it will execute successfully. Provide your answer inside a \\boxed{}, ie \\boxed{True} or \\boxed{False}.",
        }
    },
}


task2desc: dict[str, str] = {}

exp_root_dir = Path("/home/tingchenfu/airs-bench/airsbench/tasks/rad")
for task_name in AIRSBENCH_LITE_TASKS:
    exp_dir = exp_root_dir / task_name / "project_description.md"
    task2desc[task_name] = exp_dir.read_text()
exp_root_dir = Path("/home/tingchenfu/aira-bench-data/multimodal_tasks/rad/final_tasks")
for task_name in AIRSBENCH_MM_TASKS:
    exp_dir = exp_root_dir / task_name / "project_description.md"
    task2desc[task_name] = exp_dir.read_text()
exp_root_dir = Path("/home/tingchenfu/aira-bench-data/multimodal_tasks/rad/final_tasks")
for task_name in WMRL_TRAIN_TASKS:
    exp_dir = exp_root_dir / task_name / "project_description.md"
    task2desc[task_name] = exp_dir.read_text()

from .templates import (
    AGENT_TEMPLATE_V1,
    AGENT_TEMPLATE_V2,
    AGENT_TEMPLATE_V3,
    AGENT_TEMPLATE_V4,
    AGENT_TEMPLATE_V5,
    AGENT_TEMPLATE_V6,
    CC_TEMPLATE_V1,
    CC_TEMPLATE_V2,
    CODEGEN_TEMPLATE_V1,
    CODEGEN_TEMPLATE_V2,
    CODEGEN_TEMPLATE_V3,
    CODEGEN_TEMPLATE_V4,
    CODEGEN_TEMPLATE_V5,
    CODEGEN_TEMPLATE_V6,
    CODEGEN_TEMPLATE_V7,
    CODEGEN_TEMPLATE_V8,
    CODEGEN_TEMPLATE_V9,
    CODEGEN_TEMPLATE_V10,
    CODEGEN_TEMPLATE_V11,
    FINAL_PREDICTION_TEMPLATE_V1,
    FINAL_PREDICTION_TEMPLATE_V2,
    FINAL_PREDICTION_TEMPLATE_V3,
    FINAL_PREDICTION_TEMPLATE_V4,
    FINAL_PREDICTION_TEMPLATE_V5,
    ICL_TEMPLATE_V1,
    ICL_TEMPLATE_V2,
    REACT_FEEDBACK_TEMPLATE_V1,
    REACT_FEEDBACK_TEMPLATE_V2,
    REACT_TEMPLATE_V1,
    REACT_TEMPLATE_V2,
    REACT_TEMPLATE_V3,
    RULEBOOK_TEMPLATE_V1,
    RULEBOOK_TEMPLATE_V2,
)

version2template = {
    "v1": AGENT_TEMPLATE_V1,
    "v2": AGENT_TEMPLATE_V2,
    "v3": AGENT_TEMPLATE_V3,
    "v4": AGENT_TEMPLATE_V4,
    "v5": AGENT_TEMPLATE_V5,
    "v6": AGENT_TEMPLATE_V6,
    "icl_v1": ICL_TEMPLATE_V1,
    "icl_v2": ICL_TEMPLATE_V2,
    "final_prediction_v1": FINAL_PREDICTION_TEMPLATE_V1,
    "final_prediction_v2": FINAL_PREDICTION_TEMPLATE_V2,
    "final_prediction_v3": FINAL_PREDICTION_TEMPLATE_V3,
    "final_prediction_v4": FINAL_PREDICTION_TEMPLATE_V4,
    "final_prediction_v5": FINAL_PREDICTION_TEMPLATE_V5,
    "codegen_v1": CODEGEN_TEMPLATE_V1,
    "codegen_v2": CODEGEN_TEMPLATE_V2,
    "codegen_v3": CODEGEN_TEMPLATE_V3,
    "codegen_v4": CODEGEN_TEMPLATE_V4,
    "codegen_v5": CODEGEN_TEMPLATE_V5,
    "codegen_v6": CODEGEN_TEMPLATE_V6,
    "codegen_v7": CODEGEN_TEMPLATE_V7,
    "codegen_v8": CODEGEN_TEMPLATE_V8,
    "codegen_v9": CODEGEN_TEMPLATE_V9,
    "codegen_v10": CODEGEN_TEMPLATE_V10,
    "rulebook_v1": RULEBOOK_TEMPLATE_V1,
    "rulebook_v2": RULEBOOK_TEMPLATE_V2,
    "codegen_v11": CODEGEN_TEMPLATE_V11,
    "react_v1": REACT_TEMPLATE_V1,
    "react_v2": REACT_TEMPLATE_V2,
    "react_v3": REACT_TEMPLATE_V3,
    "react_feedback_v1": REACT_FEEDBACK_TEMPLATE_V1,
    "react_feedback_v2": REACT_FEEDBACK_TEMPLATE_V2,
    "cc_v1": CC_TEMPLATE_V1,
    "cc_v2": CC_TEMPLATE_V2,
}


def second2timestr(time_in_second: int | float) -> str:
    time_in_second: int = round(time_in_second)
    hour = time_in_second // 3600
    minute = time_in_second % 3600 // 60
    second: int = time_in_second % 3600 % 60

    parts = []
    if hour:
        parts.append(f"{hour} hour{'s' if hour > 1 else ''}")
    if minute:
        parts.append(f"{minute} minute{'s' if minute > 1 else ''}")
    if second or not parts:
        parts.append(f"{second} second{'s' if second != 1 else ''}")
    return " and ".join(parts)


import random


def datapoint_to_prompt(
    label_type,
    label_name,
    datapoint,
    do_shuffle=True,
    limit_context_nodes=None,
    only_use_validation_metric_in_context=True,
):
    if label_type == "ranking":
        # Handle ranking tasks (comparing two solutions)
        if do_shuffle:
            random_bit = random.randint(0, 1)
        else:
            random_bit = 0
        if datapoint.get("candidates", None) is not None:
            slot_a = datapoint["candidates"][0] if random_bit == 0 else datapoint["candidates"][1]
            slot_b = datapoint["candidates"][1] if random_bit == 0 else datapoint["candidates"][0]
        else:
            slot_a = datapoint["node_a"] if random_bit == 0 else datapoint["node_b"]
            slot_b = datapoint["node_b"] if random_bit == 0 else datapoint["node_a"]
        choice = datapoint["choice"] if random_bit == 0 else (1 - datapoint["choice"])

        prompt = prompt_library[label_type][label_name]["prefix"]
        prompt += "Option A:\n```python\n"
        prompt += slot_a["code"] + "```\n\n"
        prompt += "Option B:\n```python\n"
        prompt += slot_b["code"] + "```\n\n"

        # add in the context nodes
        context_nodes = datapoint["context_nodes"]
        if limit_context_nodes is not None:
            context_nodes = context_nodes[:limit_context_nodes]
            logger.debug(f"Limiting context nodes to {limit_context_nodes} for this prompt. Original number of context nodes: {len(datapoint['context_nodes'])}.")
        for i, context_node in enumerate(context_nodes):
            if only_use_validation_metric_in_context and label_name == "test_metric":
                prompt += f"For potentially useful context, here is another solution that had validation metric: {context_node['validation_metric']}\n\n```python\n"
            else:
                prompt += f"For potentially useful context, here is another solution that had {label_name.replace('_', ' ')}: {context_node['label']}\n\n```python\n"
            prompt += context_node["code"] + "```\n\n"

        prompt += prompt_library[label_type][label_name]["suffix"]

        return prompt, choice

    elif label_type == "classification":
        # Handle classification tasks (single solution)
        node = datapoint["node"]
        label = node["label"]  # Boolean for is_buggy

        prompt = prompt_library[label_type][label_name]["prefix"]
        prompt += "Solution:\n```python\n"
        prompt += node["code"] + "```\n\n"

        # add in the context nodes
        context_nodes = datapoint["context_nodes"]
        if limit_context_nodes is not None:
            context_nodes = context_nodes[:limit_context_nodes]
            logger.debug(f"Limiting context nodes to {limit_context_nodes} for this prompt. Original number of context nodes: {len(datapoint['context_nodes'])}.")
        for i, context_node in enumerate(context_nodes):
            prompt += f"For potentially useful context, here is another solution that was {'buggy' if context_node['label'] else 'not buggy'}:\n\n```python\n"
            prompt += context_node["code"] + "```\n\n"

        prompt += prompt_library[label_type][label_name]["suffix"]

        return prompt, label

    else:
        raise ValueError(f"Unknown label_type: {label_type}. Supported types: 'ranking', 'classification'")


def datapoint_to_agentbox_prompt(
    label_type: str,
    label_name: str,
    datapoint: dict,
    do_shuffle: bool = False,
    version: str = "v1",
    device: str = "CPU only (No GPU)",
    time_limit: int | float = 7200,
    limit_context_nodes: Optional[int] = 0,
    feedbacks: list[tuple[str, str]] = [],
    rules: list[str] = [],
) -> tuple[str, int]:
    """Convert datapoint to prompt for code generation in AgentBox evaluation."""
    assert label_type == "ranking"

    if do_shuffle:
        random_bit = 1  # random.randint(0, 1)
    else:
        random_bit = 0

    if datapoint.get("candidates", None) is not None:
        slot_a = datapoint["candidates"][0] if random_bit == 0 else datapoint["candidates"][1]
        slot_b = datapoint["candidates"][1] if random_bit == 0 else datapoint["candidates"][0]
    else:
        slot_a = datapoint["node_a"] if random_bit == 0 else datapoint["node_b"]
        slot_b = datapoint["node_b"] if random_bit == 0 else datapoint["node_a"]

    choice = datapoint["choice"] if random_bit == 0 else (1 - datapoint["choice"])

    # Get task name for data location hints
    task_name = slot_a.get("task", "unknown")

    example_str = []
    for i in range(min(limit_context_nodes, len(datapoint["context_nodes"]))):
        text = f"""
**Reference Solution {i + 1}:**
{datapoint["context_nodes"][i]["code"].strip()}

**5-fold CV score for Reference Solution {i + 1}:**
{datapoint["context_nodes"][i].get(label_name, 0.0)}
""".strip()
        example_str.append(text)

    if len(example_str):
        example_str.insert(
            0,
            ("# REFERENCE SOLUTIONS AND PERFORMANCE:\nBelow are two reference solutions along with their observed 5-fold CV scores on the public training set.\n"),
        )

    feedback_str = []
    for i, feedback in enumerate(feedbacks):
        text = f"** Code in Previous Attempt {i + 1} **\n{feedback[0]}\n\n** Execution Output in Previous Attempt {i + 1} **\n{feedback[1]}\n"
        feedback_str.append(text)
    if len(feedback_str):
        feedback_str.insert(
            0,
            ("# PREVIOUS ATTEMPTS:\nBelow are earlier attempts at estimating the 5-fold CV score, including the submitted code and its execution output.\n"),
        )

    rule_str = []
    for i, rule in enumerate(rules):
        text = f"** Tip {i + 1} **: {rule}"
        rule_str.append(text)

    if len(rule_str):
        rule_str.insert(0, ("# TIPS AND HINTS:\nBelow are some tips and hints for estimating the 5-fold CV score.\n"))

    prompt = version2template[version].format(
        code_a=slot_a["code"],
        code_b=slot_b["code"],
        perf_a=slot_a.get(label_name, 0.0),
        perf_b=slot_b.get(label_name, 0.0),
        label_name=label_name,
        task_desc=task2desc.get(task_name, "No description available"),
        task_name=task_name,
        examples="\n\n".join(example_str),
        feedbacks="\n\n".join(feedback_str),
        rules="\n\n".join(rule_str),
        device=device,
        time_limit=second2timestr(time_limit),
    )

    return prompt, choice


def datapoint_to_final_prediction_prompt(
    label_type: str,
    label_name: str,
    datapoint: dict,
    code: Optional[str] = None,
    execution_output: list[Optional[str]] = [],
    do_shuffle: bool = False,
    version: str = "final_prediction_v1",
    limit_context_nodes: int = 0,
) -> tuple[str, int]:
    """Convert datapoint to prompt for final prediction based on execution output.

    Args:
        label_type: Type of label (should be "ranking")
        label_name: Name of the label metric
        datapoint: The datapoint dictionary containing node_a, node_b, etc.
        execution_output: The execution output text from the pilot experiment
        do_shuffle: Whether to shuffle A/B (should match the shuffle used in code generation)
        version: Template version to use
        limit_context_nodes: Number of context nodes to include

    Returns:
        Tuple of (prompt, choice) where choice is 0 for A, 1 for B
    """
    assert label_type == "ranking"

    # Use the same shuffling logic as datapoint_to_agentbox_prompt to maintain A/B correspondence
    if do_shuffle:
        random_bit = 1  # random.randint(0, 1)
    else:
        random_bit = 0

    if datapoint.get("candidates", None) is not None:
        slot_a = datapoint["candidates"][0] if random_bit == 0 else datapoint["candidates"][1]
        slot_b = datapoint["candidates"][1] if random_bit == 0 else datapoint["candidates"][0]
    else:
        slot_a = datapoint["node_a"] if random_bit == 0 else datapoint["node_b"]
        slot_b = datapoint["node_b"] if random_bit == 0 else datapoint["node_a"]
    choice = datapoint["choice"] if random_bit == 0 else (1 - datapoint["choice"])

    # Get task name for data location hints
    task_name = slot_a.get("task", "unknown")

    # Format examples (context nodes)
    example_str = []
    for i in range(min(limit_context_nodes, len(datapoint["context_nodes"]))):
        text = f"**Reference Solution {i + 1}:**\n{datapoint['context_nodes'][i]['code'].strip()}\n\n**5-fold CV score for Reference Solution {i + 1}:**\n{datapoint['context_nodes'][i].get(label_name, 0.0)}\n"
        example_str.append(text)

    if len(example_str):
        example_str.insert(
            0,
            ("# REFERENCE SOLUTIONS AND PERFORMANCE:\nBelow are reference solutions along with their observed 5-fold CV scores on the public training set.\n"),
        )

    code_str = ""
    if code is not None:
        code_str = f"# PILOT EXPERIMENT CODE:\n\nBelow are the code for the pilot experiments designed to estimate the potential of these two solutions:\n\n```\n{code}\n```\n"

    execution_output_str = ""
    assert isinstance(execution_output, list)
    if len(execution_output) == 1 and execution_output[0] is not None:
        execution_output_str = f"# PILOT EXPERIMENT RESULTS:\n\nBelow are the execution outputs from the pilot experiments designed to estimate the potential of these two solutions:\n\n```\n{execution_output[0]}\n```\n"

    elif len(execution_output) > 1:
        execution_output_str = []
        for i in range(len(execution_output)):
            if execution_output[i] is None:
                continue
            text = f"## The experiment results and findings for {i + 1}-th pilot experiment\n```\n{execution_output[i]}\n```\n"
            execution_output_str.append(text)

        assert len(execution_output_str) <= len(execution_output)

        if len(execution_output_str) > 0:
            execution_output_str.insert(
                0,
                ("# PILOT EXPERIMENT RESULTS:\n\nWe design and run multiple pilot experiments to estimate the potential of these two solutions. The outputs for multiple pilot experiments are shown below:\n"),
            )
        execution_output_str = "\n\n".join(execution_output_str)

    else:
        assert (len(execution_output) == 0) or (len(execution_output) == 1 and execution_output[0] is None)

    prompt = version2template[version].format(
        code_a=slot_a["code"],
        code_b=slot_b["code"],
        task_desc=task2desc.get(task_name, "No description available"),
        code=code_str,
        execution_output=execution_output_str,
        examples="\n\n".join(example_str),
    )

    return prompt, choice


def datapoint_to_final_prediction_prompt_for_tuple(
    label_type: str,
    label_name: str,
    datapoint: dict,
    code: Optional[str] = None,
    execution_output: list[Optional[str]] = [],
    do_shuffle: bool = False,
    version: str = "final_prediction_v1",
    limit_context_nodes: int = 0,
) -> tuple[str, int]:
    """Convert datapoint to prompt for final prediction based on execution output.

    Args:
        label_type: Type of label (should be "ranking")
        label_name: Name of the label metric
        datapoint: The datapoint dictionary containing node_a, node_b, etc.
        execution_output: The execution output text from the pilot experiment
        do_shuffle: Whether to shuffle A/B (should match the shuffle used in code generation)
        version: Template version to use
        limit_context_nodes: Number of context nodes to include

    Returns:
        Tuple of (prompt, choice) where choice is 0 for A, 1 for B
    """
    assert label_name in ("validation_metric", "max_test_metric_in_subtree", "test_metric", "normalized_test_metric"), label_name
    assert limit_context_nodes == 0

    # Get task name for data location hints
    task_name = datapoint["candidates"][0].get("task", "unknown")

    # Format examples (context nodes)
    example_str = []
    # for i in range(min(limit_context_nodes, len(datapoint["context_nodes"]))):
    #     text = (
    #         f"**Reference Solution {i+1}:**\n"
    #         f"{datapoint['context_nodes'][i]['code'].strip()}\n"
    #         f"\n"
    #         f"**5-fold CV score for Reference Solution {i+1}:**\n"
    #         f"{datapoint['context_nodes'][i].get(label_name, 0.0)}\n"
    #     )
    #     example_str.append(text)

    # if len(example_str):
    #     example_str.insert(0, (
    #         "# REFERENCE SOLUTIONS AND PERFORMANCE:\n"
    #         "Below are reference solutions along with their observed 5-fold CV scores on the public training set.\n"
    #     ))

    # code_str = ""
    # if code is not None:
    #     code_str = (
    #         "# PILOT EXPERIMENT CODE:\n"
    #         "\n"
    #         "Below are the code for the pilot experiments designed to estimate the potential of these two solutions:\n"
    #         "\n"
    #         "```\n"
    #         f"{code}\n"
    #         "```\n"
    #     )

    execution_output_str = ""
    assert isinstance(execution_output, list)
    if len(execution_output) == 1 and execution_output[0] is not None:
        execution_output_str = f"# PILOT EXPERIMENT RESULTS:\n\nBelow are the execution outputs from the pilot experiments designed to estimate the potential of these two solutions:\n\n```\n{execution_output[0]}\n```\n"

    elif len(execution_output) > 1:
        execution_output_str = []
        for i in range(len(execution_output)):
            if execution_output[i] is None:
                continue
            text = f"## The experiment results and findings for {i + 1}-th pilot experiment\n```\n{execution_output[i]}\n```\n"
            execution_output_str.append(text)

        assert len(execution_output_str) <= len(execution_output)

        if len(execution_output_str) > 0:
            execution_output_str.insert(
                0,
                ("# PILOT EXPERIMENT RESULTS:\n\nWe design and run multiple pilot experiments to estimate the potential of these two solutions. The outputs for multiple pilot experiments are shown below:\n"),
            )
        execution_output_str = "\n\n".join(execution_output_str)

    else:
        assert (len(execution_output) == 0) or (len(execution_output) == 1 and execution_output[0] is None)

    prompt = version2template[version].format(
        code_a=datapoint["candidates"][0]["code"],
        code_b=datapoint["candidates"][1]["code"],
        code_c=datapoint["candidates"][2]["code"],
        code_d=datapoint["candidates"][3]["code"],
        code_e=datapoint["candidates"][4]["code"],
        task_desc=task2desc.get(task_name, "No description available"),
        code="",
        execution_output=execution_output_str,
        examples="\n\n".join(example_str),
    )

    return prompt, datapoint["choice"]


def datapoint_to_rulebook_prompt(
    datapoint: dict,
    positive_code: str,
    positive_estimation: str,
    positive_score: float,
    negative_code: str,
    negative_estimation: str,
    negative_score: float,
    version: str = "rulebook_v1",
) -> tuple[str, int]:
    """Convert datapoint to prompt for making rule book based on execution output."""

    slot_a = datapoint["node_a"]
    slot_b = datapoint["node_b"]
    task_name = slot_a.get("task", "unknown")

    prompt = version2template[version].format(
        code_a=slot_a["code"],
        code_b=slot_b["code"],
        task_desc=task2desc[task_name],
        code_good=positive_code,
        code_bad=negative_code,
        estimation_good=positive_estimation,
        estimation_bad=negative_estimation,
        score_good=positive_score,
        score_bad=negative_score,
    )

    return prompt, datapoint["choice"]


def arq_code_generate_execution_prompt(
    datapoint: dict,
    version: str = "v1",
    device: str = "CPU only (No GPU)",
    time_limit: int | float = 7200,
    limit_context_nodes: Optional[int] = 0,
    feedbacks: list[tuple[str, str]] = [],
    rules: list[str] = [],
) -> tuple[str, int]:
    """Convert datapoint to prompt for code generation in AgentBox evaluation."""

    label_name = "validation_metric"

    if datapoint.get("candidates", None) is not None:
        datapoint["node_a"] = datapoint["candidates"][0]
        datapoint["node_b"] = datapoint["candidates"][1]
    task_name = datapoint["node_a"].get("task", "unknown")
    n_context_node = min(len(datapoint["context_nodes"]), limit_context_nodes)

    example_str = []
    if n_context_node == 1:
        text = f"**The two options are both adpoted from a base reference solution:**\n{datapoint['context_nodes'][0]['code'].strip()}\n\n**5-fold CV score for Reference Solution:**\n{datapoint['context_nodes'][0].get(label_name, 0.0)}\n"
        example_str.append(text)
    elif n_context_node > 1:
        for i in range(n_context_node):
            text = f"**Reference Solution {i + 1}:**\n{datapoint['context_nodes'][i]['code'].strip()}\n\n**5-fold CV score for Reference Solution {i + 1}:**\n{datapoint['context_nodes'][i].get(label_name, 0.0)}\n"
            example_str.append(text)

    feedback_str = []
    for i, feedback in enumerate(feedbacks):
        text = f"** Code in Previous Attempt {i + 1} **\n{feedback[0]}\n\n** Execution Output in Previous Attempt {i + 1} **\n{feedback[1]}\n"
        feedback_str.append(text)
    if len(feedback_str):
        feedback_str.insert(
            0,
            ("# PREVIOUS ATTEMPTS:\nBelow are earlier attempts at estimating the 5-fold CV score, including the submitted code and its execution output.\n"),
        )

    rule_str = []
    for i, rule in enumerate(rules):
        text = f"** Tip {i + 1} **: {rule}\n"
        rule_str.append(text)

    if len(rule_str):
        rule_str.insert(0, ("# TIPS AND HINTS:\nBelow are some tips and hints for estimating the 5-fold CV score.\n"))

    prompt = version2template[version].format(
        code_a=datapoint["node_a"]["code"],
        code_b=datapoint["node_b"]["code"],
        perf_a=datapoint["node_a"].get(label_name, 0.0),
        perf_b=datapoint["node_b"].get(label_name, 0.0),
        label_name=label_name,
        task_desc=task2desc.get(task_name, "No description available"),
        task_name=task_name,
        examples="\n\n".join(example_str),
        feedbacks="\n\n".join(feedback_str),
        rules="\n\n".join(rule_str),
        device=device,
        time_limit=second2timestr(time_limit),
    )

    choice = datapoint["choice"]
    return prompt, choice


def arq_code_generate_execution_prompt_for_tuple(
    datapoint: dict,
    version: str = "v1",
    device: str = "CPU only (No GPU)",
    time_limit: int | float = 7200,
    limit_context_nodes: Optional[int] = 0,
    feedbacks: list[tuple[str, str]] = [],
    rules: list[str] = [],
) -> tuple[str, int]:
    """Convert datapoint to prompt for code generation in AgentBox evaluation."""

    # label_name = "validation_metric"
    task_name = datapoint["candidates"][0].get("task", "unknown")
    assert limit_context_nodes == 0

    example_str = []
    # if n_context_node == 1:
    #     text = (
    #         f"**The two options are both adpoted from a base reference solution:**\n"
    #         f"{datapoint['context_nodes'][0]['code'].strip()}\n"
    #         f"\n"
    #         f"**5-fold CV score for Reference Solution:**\n"
    #         f"{datapoint['context_nodes'][0].get(label_name, 0.0)}\n"
    #     )
    #     example_str.append(text)
    # elif n_context_node > 1:
    #     for i in range(n_context_node):
    #         text = (
    #             f"**Reference Solution {i+1}:**\n"
    #             f"{datapoint['context_nodes'][i]['code'].strip()}\n"
    #             f"\n"
    #             f"**5-fold CV score for Reference Solution {i+1}:**\n"
    #             f"{datapoint['context_nodes'][i].get(label_name, 0.0)}\n"
    #         )
    #         example_str.append(text)

    feedback_str = []
    # for i, feedback in enumerate(feedbacks):
    #     text = (
    #         f"** Code in Previous Attempt {i+1} **\n"
    #         f"{feedback[0]}\n"
    #         f"\n"
    #         f"** Execution Output in Previous Attempt {i+1} **\n"
    #         f"{feedback[1]}\n"
    #     )
    #     feedback_str.append(text)
    # if len(feedback_str):
    #     feedback_str.insert(0, (
    #         "# PREVIOUS ATTEMPTS:\n"
    #         "Below are earlier attempts at estimating the 5-fold CV score, including the submitted code and its execution output.\n"
    #     ))

    # rule_str = []
    # for i, rule in enumerate(rules):
    #     text = f"** Tip {i+1} **: {rule}\n"
    #     rule_str.append(text)

    # if len(rule_str):
    #     rule_str.insert(0, (
    #         "# TIPS AND HINTS:\n"
    #         "Below are some tips and hints for estimating the 5-fold CV score.\n"
    #     ))

    prompt = version2template[version].format(
        code_a=datapoint["candidates"][0]["code"],
        code_b=datapoint["candidates"][1]["code"],
        code_c=datapoint["candidates"][2]["code"],
        code_d=datapoint["candidates"][3]["code"],
        code_e=datapoint["candidates"][4]["code"],
        task_desc=task2desc.get(task_name, "No description available"),
        task_name=task_name,
        device=device,
        time_limit=second2timestr(time_limit),
        examples="\n\n".join(example_str),
        feedbacks="\n\n".join(feedback_str),
    )

    choice = datapoint["choice"]
    return prompt, choice


def datapoint_to_feedback_prompt(
    datapoint: dict,
    version: str,
    device: str,
    time_limit: int | float,
    limit_context_nodes: int,
    prev_estimations: list[str],
):
    label_name = "validation_metric"
    if datapoint.get("candidates", None) is not None:
        datapoint["node_a"] = datapoint["candidates"][0]
        datapoint["node_b"] = datapoint["candidates"][1]
    task_name = datapoint["node_a"].get("task", "unknown")
    n_context_node = min(limit_context_nodes, len(datapoint["context_nodes"]))

    estimation_str_parts = []
    for i in range(len(prev_estimations)):
        text = f"Results and Findings from Attempt {i + 1}\n{prev_estimations[i]}\n"
        estimation_str_parts.append(text)

    if len(estimation_str_parts):
        estimation_str_head = "## Previous Pilot Experiment Results\nBelow are earlier attempts of pilot experiments and their experimental results.\n"
        estimation_str_parts.insert(0, estimation_str_head)

    example_str_parts = []
    if n_context_node == 1:
        text = f"**The two options are both adpoted from a base reference solution:**\n{datapoint['context_nodes'][0]['code'].strip()}\n\n**5-fold CV score for Reference Solution:**\n{datapoint['context_nodes'][0].get(label_name, 0.0)}\n"
        example_str_parts.append(text)
    elif n_context_node > 1:
        for i in range(n_context_node):
            text = f"**Reference Solution {i + 1}:**\n{datapoint['context_nodes'][i]['code'].strip()}\n\n**5-fold CV score for Reference Solution {i + 1}:**\n{datapoint['context_nodes'][i].get(label_name, 0.0)}\n"
            example_str_parts.append(text)

    if len(example_str_parts) > 0:
        example_str_head = "## Reference Solutions for This Task"
        example_str_parts.insert(0, example_str_head)

    prompt = version2template[version].format(
        code_a=datapoint["node_a"]["code"],
        code_b=datapoint["node_b"]["code"],
        perf_a=datapoint["node_a"].get(label_name, 0.0),
        perf_b=datapoint["node_b"].get(label_name, 0.0),
        label_name=label_name,
        task_desc=task2desc.get(task_name, "No description available"),
        task_name=task_name,
        examples="\n\n".join(example_str_parts),
        prev_estimations="\n\n".join(estimation_str_parts),
        device=device,
        time_limit=second2timestr(time_limit),
    )

    return prompt


def datapoint_to_feedback_prompt_for_tuple(
    datapoint: dict,
    version: str,
    device: str,
    time_limit: int | float,
    limit_context_nodes: int,
    prev_estimations: list[str],
):
    label_name = "validation_metric"
    task_name = datapoint["candidates"][0].get("task", "unknown")
    assert limit_context_nodes == 0

    estimation_str_parts = []
    for i in range(len(prev_estimations)):
        text = f"Results and Findings from Attempt {i + 1}\n{prev_estimations[i]}\n"
        estimation_str_parts.append(text)

    if len(estimation_str_parts):
        estimation_str_head = "## Previous Pilot Experiment Results\nBelow are earlier attempts of pilot experiments and their experimental results.\n"
        estimation_str_parts.insert(0, estimation_str_head)

    example_str_parts = []
    # if n_context_node == 1:
    #     text = (
    #         f"**The two options are both adpoted from a base reference solution:**\n"
    #         f"{datapoint['context_nodes'][0]['code'].strip()}\n"
    #         f"\n"
    #         f"**5-fold CV score for Reference Solution:**\n"
    #         f"{datapoint['context_nodes'][0].get(label_name, 0.0)}\n"
    #     )
    #     example_str_parts.append(text)
    # elif n_context_node > 1:
    #     for i in range(n_context_node):
    #         text = (
    #             f"**Reference Solution {i+1}:**\n"
    #             f"{datapoint['context_nodes'][i]['code'].strip()}\n"
    #             f"\n"
    #             f"**5-fold CV score for Reference Solution {i+1}:**\n"
    #             f"{datapoint['context_nodes'][i].get(label_name, 0.0)}\n"
    #         )
    #         example_str_parts.append(text)

    # if len(example_str_parts) > 0:
    #     example_str_head = (
    #         f"## Reference Solutions for This Task"
    #     )
    #     example_str_parts.insert(0, example_str_head)

    prompt = version2template[version].format(
        code_a=datapoint["candidates"][0]["code"],
        code_b=datapoint["candidates"][1]["code"],
        code_c=datapoint["candidates"][2]["code"],
        code_d=datapoint["candidates"][3]["code"],
        code_e=datapoint["candidates"][4]["code"],
        label_name=label_name,
        task_desc=task2desc.get(task_name, "No description available"),
        task_name=task_name,
        examples="\n\n".join(example_str_parts),
        prev_estimations="\n\n".join(estimation_str_parts),
        device=device,
        time_limit=second2timestr(time_limit),
    )
    return prompt
