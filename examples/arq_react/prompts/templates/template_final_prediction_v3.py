# inspired by the llm-as-judge baseline from main branch
FINAL_PREDICTION_TEMPLATE_V3 = """
You are a strict judge selecting between TWO candidate solutions to the SAME machine learning task.

Your goal is to choose the candidate more likely to achieve a better test score, using pilot experiment evidence.

# TASK DESCRIPTION
````
{task_desc}
````

# CANDIDATE SOLUTIONS:

## Candidate A:
```python
{code_a}
```

## Candidate B:
```python
{code_b}
```

{examples}


{code}


{execution_output}

# DECISION RULES:
- Prefer the candidate more likely to produce a better test score for the given task.
- Trust empirical evidence (pilot experiment results) over theoretical concerns when the signal is clear.
- If pilot results are missing, truncated, or contain no numeric scores, reason purely from code quality.
- Reference solutions (if provided above) are NOT candidates — use them only as supporting evidence.
- Prefer correctness, robustness, and task-fit over style or verbosity.
- If you are genuinely uncertain after weighing all evidence, pick the candidate with stronger code quality fundamentals.

# OUTPUT FORMAT (STRICT):
- Think step by step and provide your reasoning before giving a final answer.
- Give a final answer of A for Candidate A and B for Candidate B.
- Provide your answer inside a \\boxed{{}}, ie \\boxed{{A}} or \\boxed{{B}}.
"""
