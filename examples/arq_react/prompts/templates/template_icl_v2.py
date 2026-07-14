ICL_TEMPLATE_V2 = """
You are a strict judge selecting between TWO candidate solutions to the SAME machine learning task.

Your goal is to choose the candidate more likely to achieve a better test score.

Task description:
```markdown
{task_desc}
```

Context from various solutions to the same machine learning task. These are NOT the candidates you are judging.
{examples}

Candidate A — Code:
```python
{code_a}
```

Candidate B — Code:
```python
{code_b}
```

Decision rules:
- Prefer the candidate more likely to produce a better test score for the given task.
- Prefer correctness, robustness, and task-fit over style or verbosity.
- Use context nodes only as supporting evidence (e.g., what has already been tried, what validation score looked like).
- Do not assume the context nodes are optimal; the new candidates may be better.

Output format (STRICT):
- Think step by step and provide your reasoning before giving a final answer.
- Give a final answer of A for Candidate A and B for Candidate B.
- Provide your answer inside a \\boxed{{}}, ie \\boxed{{A}} or \\boxed{{B}}.

"""
