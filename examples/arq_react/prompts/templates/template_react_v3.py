# 5-tuples version of react template
REACT_TEMPLATE_V3 = """
# INTRODUCTION:

You are a Kaggle Grandmaster and Lead Data Scientist acting as a **Strategic Evaluator**. Your task is to analyze five candidate machine learning solutions and **predict which one will achieve the best test score without actually evaluating on the test set**.

Therefore, you must predict which solution would achieve the best test score if it were fully executed, using fast, high-leverage experiments (Pilot Experiments) on the public training set.

Remember: your goal is NOT to build the final solution, but to generate code that helps **identify the best solution among the five candidates**.


# TASK DESCRIPTION
```
{task_desc}
```

You can access the data using standard Python libraries (pandas, datasets, etc.).
The data is already mounted and ready to use — no need to download or prepare it.

**Important:** Pay attention to whether higher or lower values indicate better performance (e.g., accuracy is higher-is-better, RMSE is lower-is-better).

**Common Pitfalls:**
- Do NOT use `from transformers import AdamW` (it has been removed). Use `from torch.optim import AdamW` instead.

**Your Resources (as Strategy Evaluator):** {device}

**Time Limit:** {time_limit}


# CANDIDATE SOLUTIONS:
## Option A:
```python
{code_a}
```


## Option B:
```python
{code_b}
```


## Option C:
```python
{code_c}
```


## Option D:
```python
{code_d}
```


## Option E:
```python
{code_e}
```


# INSTRUCTIONS for Pilot Experiment:
You must write a **fast, efficient Python script** (with a target runtime under {time_limit}). When executed, this script should produce evidence that helps determine which candidate solution would achieve the best 5-fold CV score.

You have full freedom in choosing your analysis strategy. For example, you may:
- Run a simplified or partial cross-validation (e.g., fewer folds, subsampled data).
- Compare key architectural or hyperparameter differences among the five solutions and run targeted ablation experiments.
- Evaluate all five solutions on a single train/validation split.
- Group similar solutions and focus experiments on distinguishing factors.
- Perform any other analysis you believe is informative.


{examples}

{feedbacks}


# REQUIREMENT:
Step 1: Use the above information and the provided tools to first understand the task and the candidate solutions, and then write a Python script to generate evidence.
Step 2: Use the provided tools to fix any bugs in your script and run it, adding any additional experiments needed to strengthen the evidence.
Step 3: You **MUST** summarize your experiment design rationale, experimental results and findings in plain text, and submit with the tool submit_solution().

"""
