# break through 5-fold CV
CC_TEMPLATE_V2 = """
# INTRODUCTION

You are a Kaggle Grandmaster and Lead Data Scientist. Your job is to **predict which of two candidate machine learning solutions will score higher on the test set**, without actually running the test evaluation.

You have two candidate solutions for a Kaggle machine learning competition. Since the test set is unavailable, you must design targeted experiments (Pilot Experiments) on the training set to gather evidence and predict which solution is more likely to achieve the better test score.

Rather than relying on standard 5-fold cross-validation, you are encouraged to design experiments that better capture the factors affecting test-time performance (e.g., robustness to distribution shift, sensitivity to hyperparameters, overfitting behavior). You have full freedom to choose your experimental methodology.

Keep in mind: your goal is **not** to build a final solution, but to produce code that helps **predict the relative ranking of the two candidates** and output a clear verdict.


# TASK DESCRIPTION
```
{task_desc}
```

The data is already mounted and accessible via standard Python libraries (e.g., pandas, datasets) — no downloading or preprocessing required.

**Note:** Be mindful of the metric direction — whether higher is better (e.g., accuracy) or lower is better (e.g., RMSE).

**Common Pitfalls:**
- `from transformers import AdamW` has been deprecated and removed. Use `from torch.optim import AdamW` instead.

**Available Resources:** {device}

**Time Budget:** {time_limit}. This is generous — prioritize experimental quality and thoroughness over speed, while keeping execution within the budget.


# CANDIDATE SOLUTIONS

## Option A
```python
{code_a}
```

## Option B
```python
{code_b}
```


# PILOT EXPERIMENT INSTRUCTIONS

Design and implement a Python script that produces concrete evidence to predict which candidate solution is more likely to achieve a better test score.

You have complete freedom in your experimental approach. Some strategies you might consider:
- Run a lightweight version of cross-validation (e.g., fewer folds, a data subset).
- Identify key architectural or hyperparameter differences between the two solutions and run targeted ablations.
- Train and evaluate both solutions on a single train/validation split.
- Analyze overfitting behavior, generalization gaps, or sensitivity to data perturbations.
- Any other analysis you find informative.

If no better strategy comes to mind, standard 5-fold CV on the training set is an acceptable fallback.

{examples}

{feedbacks}


# REQUIREMENTS

1. Study the task description and both candidate solutions, then write a Python script that generates supporting evidence.
2. Debug and execute your script. Add follow-up experiments as needed to strengthen your conclusions.
3. You **must** write a plain-text summary to `./execution.out` that includes:
   - Your experimental rationale and methodology.
   - Key results and numerical findings.

"""
