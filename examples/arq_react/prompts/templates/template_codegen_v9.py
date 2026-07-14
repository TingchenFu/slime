# template for overfit experiment
CODEGEN_TEMPLATE_V9 = """
# INTRODUCTION:

You are a Kaggle Grandmaster and Lead Data Scientist.
Your task is to **directly measure** which of two candidate machine learning solutions achieves a better **5-fold cross-validation score** by running the exact 5-fold CV — no approximation, no subsampling.

Imagine there are two candidate code solutions for a specific machine learning competition at Kaggle. We evaluate them by their 5-fold CV score on the public training set. **You must run the full, exact 5-fold CV for both options and report the resulting scores.**

This is a measurement task, not a prediction task. You have enough time to run the full 5-fold CV.


# TASK DESCRIPTION
````
{task_desc}
````

You can access the data using standard Python libraries (pandas, datasets, etc.).
The data is already mounted and ready to use — no need to download or prepare it.

**Important:** Pay attention to whether higher or lower values indicate better performance (e.g., accuracy is higher-is-better, RMSE is lower-is-better).

**Common Pitfalls:**
- Do NOT use `from transformers import AdamW` (it has been removed). Use `from torch.optim import AdamW` instead.

**Your Resources:** {device}

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


# INSTRUCTIONS:
You must write a **Python script** that runs the **full, exact 5-fold CV** for both Option A and Option B and writes the results to `./execution.out`.

Requirements:
- Use **exactly 5 folds** — do not reduce the number of folds.
- Use the **complete training dataset** — do not subsample or truncate.
- Reproduce the ML approach of each option faithfully (same model, same hyperparameters, same preprocessing).
- Do **not** estimate or predict which option is better — measure it by running the CV.

The only requirement is that your script writes its findings to the output file described below.


# OUTPUT REQUIREMENT:
Your Python code **MUST** generate a **plain text** file at `./execution.out` in the current working directory. The content of `./execution.out` will be read and appended to the input context when asking a language model to predict which solution has a superior 5-fold CV score. Therefore, write your findings in clear, readable text that would help a language model make an informed judgment.


{examples}

{feedbacks}

# RESPONSE FORMAT:

## Step 1: Plan

Briefly describe how you will implement the 5-fold CV for each option (e.g., how you load data, what model/preprocessing you use for each option, how you compute the metric per fold).

## Step 2: Code

Then provide a **SINGLE** Markdown code block (wrapped in ```) containing a **SELF-CONTAINED** Python script that runs the full 5-fold CV for both options and writes all per-fold scores and the mean CV score to `./execution.out`.


## Example Output:

## Step 1: Plan
<brief description of the CV implementation plan for Option A and Option B>

## Step 2: Code
```python
import pandas as pd
from sklearn.model_selection import KFold
# ... full 5-fold CV for Option A and Option B ...

findings = []
findings.append("=== Full 5-Fold CV Results ===")
findings.append(f"Option A — fold scores: {{scores_a}}, mean CV: {{mean_a:.4f}}")
findings.append(f"Option B — fold scores: {{scores_b}}, mean CV: {{mean_b:.4f}}")
findings.append(f"Conclusion: Option {{'A' if mean_a > mean_b else 'B'}} achieves a higher mean CV score.")

with open('./execution.out', 'w') as f:
    f.write("\\n".join(findings))
```
"""
