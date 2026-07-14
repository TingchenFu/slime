CODEGEN_TEMPLATE_V10 = """
# INTRODUCTION:

You are a Kaggle Grandmaster and Lead Data Scientist acting as a **Strategic Evaluator**.
Your task is to analyze two candidate machine learning solutions and **predict which one has superior 5-fold CV scores without actually running the full 5-fold CV**.

Imagine there are two candidate code solutions for a specific machine learning competition at Kaggle. We don't have access to the test set and we can only evaluate these two code solutions by 5-fold CV score on the public training set. However, running the full 5-fold CV on the complete dataset is prohibitively time-consuming and out of our budget. Therefore, you must predict which solution would achieve a superior 5-fold CV score if they were fully executed, using fast, high-leverage experiments (Pilot Experiment).

Remember: your goal is NOT to build the final solution, but to generate code that is helpful for **predicting the relative rank between two solutions in 5-fold CV score**.


# TASK DESCRIPTION
````
{task_desc}
````

You can access the data using standard Python libraries (pandas, datasets, etc.).
The data is already mounted and ready to use — no need to download or prepare it.

**Important:** Pay attention to whether higher or lower values indicate better performance (e.g., accuracy is higher-is-better, RMSE is lower-is-better).

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


# INSTRUCTIONS for Pilot Experiment:
You must write a **fast, efficient Python script** (targeting under {time_limit} of runtime).

**Mandatory Strategy: Subsampled 3-Fold Cross-Validation**

You MUST use the following strategy for ALL task types unless the task explicitly involves sequential/temporal data:
1. Randomly subsample **40% of the training data** (use `random_state=42` for reproducibility).
2. Run **3-fold cross-validation** on this subsample for both Option A and Option B.
3. Report the mean CV score for each option across the 3 folds.

This strategy is mandatory because it provides the best balance of speed and reliability for predicting full 5-fold CV rank. Do NOT use a single train/validation holdout split — holdout estimates are too noisy and often give the wrong winner.

**Exception:** For neural network / deep learning tasks where 3-fold CV is too slow even on 40% data, use 3 epochs on a 20% subsample as a proxy, but this is a last resort.


## Environment Constraints (CRITICAL — violations will crash your code):

**Data loading:**
- `load_from_disk('/workspace/data/train')` may raise `ValueError: Feature type 'List' not found` on some tasks. If this happens, **catch the error and load with pyarrow directly**:
  ```python
  try:
      import datasets; train_ds = datasets.load_from_disk('/workspace/data/train')
      df = train_ds.to_pandas()
  except ValueError:
      import pyarrow as pa
      reader = pa.ipc.open_stream(open('/workspace/data/train/data-00000-of-00001.arrow', 'rb'))
      df = reader.read_all().to_pandas()
  ```
- Do NOT assume the dataset has a `fold` column — it typically does not.
- `/workspace/data/train/` is mounted **read-only**. Do NOT call `datasets.Dataset.select()`, `.map()`, `.filter()` — these write temp files. Convert to pandas first with `.to_pandas()`, then subsample.

**Package availability:**
- **NOT installed**: `rouge_score`, `sktime`, `pyext`, `jiwer`, `pycocoevalcap`. Do not import them.
- No external network: do NOT download model weights or data at runtime.

**API compatibility:**
- HuggingFace dataset indexing: always wrap numpy indices with `int()` — `numpy.int64` is not accepted.
- LightGBM: use callbacks for early stopping: `callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]`. Do NOT pass `verbose_eval` or `early_stopping_rounds` to `lgb.train()`.
- XGBoost: pass `early_stopping_rounds` and `eval_metric` to the **constructor**, not to `.fit()`.
- scikit-learn: use `OneHotEncoder(sparse_output=...)`, not `sparse=...`.
- transformers: use `from torch.optim import AdamW`, not `from transformers import AdamW`.


# OUTPUT REQUIREMENT:
Your Python code **MUST** generate a **plain text** file at `./execution.out` in the current working directory. The content of `./execution.out` will be read by a language model to predict which solution has a superior 5-fold CV score.

**Always end `execution.out` with this exact summary block:**
```
=== SUMMARY ===
Option A score: 0.XXXX (mean of 3-fold CV on 40% subsample)
Option B score: 0.XXXX (mean of 3-fold CV on 40% subsample)
Delta (B - A): +/-0.XXXX
Winner: A  [or B, or UNCLEAR if |delta| < 0.5% of expected metric range]
```
- Report `Winner: UNCLEAR` if the delta is very small (e.g., < 0.002 for accuracy, < 0.01 for RMSE) — do not force a winner when results are within noise.
- A clear numeric summary makes it significantly easier to predict the correct answer.


{examples}

{feedbacks}

# RESPONSE FORMAT:

## Step 1: Experimental Design

Briefly confirm you will use subsampled 3-fold CV (40% data, 3 folds). If you must deviate (e.g., neural network task), explain why.

## Step 2: Probing Code

Then provide a **SINGLE** Markdown code block (wrapped in ```) containing a **SELF-CONTAINED** Python script. It must be self-contained, complete within {time_limit}, and **write the results to `./execution.out` in the current working directory**.


## Example Output:

## Step 1: Experimental Design
Using subsampled 3-fold CV: 40% of training data, 3 folds, random_state=42. This gives a reliable relative ranking proxy within the time budget.

## Step 2: Probing Code
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# Load data
# ... load and subsample to 40% ...

scores_a, scores_b = [], []
kf = KFold(n_splits=3, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(X):
    # train and score Option A and B ...
    scores_a.append(score_a_fold)
    scores_b.append(score_b_fold)

mean_a = np.mean(scores_a)
mean_b = np.mean(scores_b)
delta = mean_b - mean_a
winner = "UNCLEAR" if abs(delta) < 0.002 else ("B" if delta > 0 else "A")

findings = []
findings.append("=== Pilot Results (3-fold CV, 40% subsample) ===")
findings.append(f"Option A fold scores: {{[round(s,4) for s in scores_a]}}")
findings.append(f"Option B fold scores: {{[round(s,4) for s in scores_b]}}")
findings.append("")
findings.append("=== SUMMARY ===")
findings.append(f"Option A score: {{mean_a:.4f}} (mean of 3-fold CV on 40% subsample)")
findings.append(f"Option B score: {{mean_b:.4f}} (mean of 3-fold CV on 40% subsample)")
findings.append(f"Delta (B - A): {{delta:+.4f}}")
findings.append(f"Winner: {{winner}}")

with open('./execution.out', 'w') as f:
    f.write("\\n".join(findings))
```
"""
