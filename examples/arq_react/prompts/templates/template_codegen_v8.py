# template proposed by claude code
CODEGEN_TEMPLATE_V8 = """
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
You must write a **fast, efficient Python script** (targeting under {time_limit} of runtime). When executed, this script should produce evidence that helps determine which candidate solution would achieve a better 5-fold CV score.

You have full freedom in choosing your analysis strategy. For example, you may:
- Run a simplified or partial cross-validation (e.g., fewer folds, subsampled data) — best for tabular/sklearn-style models.
- Train both options for a few epochs on a small data subsample and compare validation metrics — best for neural networks.
- Compare key architectural or hyperparameter differences between the two solutions and run targeted ablation experiments.

**Strategy tips:**
- For tabular/sklearn tasks: prefer 2–3 fold CV on a 20% subsample over a single holdout split, as it is more reliable.
- For neural network tasks: run both options for enough epochs to see meaningful separation; if results are within noise, try a different seed or secondary split.
- Avoid pure single train/val holdout if you can afford more folds — holdout estimates are noisier.
- When pilot results are close (delta < 0.5% of the metric range), explicitly state the ambiguity and reason from code structure rather than just the numbers.


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

**Always end `execution.out` with a compact summary block:**
```
=== SUMMARY ===
Option A score: 0.XXXX
Option B score: 0.XXXX
Delta (B - A): +/-0.XXXX
Winner: A  [or B, or UNCLEAR if delta < noise threshold]
```
A clear, numeric summary makes it significantly easier to predict the correct answer.


{examples}

{feedbacks}

# RESPONSE FORMAT:

## Step 1: Experimental Design

Start by explaining your pilot strategy and plan. Why is this specific experiment a good proxy for predicting the relative performance of the two solutions in full 5-fold CV?

## Step 2: Probing Code

Then provide a **SINGLE** Markdown code block (wrapped in ```) containing a **SELF-CONTAINED** Python script. It must be self-contained, complete within {time_limit}, and **write the results to `./execution.out` in the current working directory**.


## Example Output:

## Step 1: Experimental Design
<the proposed probing plan>

## Step 2: Probing Code
```python
import pandas as pd
# ... your fast experiment ...

findings = []
findings.append("=== Pilot Experiment Results ===")
findings.append(f"Option A - Validation Score: {{score_a:.4f}}")
findings.append(f"Option B - Validation Score: {{score_b:.4f}}")
findings.append("")
findings.append("=== SUMMARY ===")
findings.append(f"Option A score: {{score_a:.4f}}")
findings.append(f"Option B score: {{score_b:.4f}}")
findings.append(f"Delta (B - A): {{score_b - score_a:+.4f}}")
findings.append(f"Winner: {{'B' if score_b > score_a else 'A'}}")

with open('./execution.out', 'w') as f:
    f.write("\\n".join(findings))
```
"""
