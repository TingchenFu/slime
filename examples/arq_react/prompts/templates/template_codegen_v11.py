CODEGEN_TEMPLATE_V11 = """
You are a code evaluator designing a fast pilot experiment to determine which of TWO candidate solutions achieves a better test score.

# TASK DESCRIPTION
````
{task_desc}
````

You can access the data using standard Python libraries (pandas, datasets, etc.).
The data is already mounted and ready to use — no need to download or prepare it.

**Your Resources:** {device}

**Time Limit:** {time_limit}


# CANDIDATE SOLUTIONS:

## Candidate A:
```python
{code_a}
```

## Candidate B:
```python
{code_b}
```


# RULES:
- Your goal is NOT to build a final solution, but to generate code that helps **predict which candidate achieves a better test score**.
- Pay attention to whether higher or lower values indicate better performance (e.g., accuracy is higher-is-better, RMSE is lower-is-better).
- Reference solutions (if provided below) are NOT candidates — use them only as supporting evidence for what works on this task.
- Do NOT use `from transformers import AdamW` (removed). Use `from torch.optim import AdamW` instead.


# INSTRUCTIONS for Pilot Experiment:

Write a **fast, efficient Python script** (targeting under {time_limit} of runtime). When executed, this script should produce evidence that helps determine which candidate solution would achieve a better test score.

You have full freedom in choosing your analysis strategy. For example, you may:
- Run a simplified or partial cross-validation (e.g., fewer folds, subsampled data).
- Compare key architectural or hyperparameter differences and run targeted ablation experiments.
- Evaluate both solutions on a single train/validation split.
- Perform any other analysis you believe is informative.

The only requirement is that your script writes its findings to the output file described below.


# OUTPUT REQUIREMENT:

Your Python code **MUST** generate a **plain text** file at `./execution.out` in the current working directory. The content of `./execution.out` will be read by a language model to predict which candidate achieves a better test score. Write your findings in clear, readable text that would help a language model make an informed judgment.


{examples}

{feedbacks}

# RESPONSE FORMAT:

## Step 1: Experimental Design

Start by explaining your pilot strategy and plan. Why is this specific experiment a good proxy for predicting the relative performance of the two candidates?

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
findings.append(f"Candidate A - Validation Score (1-fold, 20%% subsample): {{score_a}}")
findings.append(f"Candidate B - Validation Score (1-fold, 20%% subsample): {{score_b}}")
findings.append(f"Candidate A trains in {{time_a:.1f}}s, Candidate B trains in {{time_b:.1f}}s")
findings.append("Conclusion: Candidate B shows higher validation accuracy on the subsample.")

with open('./execution.out', 'w') as f:
    f.write("\\n".join(findings))
```
"""
