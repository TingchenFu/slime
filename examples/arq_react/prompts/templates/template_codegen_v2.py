CODEGEN_TEMPLATE_V2 = """
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


# INSTRUCTIONS for Pilot Experiment:
You must write a **fast, efficient Python script** (targeting under {time_limit} of runtime). When executed, this script should produce evidence that helps determine which candidate solution would achieve a better 5-fold CV score.

You have full freedom in choosing your analysis strategy. For example, you may:
- Run a simplified or partial cross-validation (e.g., fewer folds, subsampled data).
- Compare key architectural or hyperparameter differences between the two solutions and run targeted ablation experiments.
- Evaluate both solutions on a single train/validation split.
- Perform any other analysis you believe is informative.

The only requirement is that your script writes its findings to the output file described below. Here are some helpful tips for writing efficient and effective code:
1. Include explicit, quantitative metrics (e.g., accuracy and execution time) alongside a definitive text conclusion in the output to clearly guide the final prediction.
2. Implement robust data loading mechanisms with fallbacks (such as parsing raw underlying files) to prevent the entire experiment from failing due to library version incompatibilities.
3. Train candidate models for a sufficient number of epochs to ensure performance differences reflect actual learning capacity rather than noisy early-stage convergence.
4. End the evidence output with a clear, explicit conclusion statement that directly compares the quantitative results of both options to remove ambiguity for the evaluator.
5. Record and output step-by-step validation metrics across epochs to clearly demonstrate training trajectories and expose models that fail to converge.
6. Wrap each candidate's evaluation loop in a `try-except` block that writes the full traceback to the output file, ensuring actionable partial evidence is preserved even if an experiment crashes.
7. Conduct side-by-side evaluations of candidate solutions using the exact same data splits and random seeds to ensure a fair, apples-to-apples comparison.
8. Safeguard against degenerate or identical metric outputs (such as exactly 0.0 or 1.0) by logging a 'Tie' or 'Metric Evaluation Error' rather than blindly trusting comparative operators.
9. Run a subset of cross-validation folds (e.g., 2 out of 5) to obtain a reliable and fast comparative proxy for the full evaluation.
10. Pilot experiments must strictly replicate the candidate scripts' idiosyncratic training loops and evaluation logic to capture the true impact of any implementation bugs on the final metric.





# OUTPUT REQUIREMENT:
Your Python code **MUST** generate a **plain text** file at `./execution.out` in the current working directory. The content of `./execution.out` will be read and appended to the input context when asking a language model to predict which solution has a superior 5-fold CV score. Therefore, write your findings in clear, readable text that would help a language model make an informed judgment.


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
findings.append(f"Option A - Validation Score (1-fold, 20%% subsample): {{score_a}}")
findings.append(f"Option B - Validation Score (1-fold, 20%% subsample): {{score_b}}")
findings.append(f"Option A trains in {{time_a:.1f}}s, Option B trains in {{time_b:.1f}}s")
findings.append("Conclusion: Option B shows higher validation accuracy on the subsample.")

with open('./execution.out', 'w') as f:
    f.write("\\n".join(findings))
```
"""
