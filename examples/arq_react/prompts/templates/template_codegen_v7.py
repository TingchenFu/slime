# ## Add random-10 rulebooks collected from eval_arq12 and eval_arq20
CODEGEN_TEMPLATE_V7 = """
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
**TIP1**: When candidate methods use different native training recipes (optimizer/augmentation), include cross-evaluation (train each distilled set under both pipelines) because this disentangles data quality from pipeline advantages and prevents misranking due to recipe-specific bias.
**TIP2**: When an LLM will read the evidence to pick a winner and the models’ gap is likely modest, end the output with a compact numeric summary (Option A avg AUC, Option B avg AUC, Delta, and an explicit winner), because verbose or truncated logs can hide small but decisive differences.
**TIP3**: When performance differences are expected to be in low single-digit accuracy for scene text recognition with NN retrieval, add a secondary 80/20 hold-out and write both CV and hold-out numbers to the evidence, because a second view provides a stable tie-breaker when CV means are too close to separate.
**TIP4**: When candidates bind different representation dimensionalities to their algorithms (e.g., LightGBM on SVD=128 vs XGBoost on SVD=200), sweep those specific dimensions in the proxy CV for each algorithm because booster performance is sensitive to embedding dimension and matching dims avoids biased conclusions.
**TIP5**: When comparing vision dataset distillation methods that differ mainly in image priors (e.g., Gaussian blur vs. patch self-similarity/TV) and the downstream evaluator is a fixed ConvNet trained on real images, run cross-validation on the full training distribution (≥3 folds) with the exact evaluator hyperparameters, because texture-statistics priors can overperform on small subsamples and only full-distribution CV preserves the correct ranking.
**TIP6**: When the dataset has many classes (≈200+) and you must cap sample counts for speed, first draw a stratified subset from the full pool and then do a stratified train/val split within it, rather than taking a CV fold and then capping, because post-fold capping distorts class coverage in the tiny pilot and can suppress the cross-class knowledge transfer advantages of mutual-learning methods.
**TIP7**: When one candidate computes inverse-frequency class weights per KFold split in a many-class setting, simulate those splits in the pilot and enforce weight vectors with minlength=num_classes and capped ratios (or smoothing) because zero-count classes or mismatched weight lengths can cause instabilities that materially affect CV outcomes.
**TIP8**: When comparing a lighter ResNet-style CNN to a larger EfficientNet-style CNN both trained from scratch on fine-grained natural images, avoid tiny per-class caps (e.g., ≤30) and either use ≥100 samples per class or equalize by a fixed number of optimizer steps because small caps/short pilots reward faster initial convergence and can mis-rank the higher-capacity model.
**TIP9**: When one candidate is a light CNN with an auxiliary self-supervised head and the other is a higher-capacity supervised model, mirror each model’s distinctive training signals in the pilot (e.g., rotation head + cosine LR vs MixUp/CutMix + EMA) and compare on the same split after a few epochs, because early-phase dynamics often determine their relative ranking under tight budgets.
**TIP10**: When one candidate includes a dimensionality reduction stage before cosine retrieval (e.g., PCA on flattened word images), include that stage in the pilot for that candidate and use its native similarity (cosine/dot on normalized), because the reduction changes neighborhood structure and can flip which method wins.



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
