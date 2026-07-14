## Add top-10 rulebooks collected from eval_arq12 and eval_arq20
CODEGEN_TEMPLATE_V5 = """
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
**TIP1**: When comparing segmentation candidates where one relies on pretrained encoders with a short fine-tune and the other relies on a long training schedule, keep the pilot\u2019s initialization and schedule faithful to each option (use pretrained weights where the candidate does, and allocate extra pilot epochs to the long-schedule model) because changing initialization or compressing the schedule can invert the early-epoch ranking.
**TIP2**: When one candidate\u2019s strength is model ensembling at inference, evaluate that option in the pilot with the same aggregation (e.g., average sigmoid probabilities) and report both component and ensemble IoU because measuring only single components underestimates the ensemble\u2019s advantage.
**TIP3**: When candidates use materially different augmentations on histopathology segmentation (e.g., Option A uses H/V flips + color jitter vs Option B uses H flip only), mirror these augmentations in the pilot because augmentation choice affects convergence and generalization in tissue textures and can bias the relative proxy performance.
**TIP4**: When comparing video captioning models whose main architectural difference is how they encode motion (e.g., temporal conv vs. SE+frame-diff), use training loss or convergence speed as a tie-breaker when early BLEU is within a small margin, because explicit frame-differencing tends to show faster optimization before BLEU separates under short pilots.
**TIP5**: When running a quick proxy for sequence-to-sequence video captioning with a learned vocabulary, build the vocabulary from the pilot\u2019s training split only (exclude validation) because including validation captions inflates BLEU and masks relative generalization differences between temporal encoders.
**TIP6**: When comparing a mutual-learning or multi-network method (e.g., two nets with KL in Deep Mutual Learning) to a single-network fusion model trained from scratch on images, include a short warm-up (1 epoch of CE only) or run at least 3\u20134 epochs before ranking because mutual-learning\u2019s benefit appears only after soft targets stabilize and early-epoch proxies systematically undervalue it.
**TIP7**: When the dataset has many classes (\u2248200+) and you must cap sample counts for speed, first draw a stratified subset from the full pool and then do a stratified train/val split within it, rather than taking a CV fold and then capping, because post-fold capping distorts class coverage in the tiny pilot and can suppress the cross-class knowledge transfer advantages of mutual-learning methods.
**TIP8**: When the evaluation metric is accuracy on a potentially imbalanced medical image dataset and you compare a rotation-auxiliary multi-head CNN to a MixUp/label-smoothed supervised CNN, use a single shared stratified holdout that preserves the natural class distribution for both options, because accuracy is base-rate sensitive and this setup better predicts the true CV ordering.
**TIP9**: When one candidate is a light CNN with an auxiliary self-supervised head and the other is a higher-capacity supervised model, mirror each model\u2019s distinctive training signals in the pilot (e.g., rotation head + cosine LR vs MixUp/CutMix + EMA) and compare on the same split after a few epochs, because early-phase dynamics often determine their relative ranking under tight budgets.
**TIP10**: When runtime and evidence reliability are constrained and both candidates are CNNs on images, prefer one stratified holdout on a large stratified subsample over small-k CV on a rebalanced subset, because it reduces failure surface and yields clearer, comparable evidence tied to the real label distribution.

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
