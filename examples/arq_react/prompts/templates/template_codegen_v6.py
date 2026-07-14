## Add top-10 rulebooks collected from eval_arq12 and eval_arq20, requesting that one dp could produce at most 1 rules.
CODEGEN_TEMPLATE_V6 = """
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
**TIP1**: When comparing segmentation candidates where one relies on pretrained encoders with a short fine-tune and the other relies on a long training schedule, keep the pilot’s initialization and schedule faithful to each option (use pretrained weights where the candidate does, and allocate extra pilot epochs to the long-schedule model) because changing initialization or compressing the schedule can invert the early-epoch ranking.
**TIP2**:When comparing video captioning models whose main architectural difference is how they encode motion (e.g., temporal conv vs. SE+frame-diff), use training loss or convergence speed as a tie-breaker when early BLEU is within a small margin, because explicit frame-differencing tends to show faster optimization before BLEU separates under short pilots.
**TIP3**:When comparing a mutual-learning or multi-network method (e.g., two nets with KL in Deep Mutual Learning) to a single-network fusion model trained from scratch on images, include a short warm-up (1 epoch of CE only) or run at least 3–4 epochs before ranking because mutual-learning’s benefit appears only after soft targets stabilize and early-epoch proxies systematically undervalue it.
**TIP4**:When the evaluation metric is accuracy on a potentially imbalanced medical image dataset and you compare a rotation-auxiliary multi-head CNN to a MixUp/label-smoothed supervised CNN, use a single shared stratified holdout that preserves the natural class distribution for both options, because accuracy is base-rate sensitive and this setup better predicts the true CV ordering.
**TIP5**:When comparing fine-grained image classifiers that share a pretrained CNN backbone but differ mainly in augmentation vs. class-weighting, use a small pretrained backbone (e.g., ResNet-18) in the pilot and evaluate both options with identical, non-augmented validation transforms because this preserves transfer-learning dynamics and avoids eval-time asymmetry that can mis-rank the methods.
**TIP6**:When the comparison is between mild augmentation parameter tweaks in CNN image classification (e.g., rotation angle changes within ~5–15°), use a pretrained ImageNet backbone with a linear-probe plus brief fine-tuning on the last block rather than training from scratch, because stable features make small augmentation effects measurable in short pilots.
**TIP7**:When comparing embedding+clustering (discriminative loss + DBSCAN) to contour+watershed pipelines for nuclei panoptic segmentation, build method-specific supervision and post-processing into the pilot (instance IDs + DBSCAN for the embedding model; explicit contour/border targets + watershed markers for the contour model) because collapsing both into mask-only training or generic post-processing fails to reflect their true relative CV performance.
**TIP8**:When comparing 2D-per-frame+temporal-transformer vs 3D-CNN backbones for video captioning, build the vocabulary from the full training captions and evaluate smoothed BLEU on at least ~80 validation videos because small-vocab, tiny-val BLEU is too high-variance to reliably rank these temporal inductive biases.
**TIP9**:When comparing a late-fusion transformer against a cross-attention CNN/LSTM on binary audio–text QA with noticeable label skew (>55/45), build a class-balanced train/validation subsample and report the mean over ≥2 random stratified splits because label priors and seed variance can make both models tie near the majority baseline on small, imbalanced splits.
**TIP10**:When comparing a retrieval-augmented captioner (e.g., soft-NN over video-level text with confidence gating) against a pure TF-IDF classifier, keep the base classifier identical in the pilot and only toggle the retrieval/gating path because isolating the differentiator prevents proxy noise from different learners from flipping the ranking.



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
