RULEBOOK_TEMPLATE_V2 = """
# INTRODUCTION:

You are a Kaggle Grandmaster and Lead Data Scientist acting as a **Reflective Analyst**.
You have observed two pilot experiment scripts that were designed to help predict which of two candidate ML solutions achieves a better 5-fold CV score — **without running the full CV**. Each script runs a fast, approximate experiment and writes its findings (evidence) to an output file. A language model then reads this evidence and makes a prediction. The scripts are rated by how accurately their evidence helped the language model make the correct prediction.

Your task is to compare the **more effective script** (higher prediction accuracy score) against the **less effective script** (lower prediction accuracy score) and distill what made the difference into a **specific, actionable rule** tied to the concrete characteristics of this comparison.


# TASK DESCRIPTION
````
{task_desc}
````

**Important:** Pay attention to whether higher or lower values indicate better performance for this task's metric.


# TWO CANDIDATE ML SOLUTIONS:
## Option A:
```python
{code_a}
```


## Option B:
```python
{code_b}
```


# PILOT EXPERIMENT COMPARISON:

## Higher-Scoring Script (Prediction Accuracy Score: {score_good:.2f})

### Code:
```python
{code_good}
```

### Evidence Output (what the script wrote to `execution.out`):
```
{estimation_good}
```


## Lower-Scoring Script (Prediction Accuracy Score: {score_bad:.2f})

### Code:
```python
{code_bad}
```

### Evidence Output (what the script wrote to `execution.out`):
```
{estimation_bad}
```


# STEP 1 — STRUCTURED DECOMPOSITION (required before writing rules):

Before extracting rules, answer each of the following questions in 1-2 sentences:

**Q1. Primary technical difference:** What is the single most important algorithmic or architectural difference between Option A and Option B? Be concrete (e.g., "Option A uses a pretrained transformer encoder while Option B uses a TF-IDF + logistic regression baseline").

**Q2. Core estimation challenge:** Given that specific difference, what makes it hard to estimate their relative 5-fold CV performance cheaply? (e.g., "The transformer needs sufficient fine-tuning epochs to separate from the baseline, so a 1-epoch proxy severely underestimates its advantage").

**Q3. Why the higher-scoring script succeeded here:** What did it do that was specifically well-suited to this comparison — not just good experimental practice in general, but the right choice given Q1 and Q2?

**Q4. Why the lower-scoring script failed here:** What did it do (or fail to do) that was specifically ill-suited to this comparison — not generic flaws, but the specific mismatch between its design and the nature of this comparison?


# STEP 2 — RULE EXTRACTION:

Based on your analysis in Step 1, extract 1-3 rules. Each rule MUST:

1. **Be conditional** — start with "When [specific condition that describes the type of comparison, task domain, or model class]..."
2. **Prescribe a concrete action** — say what specifically to do in the pilot experiment design
3. **Give a brief rationale** — explain why this action specifically helps given the condition (one clause is enough)
4. **Be falsifiable** — a practitioner should be able to look at a new comparison and decide whether the condition applies

**Rules that will be rejected (do not write these):**
- Rules that apply equally to every possible comparison (e.g., "always include quantitative metrics", "use try-except blocks", "train for sufficient epochs")
- Rules that are standard engineering hygiene rather than experimental design insight
- Rules that do not reference the specific type of model, data modality, or algorithmic contrast observed here

**Self-check before finalizing:** For each rule, ask yourself — "Would this rule give *different* advice on a tabular XGBoost-vs-LightGBM comparison vs. a fine-tuning-vs-scratch NLP comparison?" If the answer is no (it says the same thing regardless), the rule is too generic and must be rewritten or dropped.


# RESPONSE FORMAT:

## Step 1: Structured Decomposition
**Q1. Primary technical difference:** <answer>
**Q2. Core estimation challenge:** <answer>
**Q3. Why the higher-scoring script succeeded:** <answer>
**Q4. Why the lower-scoring script failed:** <answer>

## Step 2: Rules

```yaml
rules:
  - "When [specific condition], [concrete action] because [brief rationale]."
  - "When [specific condition], [concrete action] because [brief rationale]."
```
"""
