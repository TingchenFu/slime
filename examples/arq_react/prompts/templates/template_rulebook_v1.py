RULEBOOK_TEMPLATE_V1 = """
# INTRODUCTION:

You are a Kaggle Grandmaster and Lead Data Scientist acting as a **Reflective Analyst**.
You have observed two pilot experiment scripts that were designed to help predict which of two candidate ML solutions achieves a better 5-fold CV score — **without running the full CV**. Each script runs a fast, approximate experiment and writes its findings (evidence) to an output file. A language model then reads this evidence and makes a prediction. The scripts are rated by how accurately their evidence helped the language model make the correct prediction.

Your task is to compare the **more effective script** (higher prediction accuracy score) against the **less effective script** (lower prediction accuracy score) and distill what made the difference into a concise, reusable rule.


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


# INSTRUCTIONS:

Carefully compare the two scripts and their evidence outputs. Note that neither script directly computes the full 5-fold CV score — they each produce approximate, partial evidence (e.g., a single-fold result, a subsampled validation score, an ablation comparison, or any other signal). The quality of a script is judged by how well its evidence guided the language model toward the correct prediction.

Focus on **why the higher-scoring script's evidence was more useful**. Consider factors such as:
- Whether the experiment directly probes the key differentiating factor between the two candidate solutions
- Whether the evidence output is clear, quantitative, and easy for a language model to interpret
- Whether the experimental proxy is reliable (e.g., consistent split, appropriate data size, meaningful metric)
- Whether the output avoids noise, ambiguity, or information that could mislead the prediction


# RESPONSE FORMAT:

First, write your comparative analysis in free-form prose (2-4 sentences explaining the key difference between the two scripts and why the higher-scoring script produced more informative evidence).

Then output a YAML block containing the extracted rules:

```yaml
rules:
  - "<one-sentence rule>"
  - "<one-sentence rule>"
  - "<one-sentence rule>"
```

Requirements for each rule in `rules`:
- Exactly one sentence per entry
- At most three entries total (include only rules genuinely supported by this example)
- Concise and actionable (a practitioner can apply it when writing future pilot scripts)
- General enough to transfer across different ML tasks and competition types
- Grounded in the specific contrast observed above
"""
