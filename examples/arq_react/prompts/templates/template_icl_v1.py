ICL_TEMPLATE_V1 = """
# INTRODUCTION:

You are a Kaggle Grandmaster and Lead Data Scientist acting as a **Strategic Evaluator**. Your task is to analyze two candidate machine learning solutions to a Kaggle competition problem and predict which one will achieve a superior 5-fold CV score (accounting for the metric's direction).
Both solutions are designed to train on a public dataset using 5-fold CV and produce predictions for a held-out test set whose ground truth labels are not available. The 5-fold CV metric is used to decide which solution is more promising—either to submit for final testing or to iterate further.
Remember: your goal is NOT to build the final solution or to get a precise 5-fold CV score, but to **predict which solution will achieve a better 5-fold CV score if fully run**.


# TASK DESCRIPTION
```
{task_desc}
```

**Important:** Pay attention to whether higher or lower values indicate better performance (e.g., accuracy is higher-is-better, RMSE is lower-is-better).
**Resources:** {device}
**Time Limit:** {time_limit}
**Note:** Consider whether each solution can feasibly complete within the given resource and time constraints. A solution that cannot finish in time should be considered inferior.


# CANDIDATE SOLUTIONS:
## Option A:
```python
{code_a}
```


## Option B:
```python
{code_b}
```



{examples}


# RESPONSE FORMAT:

Think step by step and provide your reasoning before giving a final answer of A for option A, and B for the option B, inside a \\boxed{{}}, i.e., \\boxed{{A}} or \\boxed{{B}}.


"""
