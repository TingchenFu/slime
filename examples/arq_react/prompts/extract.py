import json
import re

import black
import yaml


def format_code(code) -> str:
    """Format Python code using Black."""
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        return code


def is_valid_python_script(script):
    """Check if a script is a valid Python script."""
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def extract_code(text):
    """
    Extract Python code blocks from a given text.

    This function searches the input text for Python code blocks enclosed within
    triple backticks (```python ... ```). It extracts these code blocks, formats
    them using Black, and ensures they are valid Python scripts. If no such
    blocks are found, it attempts to extract the entire text as a code block.

    Args:
        text (str): The input text containing potential Python code blocks.

    Returns:
        str: A single string containing all valid, formatted Python code blocks
             extracted from the input text, separated by two newlines.
    """
    parsed_codes = []

    # Regex pattern to find code blocks enclosed in ```python ... ```
    # The (python)? makes the 'python' specifier optional
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    # If no code blocks were found with backticks, try extracting the entire text
    if len(parsed_codes) == 0:
        # Regex pattern to match the entire text as a code block, with optional backticks
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    # Validate each extracted code block and format it
    valid_code_blocks = [format_code(c) for c in parsed_codes if is_valid_python_script(c)]

    if len(valid_code_blocks) == 0:
        return None

    # Combine all valid code blocks into a single string separated by two newlines
    return format_code("\n\n".join(valid_code_blocks))


def extract_yaml_data(text: str) -> dict:
    text = text.strip()

    # 1) First try fenced YAML
    fence_re = re.compile(r"^```(?:yaml)?\s*$", re.M | re.I)
    fences = list(fence_re.finditer(text))
    if len(fences) >= 2:
        opening, closing = fences[0], fences[-1]
        candidate = text[opening.end() : closing.start()].strip()
        parsed = yaml.safe_load(candidate)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"Parsed fenced YAML is not a dict: {parsed!r}")

    # 2) Fallback: treat the whole text as YAML
    parsed = yaml.safe_load(text)
    if isinstance(parsed, dict):
        return parsed
    raise ValueError(f"Can't parse YAML dict from text:\n{text!r}")


def extract_json_data(text: str) -> dict:
    text = text.strip()

    # 1) First try plain JSON directly
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 2) Try fenced JSON blocks: ```json ... ``` or generic ``` ... ```
    fence_re = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.I)
    matches = fence_re.findall(text)
    for candidate in matches:
        try:
            parsed = json.loads(candidate.strip())
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    # 3) Try extracting the outermost JSON object from surrounding text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"Can't parse JSON dict from text:\n{text!r}")
