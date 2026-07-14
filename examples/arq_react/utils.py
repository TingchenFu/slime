import json
import re
import os


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


def select_read_only_overlays(overlays: list[str]) -> list[str]:
    mode = os.environ.get("ARQ_READ_ONLY_OVERLAYS", "safe").strip()
    lowered = mode.lower()
    if lowered in {"", "safe"}:
        return [overlay for overlay in overlays if "patch-gpu-memory-limit" in overlay]
    if lowered in {"none", "0", "false", "no"}:
        return []
    if lowered in {"native", "all", "upstream"}:
        return overlays
    return [part for part in mode.replace(",", ":").split(":") if part]
