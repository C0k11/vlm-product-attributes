"""Prompt and output schema for attribute extraction."""
import json
import re

from .labels import COLORS, MATERIALS

ATTRIBUTES = ["product_type", "color", "material"]


def build_instruction(product_types: list[str], title: str | None = None) -> str:
    lines = [
        "You are labeling a product image for an e-commerce catalog.",
        "Return only a JSON object with exactly these keys:",
        f'- "product_type": one of {json.dumps(product_types)}',
        f'- "color": the main color of the product, one of {json.dumps(COLORS)}',
        f'- "material": the main material of the product, one of {json.dumps(MATERIALS)}',
        "Pick the closest option for every key. Do not add any other text.",
    ]
    if title:
        lines.insert(1, f"Product title: {title}")
    return "\n".join(lines)


SHORT_INSTRUCTION = "Return the product_type, color and material of this product as a JSON object."


MAX_TITLE_CHARS = 300


def build_prompt(style: str, product_types: list[str], title: str | None = None) -> str:
    """full: label lists spelled out (needed zero-shot). short: for fine-tuned
    models that have learned the label space; valid values are still enforced by
    the JSON schema at decode time. Titles are cut to MAX_TITLE_CHARS."""
    if title:
        title = " ".join(title.split())[:MAX_TITLE_CHARS]
    if style == "full":
        return build_instruction(product_types, title)
    if style == "short":
        return SHORT_INSTRUCTION if not title else f"Product title: {title}\n{SHORT_INSTRUCTION}"
    raise ValueError(style)


def json_schema(product_types: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "product_type": {"type": "string", "enum": product_types},
            "color": {"type": "string", "enum": COLORS},
            "material": {"type": "string", "enum": MATERIALS},
        },
        "required": ATTRIBUTES,
        "additionalProperties": False,
    }


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_output(text: str) -> tuple[dict | None, str]:
    """Parse model output.

    Returns (parsed, status). status is "strict" when the text is a bare JSON
    object, "fenced" when it only parses after removing a markdown code fence,
    and "invalid" otherwise.
    """
    s = text.strip()
    try:
        obj = json.loads(s)
        return (obj, "strict") if isinstance(obj, dict) else (None, "invalid")
    except json.JSONDecodeError:
        pass
    m = _FENCE.match(s)
    if m:
        try:
            obj = json.loads(m.group(1))
            return (obj, "fenced") if isinstance(obj, dict) else (None, "invalid")
        except json.JSONDecodeError:
            pass
    return None, "invalid"
