"""
Shared helpers for all specialist agents.
Each agent receives a small dict of latest numbers and returns:
  {signal: -1..+1, confidence: 0..1, predicted_direction: "up"|"down",
   rationale: str, key_numbers: dict}
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

SIGNAL_SCHEMA = {
    "signal":              "float in [-1.0, +1.0]; -1=strong bearish, +1=strong bullish",
    "confidence":          "float in [0.0, 1.0]",
    "predicted_direction": '"up" or "down"',
    "rationale":           "one sentence max",
    "key_numbers":         "dict of the 2-3 most important numbers used",
}


def _format_schema() -> str:
    lines = [f'  "{k}": {v}' for k, v in SIGNAL_SCHEMA.items()]
    return "{\n" + ",\n".join(lines) + "\n}"


def build_system_prompt(agent_name: str, domain: str) -> str:
    return (
        f"You are {agent_name}, a quantitative analyst specializing in {domain}. "
        f"You analyze the provided data and return ONLY valid JSON matching this schema exactly:\n"
        f"{_format_schema()}\n"
        "Do not add any other fields. Be concise. No markdown, just the JSON object."
    )


def validate_signal(data: Any) -> dict[str, Any]:
    """
    Validate and clamp a signal dict from an LLM response.
    Returns a sanitized dict or raises ValueError.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data)}")

    signal     = float(data.get("signal", 0.0))
    confidence = float(data.get("confidence", 0.5))
    direction  = str(data.get("predicted_direction", "up")).lower()
    rationale  = str(data.get("rationale", ""))[:300]
    key_nums   = data.get("key_numbers", {})
    if not isinstance(key_nums, dict):
        key_nums = {}

    # Clamp
    signal     = max(-1.0, min(1.0, signal))
    confidence = max(0.0,  min(1.0, confidence))
    if direction not in ("up", "down"):
        direction = "up" if signal >= 0 else "down"

    return {
        "signal":              round(signal, 3),
        "confidence":          round(confidence, 3),
        "predicted_direction": direction,
        "rationale":           rationale,
        "key_numbers":         key_nums,
    }
