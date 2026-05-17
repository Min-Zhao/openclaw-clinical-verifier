from __future__ import annotations

from .rubric import DIMENSIONS

Decision = str


def decide(scores: dict[str, int]) -> Decision:
    """Map rubric scores to APPROVE, FLAG, or BLOCK."""
    has_flag = False
    for dimension in DIMENSIONS:
        score = scores[dimension.key]
        if score <= dimension.block_threshold:
            return "BLOCK"
        if score <= dimension.flag_threshold:
            has_flag = True
    return "FLAG" if has_flag else "APPROVE"
