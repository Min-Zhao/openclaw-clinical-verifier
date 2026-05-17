from __future__ import annotations

from .rubric import DIMENSIONS

Decision = str


def decide(scores: dict[str, int]) -> Decision:
    """Map rubric scores to APPROVE, FLAG, or BLOCK."""
    decision, _worst_dimension = decide_with_worst(scores)
    return decision


def decide_with_worst(scores: dict[str, int]) -> tuple[Decision, str | None]:
    """Map rubric scores to a decision and the dimension that drove it."""
    worst_decision: Decision = "APPROVE"
    worst_dimension: str | None = None
    worst_score = 6

    for dimension in DIMENSIONS:
        score = scores[dimension.key]
        if score <= dimension.block_threshold:
            level: Decision = "BLOCK"
        elif score <= dimension.flag_threshold:
            level = "FLAG"
        else:
            level = "APPROVE"

        if _severity(level) > _severity(worst_decision) or (
            level == worst_decision and score < worst_score
        ):
            worst_decision = level
            worst_dimension = dimension.key
            worst_score = score

    return worst_decision, worst_dimension


def _severity(decision: Decision) -> int:
    return {"APPROVE": 0, "FLAG": 1, "BLOCK": 2}[decision]
