from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Protocol

from .decision import decide
from .prompt import build_verifier_prompt
from .rubric import DIMENSION_KEYS, FLAG_NOTE, SAFE_FALLBACK


class JudgeClient(Protocol):
    def evaluate(
        self,
        user_question: str,
        agent_response: str,
        retrieved_context: str | None = None,
    ) -> dict[str, Any]:
        """Return parsed verifier JSON."""


@dataclass(frozen=True)
class VerifierResult:
    scores: dict[str, int]
    justifications: dict[str, str]
    decision: str
    decision_reason: str
    raw: dict[str, Any]
    original_response: str

    def output_text(self) -> str:
        if self.decision == "BLOCK":
            return SAFE_FALLBACK
        if self.decision == "FLAG":
            return f"{self.original_response}{FLAG_NOTE}"
        return self.original_response


class OpenAIJudgeClient:
    def __init__(self, model: str = "gpt-4.1-mini", temperature: float = 0.0):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for OpenAIJudgeClient. "
                "Install with `python3 -m pip install -r requirements.txt`."
            ) from exc

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY must be set for OpenAIJudgeClient.")
        self._client = OpenAI()
        self._model = model
        self._temperature = temperature

    def evaluate(
        self,
        user_question: str,
        agent_response: str,
        retrieved_context: str | None = None,
    ) -> dict[str, Any]:
        prompt = build_verifier_prompt(user_question, agent_response, retrieved_context)
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict clinical safety verifier. Return only "
                        "valid JSON matching the requested schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Verifier model returned an empty response.")
        return json.loads(content)


class FixtureJudgeClient:
    """Deterministic judge for README/demo evals.

    It reads expected verifier JSON embedded in each test case. This keeps the
    safety harness demonstrable without external API calls and without making
    tests depend on current model behavior.
    """

    def __init__(self, fixtures_by_id: dict[str, dict[str, Any]]):
        self._fixtures_by_id = fixtures_by_id

    def evaluate(
        self,
        user_question: str,
        agent_response: str,
        retrieved_context: str | None = None,
    ) -> dict[str, Any]:
        case_id = retrieved_context or ""
        if case_id not in self._fixtures_by_id:
            raise KeyError(f"No fixture verifier output for case id {case_id!r}")
        return self._fixtures_by_id[case_id]


def verify(
    user_question: str,
    agent_response: str,
    judge_client: JudgeClient,
    retrieved_context: str | None = None,
) -> VerifierResult:
    raw = judge_client.evaluate(user_question, agent_response, retrieved_context)
    scores = _validate_scores(raw.get("scores"))
    justifications = _validate_justifications(raw.get("justifications"))
    threshold_decision = decide(scores)
    model_decision = raw.get("decision")

    # Thresholds are authoritative. The model can explain; code gates.
    decision = threshold_decision
    if model_decision and model_decision != threshold_decision:
        decision_reason = (
            f"Model proposed {model_decision}, but threshold logic requires "
            f"{threshold_decision}."
        )
    else:
        decision_reason = str(raw.get("decision_reason") or _default_reason(decision))

    return VerifierResult(
        scores=scores,
        justifications=justifications,
        decision=decision,
        decision_reason=decision_reason,
        raw=raw,
        original_response=agent_response,
    )


def _validate_scores(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError("Verifier response must include a scores object.")

    scores: dict[str, int] = {}
    for key in DIMENSION_KEYS:
        if key not in value:
            raise ValueError(f"Missing score for dimension {key!r}.")
        score = value[key]
        if not isinstance(score, int) or score < 1 or score > 5:
            raise ValueError(f"Score for {key!r} must be an integer from 1 to 5.")
        scores[key] = score
    return scores


def _validate_justifications(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("Verifier response must include a justifications object.")

    justifications: dict[str, str] = {}
    for key in DIMENSION_KEYS:
        text = value.get(key)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing justification for dimension {key!r}.")
        justifications[key] = text.strip()
    return justifications


def _default_reason(decision: str) -> str:
    if decision == "BLOCK":
        return "At least one safety rubric dimension crossed a block threshold."
    if decision == "FLAG":
        return "At least one safety rubric dimension crossed a flag threshold."
    return "All rubric dimensions passed the approval thresholds."
