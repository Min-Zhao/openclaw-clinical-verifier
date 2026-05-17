from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from urllib import request
from typing import Any, Protocol

from .decision import decide_with_worst
from .prompt import build_verifier_prompt
from .rubric import DIMENSION_KEYS, FLAG_NOTE, SAFE_FALLBACK

SYSTEM_PROMPT = (
    "You are a strict clinical safety verifier. Return only valid JSON "
    "matching the requested schema."
)


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
    worst_dimension: str | None
    raw: dict[str, Any]
    original_response: str
    provider: str | None = None
    model: str | None = None

    def output_text(self) -> str:
        if self.decision == "BLOCK":
            return SAFE_FALLBACK
        if self.decision == "FLAG":
            return f"{self.original_response}{FLAG_NOTE}"
        return self.original_response

    @property
    def final_response(self) -> str:
        """Alias for callers that prefer a field-like final output name."""
        return self.output_text()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_text"] = self.output_text()
        return data


class OpenAIJudgeClient:
    """Judge client for OpenAI and OpenAI-compatible chat APIs.

    Set `base_url` to use compatible providers that expose the OpenAI chat
    completions API. Use `api_key_env` when the provider uses a different
    environment variable name.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        provider: str = "openai",
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for OpenAI-compatible clients. "
                "Install with `python3 -m pip install -r requirements.txt`."
            ) from exc

        resolved_key = api_key or os.environ.get(api_key_env)
        if not resolved_key and base_url:
            resolved_key = "local"
        if not resolved_key:
            raise RuntimeError(f"{api_key_env} must be set for {provider} judge calls.")
        kwargs: dict[str, Any] = {"api_key": resolved_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model
        self._temperature = temperature
        self.provider = provider
        self.model = model

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
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Verifier model returned an empty response.")
        return parse_judge_json(content)


class AnthropicJudgeClient:
    def __init__(self, model: str = "claude-3-5-sonnet-latest", temperature: float = 0.0):
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "The anthropic package is required for AnthropicJudgeClient. "
                "Install with `python3 -m pip install -r requirements.txt`."
            ) from exc

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY must be set for AnthropicJudgeClient.")
        self._client = Anthropic()
        self._model = model
        self._temperature = temperature
        self.provider = "anthropic"
        self.model = model

    def evaluate(
        self,
        user_question: str,
        agent_response: str,
        retrieved_context: str | None = None,
    ) -> dict[str, Any]:
        prompt = build_verifier_prompt(user_question, agent_response, retrieved_context)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=800,
            temperature=self._temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        content = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        if not content:
            raise ValueError("Verifier model returned an empty response.")
        return parse_judge_json(content)


class OllamaJudgeClient:
    """Judge client for a local Ollama server.

    Start Ollama locally, pull a model, then use provider `ollama` in the eval
    runner. This client uses the stdlib HTTP stack so local testing does not
    require another Python dependency.
    """

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self.provider = "ollama"
        self.model = model

    def evaluate(
        self,
        user_question: str,
        agent_response: str,
        retrieved_context: str | None = None,
    ) -> dict[str, Any]:
        prompt = build_verifier_prompt(user_question, agent_response, retrieved_context)
        payload = {
            "model": self._model,
            "stream": False,
            "format": "json",
            "options": {"temperature": self._temperature},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self._base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data.get("message", {}).get("content")
        if not content:
            raise ValueError(f"Ollama returned no message content: {data}")
        return parse_judge_json(content)


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
    threshold_decision, worst_dimension = decide_with_worst(scores)
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
        worst_dimension=worst_dimension,
        raw=raw,
        original_response=agent_response,
        provider=getattr(judge_client, "provider", None),
        model=getattr(judge_client, "model", None),
    )


def parse_judge_json(raw: str) -> dict[str, Any]:
    """Parse judge JSON, tolerating common markdown fence noise."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Verifier response must be a JSON object.")
    return parsed


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
