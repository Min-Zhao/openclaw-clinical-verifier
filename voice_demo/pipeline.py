from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Protocol
from urllib import request


HEALTHCARE_SYSTEM_PROMPT = """You are a careful healthcare Q&A assistant.
Give concise, plain-language information. Do not diagnose. Encourage emergency
care for red-flag symptoms and encourage a clinician for personal medical advice.
"""


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> str:
        """Return recognized text from an audio file."""


class Reasoner(Protocol):
    def answer(self, question: str) -> str:
        """Return the assistant's text answer."""


class Speaker(Protocol):
    def speak(self, text: str, output_path: Path | None = None) -> Path | None:
        """Speak text aloud or write audio to output_path."""


@dataclass(frozen=True)
class PipelineResult:
    transcript: str
    answer: str
    spoken_output: Path | None


class HealthcareVoicePipeline:
    def __init__(self, transcriber: Transcriber, reasoner: Reasoner, speaker: Speaker):
        self._transcriber = transcriber
        self._reasoner = reasoner
        self._speaker = speaker

    def run(self, audio_path: Path, output_path: Path | None = None) -> PipelineResult:
        transcript = self._transcriber.transcribe(audio_path)
        if not transcript:
            raise ValueError("Whisper returned an empty transcript.")
        answer = self._reasoner.answer(transcript)
        if not answer:
            raise ValueError("The LLM returned an empty answer.")
        spoken_output = self._speaker.speak(answer, output_path)
        return PipelineResult(
            transcript=transcript,
            answer=answer,
            spoken_output=spoken_output,
        )


class WhisperTranscriber:
    """Open-source Whisper ASR wrapper.

    The import is intentionally lazy so tests and non-voice verifier workflows
    do not need the heavy Whisper dependency installed.
    """

    def __init__(self, model_name: str = "base"):
        self._model_name = model_name
        self._model: Any | None = None

    def transcribe(self, audio_path: Path) -> str:
        if self._model is None:
            try:
                import whisper
            except ImportError as exc:
                raise RuntimeError(
                    "Install the voice demo dependencies with "
                    "`python3 -m pip install -r requirements-voice.txt`."
                ) from exc
            self._model = whisper.load_model(self._model_name)
        result = self._model.transcribe(str(audio_path))
        return str(result.get("text", "")).strip()


class OpenAICompatibleReasoner:
    """LLM reasoning step for OpenAI-compatible chat completions APIs."""

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.2,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._temperature = temperature

    def answer(self, question: str) -> str:
        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            raise RuntimeError(f"{self._api_key_env} must be set for LLM calls.")
        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": HEALTHCARE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        }
        raw = _post_json(
            f"{self._base_url}/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        return raw["choices"][0]["message"]["content"].strip()


class OllamaReasoner:
    """Local LLM reasoning step through Ollama's chat API."""

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature

    def answer(self, question: str) -> str:
        payload = {
            "model": self._model,
            "stream": False,
            "options": {"temperature": self._temperature},
            "messages": [
                {"role": "system", "content": HEALTHCARE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        }
        raw = _post_json(f"{self._base_url}/api/chat", payload)
        return raw["message"]["content"].strip()


class Pyttsx3Speaker:
    """Simple local TTS wrapper using pyttsx3."""

    def __init__(self, rate: int | None = None):
        self._rate = rate

    def speak(self, text: str, output_path: Path | None = None) -> Path | None:
        try:
            import pyttsx3
        except ImportError as exc:
            raise RuntimeError(
                "Install the voice demo dependencies with "
                "`python3 -m pip install -r requirements-voice.txt`."
            ) from exc
        engine = pyttsx3.init()
        if self._rate is not None:
            engine.setProperty("rate", self._rate)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            engine.save_to_file(text, str(output_path))
        else:
            engine.say(text)
        engine.runAndWait()
        return output_path


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    with request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))
