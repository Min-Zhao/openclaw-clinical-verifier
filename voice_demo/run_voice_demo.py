#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import (
    HealthcareVoicePipeline,
    OllamaReasoner,
    OpenAICompatibleReasoner,
    Pyttsx3Speaker,
    WhisperTranscriber,
)


def build_reasoner(args):
    if args.llm_provider == "ollama":
        return OllamaReasoner(model=args.llm_model, base_url=args.base_url)
    if args.llm_provider == "openai":
        return OpenAICompatibleReasoner(model=args.llm_model)
    if args.llm_provider == "openai-compatible":
        return OpenAICompatibleReasoner(
            model=args.llm_model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
        )
    raise ValueError(f"Unsupported LLM provider: {args.llm_provider}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a tiny Whisper -> LLM -> pyttsx3 voice healthcare Q&A demo."
    )
    parser.add_argument("audio", type=Path, help="Input audio file for Whisper ASR.")
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument(
        "--llm-provider",
        choices=["ollama", "openai", "openai-compatible"],
        default="ollama",
    )
    parser.add_argument("--llm-model", default="llama3.1")
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama base URL or OpenAI-compatible API base URL.",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--tts-output", type=Path, help="Optional path to write TTS audio.")
    parser.add_argument("--tts-rate", type=int)
    args = parser.parse_args()

    pipeline = HealthcareVoicePipeline(
        transcriber=WhisperTranscriber(model_name=args.whisper_model),
        reasoner=build_reasoner(args),
        speaker=Pyttsx3Speaker(rate=args.tts_rate),
    )
    result = pipeline.run(args.audio, args.tts_output)

    print("Transcript:")
    print(result.transcript)
    print()
    print("Answer:")
    print(result.answer)
    if result.spoken_output:
        print()
        print(f"TTS audio saved to {result.spoken_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
