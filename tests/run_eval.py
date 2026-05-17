#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verifier import (
    AnthropicJudgeClient,
    FixtureJudgeClient,
    OllamaJudgeClient,
    OpenAIJudgeClient,
    verify,
)


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_client(
    provider: str,
    cases: list[dict],
    model: str,
    base_url: str | None,
    api_key_env: str,
):
    if provider == "fixture":
        fixtures = {
            case["case_id"]: case["fixture_verifier_output"]
            for case in cases
        }
        return FixtureJudgeClient(fixtures)
    if provider == "openai":
        return OpenAIJudgeClient(model=model)
    if provider == "anthropic":
        return AnthropicJudgeClient(model=model)
    if provider == "ollama":
        return OllamaJudgeClient(model=model, base_url=base_url or "http://localhost:11434")
    if provider == "openai-compatible":
        if not base_url:
            raise ValueError("--base-url is required for --provider openai-compatible")
        return OpenAIJudgeClient(
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            provider="openai-compatible",
        )
    raise ValueError(f"Unknown provider: {provider}")


def save_result(output_dir: Path, case: dict, result) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": case["case_id"],
        "name": case["name"],
        "expected_decision": case["expected_decision"],
        "actual_decision": result.decision,
        "decision_reason": result.decision_reason,
        "scores": result.scores,
        "justifications": result.justifications,
        "worst_dimension": result.worst_dimension,
        "provider": result.provider,
        "model": result.model,
        "output_text": result.output_text(),
        "raw": result.raw,
    }
    path = output_dir / f"case_{case['case_id']}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def print_table(rows: list[dict]) -> None:
    headers = ["case_id", "expected", "actual", "pass"]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in rows))
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    for row in rows:
        print("  ".join(str(row[header]).ljust(widths[header]) for header in headers))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run clinical verifier eval cases.")
    parser.add_argument("--cases", type=Path, default=ROOT / "tests" / "test_cases.json")
    parser.add_argument(
        "--provider",
        choices=["fixture", "openai", "anthropic", "ollama", "openai-compatible"],
        default="fixture",
    )
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument(
        "--base-url",
        help=(
            "Base URL for Ollama or OpenAI-compatible providers. "
            "Ollama defaults to http://localhost:11434."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key for openai-compatible.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N cases. Useful for slower local models.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tests" / "results")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    model = args.model
    if args.provider == "anthropic" and model == "gpt-4.1-mini":
        model = "claude-3-5-sonnet-latest"
    if args.provider == "ollama" and model == "gpt-4.1-mini":
        model = "llama3.1"
    client = build_client(args.provider, cases, model, args.base_url, args.api_key_env)
    rows = []

    for case in cases:
        retrieved_context = case["case_id"] if args.provider == "fixture" else None
        result = verify(
            user_question=case["user_question"],
            agent_response=case["agent_response"],
            judge_client=client,
            retrieved_context=retrieved_context,
        )
        save_result(args.output_dir, case, result)
        passed = result.decision == case["expected_decision"]
        rows.append(
            {
                "case_id": case["case_id"],
                "expected": case["expected_decision"],
                "actual": result.decision,
                "pass": "yes" if passed else "no",
            }
        )

    print_table(rows)
    failures = [row for row in rows if row["pass"] != "yes"]
    print()
    print(f"Saved JSON results to {args.output_dir}")
    if failures:
        print(f"{len(failures)} case(s) failed.")
        return 1
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
