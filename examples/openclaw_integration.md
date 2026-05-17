# OpenClaw integration sketch

This repo is a sidecar verifier that can wrap any agent response. For OpenClaw,
the clean integration point is immediately after the primary agent produces a
draft and before the channel adapter sends the message to the user.

## Flow

```text
OpenClaw channel input
  -> primary OpenClaw agent session
  -> draft response
  -> verifier.verify(...)
  -> APPROVE: send draft
  -> FLAG: send draft plus safety note
  -> BLOCK: replace with safe fallback
```

## Pseudocode

```python
from verifier import OllamaJudgeClient, OpenAIJudgeClient, verify

# Hosted provider:
# judge = OpenAIJudgeClient(model="gpt-4.1-mini")

# Local provider for development:
judge = OllamaJudgeClient(model="llama3.1")

def before_send_to_user(user_question, draft_response, retrieved_context=None):
    result = verify(
        user_question=user_question,
        agent_response=draft_response,
        retrieved_context=retrieved_context,
        judge_client=judge,
    )

    audit_log.write(
        {
            "decision": result.decision,
            "scores": result.scores,
            "justifications": result.justifications,
            "decision_reason": result.decision_reason,
            "worst_dimension": result.worst_dimension,
            "provider": result.provider,
            "model": result.model,
        }
    )

    return result.output_text()
```

## Production notes

- Treat threshold logic as code-owned, not model-owned. The model emits scores
  and rationales; the sidecar decides.
- Log scores, justifications, decision, model id, prompt version, and retrieved
  source ids for auditability.
- Use `FixtureJudgeClient` for deterministic tests, `OllamaJudgeClient` for
  local model testing, and hosted clients only when you need production-like
  provider behavior.
- Use task-specific verifiers for higher-risk workflows: medication guidance,
  symptom triage, scheduling, benefits, and care-plan follow-up.
- For voice latency, run a lightweight red-flag classifier before the full
  verifier and reserve blocking synchronous verification for high-risk turns.
