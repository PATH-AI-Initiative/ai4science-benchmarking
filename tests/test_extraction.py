"""Tests for extract_hypotheses's two-pass call structure.

A single structured-output call asked to both list every hypothesis AND
decompose each one's claims was found to silently under-deliver on documents
with many rich hypotheses (a JSON-schema-constrained model can hit its output
budget and just close the array early, with no error). extract_hypotheses now
makes a cheap list-only call followed by one focused claims call per
hypothesis -- these tests pin that call structure with a fake chat client
rather than hitting a real model."""

from __future__ import annotations

from benchmarking_pipeline.io.extraction import extract_hypotheses


class _FakeChat:
    """Records each call and returns a scripted response keyed by whether the
    request schema is the hypothesis-list schema or the per-hypothesis claims
    schema (distinguished by their top-level required fields)."""

    name = "fake"

    def __init__(self, list_response, claims_response_by_hypothesis):
        self.list_response = list_response
        self.claims_response_by_hypothesis = claims_response_by_hypothesis
        self.calls = []

    def complete(self, *, system, user, schema, max_tokens=4096):
        self.calls.append({"user": user, "schema": schema, "max_tokens": max_tokens})
        if "hypotheses" in schema["required"]:
            return self.list_response
        for hyp_text, response in self.claims_response_by_hypothesis.items():
            if hyp_text in user:
                return response
        raise AssertionError(f"no scripted claims response matches prompt: {user[:200]!r}")


def _claims_response(text="a claim"):
    return {"claims": [{"text": text, "role": "premise", "references": [], "entities": []}]}


def test_makes_one_list_call_plus_one_claims_call_per_hypothesis():
    list_response = {
        "hypotheses": [
            {"rank": 1, "text": "Hypothesis A", "category": "", "self_reported": {
                "novelty_tier": "", "novelty_rationale": "", "feasibility_tier": "",
                "feasibility_rationale": "", "key_risks": [],
            }},
            {"rank": 2, "text": "Hypothesis B", "category": "", "self_reported": {
                "novelty_tier": "", "novelty_rationale": "", "feasibility_tier": "",
                "feasibility_rationale": "", "key_risks": [],
            }},
        ]
    }
    chat = _FakeChat(list_response, {
        "Hypothesis A": _claims_response("claim for A"),
        "Hypothesis B": _claims_response("claim for B"),
    })

    result = extract_hypotheses("raw text here", chat)

    assert len(chat.calls) == 3  # 1 list call + 2 claims calls
    assert len(result["hypotheses"]) == 2
    assert result["hypotheses"][0]["claims"][0]["text"] == "claim for A"
    assert result["hypotheses"][1]["claims"][0]["text"] == "claim for B"


def test_claims_call_is_scoped_to_one_hypothesis_at_a_time():
    """Each claims call's prompt should reference only its own hypothesis's
    text, not the full list -- that's what keeps its output bounded."""
    list_response = {
        "hypotheses": [
            {"rank": 1, "text": "Hypothesis A", "category": "", "self_reported": {
                "novelty_tier": "", "novelty_rationale": "", "feasibility_tier": "",
                "feasibility_rationale": "", "key_risks": [],
            }},
        ]
    }
    chat = _FakeChat(list_response, {"Hypothesis A": _claims_response()})

    extract_hypotheses("the full raw document text", chat)

    # The second call (index 1) is the claims call for Hypothesis A.
    claims_call = chat.calls[1]
    assert "Hypothesis A" in claims_call["user"]
    assert "the full raw document text" in claims_call["user"]


def test_max_tokens_is_forwarded_to_every_call():
    list_response = {"hypotheses": [
        {"rank": 1, "text": "H1", "category": "", "self_reported": {
            "novelty_tier": "", "novelty_rationale": "", "feasibility_tier": "",
            "feasibility_rationale": "", "key_risks": [],
        }},
    ]}
    chat = _FakeChat(list_response, {"H1": _claims_response()})

    extract_hypotheses("raw", chat, max_tokens=123)

    assert all(c["max_tokens"] == 123 for c in chat.calls)
