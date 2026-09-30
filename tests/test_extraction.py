"""Tests for extract_hypotheses's two-pass call structure.

A single structured-output call asked to both list every hypothesis AND
decompose each one's claims was found to silently under-deliver on documents
with many rich hypotheses (a JSON-schema-constrained model can hit its output
budget and just close the array early, with no error). extract_hypotheses now
makes a cheap list-only call followed by one focused claims call per
hypothesis -- these tests pin that call structure with a fake chat client
rather than hitting a real model."""

from __future__ import annotations

import json

from benchmarking_pipeline.io.extraction import (
    _kaimen_hyp_rank_order,
    extract_hypotheses,
    extract_kaimen_session,
)


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


def test_kaimen_rank_order_reads_first_occurrence_of_each_hyp_id():
    overview = (
        "## Direction 1\nHypothesis [H-hyp_bbbb] is promising.\n"
        "## Direction 2\nHypothesis [H-hyp_aaaa] is also interesting, see also [H-hyp_bbbb].\n"
    )
    assert _kaimen_hyp_rank_order(overview) == {"hyp_bbbb": 1, "hyp_aaaa": 2}


def test_kaimen_rank_order_is_decoration_agnostic():
    """Regression test: a real overview.md cited hypotheses as plain
    `hyp_XXXX` in backticks, with no `[H-...]` wrapper -- the old regex
    required that exact wrapper, found zero matches, and every hypothesis in
    that session silently fell back to the same rank."""
    overview = (
        "Hypothesis `hyp_c1bd9030b8e2b651` posits that...\n"
        "Hypothesis `hyp_2173b6818fe73605` reflects...\n"
    )
    assert _kaimen_hyp_rank_order(overview) == {
        "hyp_c1bd9030b8e2b651": 1, "hyp_2173b6818fe73605": 2,
    }


def _write_kaimen_session(session_dir):
    (session_dir / "final").mkdir(parents=True)
    (session_dir / "hypotheses").mkdir()
    (session_dir / "reviews").mkdir()

    (session_dir / "final" / "overview.md").write_text(
        "## Direction 1\nHypothesis [H-hyp_bbbb] is promising.\n"
        "## Direction 2\nHypothesis [H-hyp_aaaa] is also interesting.\n"
    )
    (session_dir / "hypotheses" / "hyp_aaaa.json").write_text(json.dumps({
        "strategy": "literature",
        "record": {
            "title": "Title A", "statement": "Statement A", "mechanism": "Mechanism A text",
            "entities": ["GeneX"], "anticipated_outcomes": "", "novelty_argument": "",
            "citations": [{"url": "http://a", "title": "Paper A", "excerpt": "..."}],
        },
    }))
    (session_dir / "hypotheses" / "hyp_bbbb.json").write_text(json.dumps({
        "strategy": "literature",
        "record": {
            "title": "Title B", "statement": "Statement B", "mechanism": "Mechanism B text",
            "entities": [], "anticipated_outcomes": "", "novelty_argument": "", "citations": [],
        },
    }))
    (session_dir / "reviews" / "rev_1.json").write_text(json.dumps({
        "hypothesis_id": "hyp_aaaa",
        "record": {
            "novelty": 0.8, "feasibility": 0.6, "notes": "solid",
            "assumptions": [
                {"assumption": "risky one", "plausibility": "uncertain", "rationale": "..."},
                {"assumption": "safe one", "plausibility": "plausible", "rationale": "..."},
            ],
        },
    }))
    # hyp_bbbb intentionally has no review file -- should still parse fine.


def test_extract_kaimen_session_maps_fields_and_decomposes_mechanism(tmp_path):
    session = tmp_path / "base"
    _write_kaimen_session(session)
    chat = _FakeChat(None, {
        "Statement A": _claims_response("decomposed claim A"),
        "Statement B": _claims_response("decomposed claim B"),
    })

    result = extract_kaimen_session(session, chat)

    assert result["raw_text"].startswith("## Direction 1")
    hyps = {h["id"]: h for h in result["hypotheses"]}
    assert hyps["hyp_bbbb"]["rank"] == 1
    assert hyps["hyp_aaaa"]["rank"] == 2
    assert hyps["hyp_aaaa"]["text"] == "Title A: Statement A"

    # The synthetic statement-level claim carries citations + entities...
    a_claims = hyps["hyp_aaaa"]["claims"]
    assert a_claims[0]["text"] == "Statement A"
    assert a_claims[0]["references"][0]["raw"] == "http://a"
    assert a_claims[0]["entities"][0]["name"] == "GeneX"
    # ...and the LLM-decomposed mechanism claim(s) follow it.
    assert a_claims[1]["text"] == "decomposed claim A"

    sr = hyps["hyp_aaaa"]["self_reported"]
    assert sr["novelty_tier"] == "0.8"
    assert sr["feasibility_tier"] == "0.6"
    assert sr["key_risks"] == ["risky one"]

    # No review file for hyp_bbbb -- self_reported stays blank, not an error.
    assert hyps["hyp_bbbb"]["self_reported"]["novelty_tier"] == ""


def test_extract_kaimen_session_skips_claims_decomposition_when_no_argument_text(tmp_path):
    """A hypothesis with no mechanism/anticipated_outcomes/novelty_argument
    text shouldn't trigger a claims call at all -- nothing to decompose."""
    session = tmp_path / "base"
    _write_kaimen_session(session)
    data = json.loads((session / "hypotheses" / "hyp_bbbb.json").read_text())
    data["record"]["mechanism"] = ""
    (session / "hypotheses" / "hyp_bbbb.json").write_text(json.dumps(data))
    chat = _FakeChat(None, {"Statement A": _claims_response("decomposed claim A")})

    result = extract_kaimen_session(session, chat)

    hyps = {h["id"]: h for h in result["hypotheses"]}
    assert len(hyps["hyp_bbbb"]["claims"]) == 1  # just the synthetic statement claim
    assert len(chat.calls) == 1  # only hyp_aaaa triggered a claims call
