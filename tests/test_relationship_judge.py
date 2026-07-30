"""Tests for the shared pairwise relationship-judging utility.

This is the primitive both logical_consistency (claim vs. claim) and the
planned citation support-check (source vs. claim) build on, so it gets its
own coverage independent of either consumer.
"""

from __future__ import annotations

from benchmarking_pipeline.services.llm_judge import Judgement
from benchmarking_pipeline.services.relationship_judge import judge_relationship


class RecordingJudge:
    """Records the last prompt/choices it was asked about; returns a fixed verdict."""

    name = "recording"

    def __init__(self, verdict: str):
        self.verdict = verdict
        self.last_prompt: str | None = None
        self.last_choices: list[str] | None = None

    def judge(self, prompt: str, *, choices: list[str] | None = None) -> Judgement:
        self.last_prompt = prompt
        self.last_choices = choices
        return Judgement(verdict=self.verdict, confidence=1.0)


def test_prompt_includes_both_labelled_texts_and_instructions():
    judge = RecordingJudge(verdict="supports")
    judge_relationship(
        judge,
        label_a="Source", text_a="The paper says X.",
        label_b="Claim", text_b="The hypothesis claims Y.",
        instructions="Choose supports, contradicts, or neutral.",
        choices=["supports", "contradicts", "neutral"],
    )
    assert "Source" in judge.last_prompt
    assert "The paper says X." in judge.last_prompt
    assert "Claim" in judge.last_prompt
    assert "The hypothesis claims Y." in judge.last_prompt
    assert "Choose supports, contradicts, or neutral." in judge.last_prompt
    assert judge.last_choices == ["supports", "contradicts", "neutral"]


def test_valid_verdict_is_returned_verbatim():
    judge = RecordingJudge(verdict="contradicts")
    result = judge_relationship(
        judge, label_a="A", text_a="a", label_b="B", text_b="b",
        instructions="...", choices=["supports", "contradicts", "neutral"],
    )
    assert result == "contradicts"


def test_unrecognized_verdict_falls_back_to_last_choice():
    judge = RecordingJudge(verdict="something the judge made up")
    result = judge_relationship(
        judge, label_a="A", text_a="a", label_b="B", text_b="b",
        instructions="...", choices=["supports", "contradicts", "neutral"],
    )
    assert result == "neutral"  # last entry in choices, the documented fallback convention
