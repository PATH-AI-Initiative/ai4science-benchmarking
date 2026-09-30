"""Tests for ProgressJudge: delegates correctly, prints progress per call."""

from __future__ import annotations

import io

from benchmarking_pipeline.services.llm_judge import Judgement, ProgressJudge


class FakeJudge:
    name = "fake"

    def __init__(self, verdicts: list[str]):
        self._verdicts = iter(verdicts)

    def judge(self, prompt: str, *, choices=None) -> Judgement:
        return Judgement(verdict=next(self._verdicts), confidence=1.0)


def test_delegates_and_returns_the_real_verdict():
    stream = io.StringIO()
    judge = ProgressJudge(FakeJudge(["supports"]), stream=stream)
    result = judge.judge("prompt")
    assert result.verdict == "supports"


def test_name_passes_through():
    judge = ProgressJudge(FakeJudge(["x"]))
    assert judge.name == "fake"


def test_prints_running_count_without_total():
    stream = io.StringIO()
    judge = ProgressJudge(FakeJudge(["a", "b"]), stream=stream)
    judge.judge("p1")
    judge.judge("p2")
    output = stream.getvalue()
    assert "call 1..." in output
    assert "call 2..." in output
    assert "a" in output and "b" in output


def test_prints_n_of_total_when_total_given():
    stream = io.StringIO()
    judge = ProgressJudge(FakeJudge(["a", "b", "c"]), total=3, stream=stream)
    judge.judge("p1")
    judge.judge("p2")
    output = stream.getvalue()
    assert "call 1/3" in output
    assert "call 2/3" in output
