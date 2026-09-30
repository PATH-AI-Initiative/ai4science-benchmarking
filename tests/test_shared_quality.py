"""Tests for ``axes.quality._shared.hypothesis_text``.

Motivating bug (found against a real Biomni capture): extraction often
leaves ``Hypothesis.text`` as just a short title (e.g. "H1 -- Force the heme
feed"), with the real content living in its claims. reproducibility/
robustness's ``top_hypothesis`` mode used to embed/judge that bare title
directly, which is why two runs naming the exact same drug combination could
still be judged "different_mechanism" -- the judge only ever saw a handful of
words, never the reasoning behind them.
"""

from __future__ import annotations

from benchmarking_pipeline.axes.quality._shared import hypothesis_text
from benchmarking_pipeline.core.models import Claim, Hypothesis


def test_no_claims_returns_bare_text():
    hyp = Hypothesis(id="h1", text="H1 -- Force the heme feed", rank=1)
    assert hypothesis_text(hyp) == "H1 -- Force the heme feed"


def test_claims_are_appended_to_the_title():
    hyp = Hypothesis(
        id="h1", text="H1 -- Force the heme feed", rank=1,
        claims=[
            Claim(text="Blocking pathway X forces compensatory hemoglobin digestion."),
            Claim(text="This restores heme-mediated artemisinin activation."),
        ],
    )
    result = hypothesis_text(hyp)
    assert result.startswith("H1 -- Force the heme feed. ")
    assert "Blocking pathway X forces compensatory hemoglobin digestion." in result
    assert "This restores heme-mediated artemisinin activation." in result


def test_empty_claims_list_is_same_as_no_claims():
    hyp = Hypothesis(id="h1", text="A title only", rank=1, claims=[])
    assert hypothesis_text(hyp) == "A title only"
