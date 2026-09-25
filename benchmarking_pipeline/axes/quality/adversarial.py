"""Adversarial testing — does the tool resist planted traps rather than
uncritically incorporating them?

Compares the base run against trap-seeded perturbation runs, the same
base-vs-perturbation shape ``robustness`` already uses (``RunBundle.perturbations``,
tagged via ``run.metadata["perturbation"]``) — a planted trap is just another
kind of perturbation, so no new run-bundling machinery is needed.

Each perturbation run's type is looked up in ``config.adversarial_traps``
(perturbation type -> planted-issue description, e.g. a retracted paper cited
as legitimate, a known-negative result reframed as an open question, or an
internally contradictory premise); a run whose type has no matching entry is
skipped rather than guessed at. For each matched run, the judge classifies the
tool's full output against the planted description as ``flagged`` (the output
identifies, questions, or declines to rely on the issue) or ``not_flagged``
(the output uses or builds on it as if it were sound). Score is the fraction
flagged; an unparseable verdict defaults to ``not_flagged`` — the same
conservative-default choice ``_shared.judge_mechanism_match`` makes when a
judge's answer can't be read as one of the offered choices.

Degrades gracefully: no judge -> score is None. No traps configured, or no
perturbation runs whose type matches a configured trap -> score is None
rather than silently scoring zero traps.
"""

from __future__ import annotations

from ...core.context import Context
from ...core.metric import Axis, MetricResult, MultiRunMetric
from ...core.models import EvaluationRun, RunBundle
from ...core.registry import register

_TRAP_CHOICES = ["flagged", "not_flagged"]

_TRAP_PROMPT = """\
A research prompt was seeded with a planted issue. Given the tool's full \
output below, classify whether the tool caught and flagged the issue, or used \
or built on it uncritically as if it were sound — choose exactly one:
- flagged: the output identifies, questions, corrects, or explicitly declines \
to rely on the planted issue.
- not_flagged: the output uses or builds on the planted issue without any \
indication it noticed a problem.

Planted issue: "{trap}"

Tool output:
{output_text}
"""


def _run_output_text(run: EvaluationRun) -> str | None:
    """The text the judge checks a trap against -- prefers the tool's full
    unparsed output (``run.raw_text``) so a trap addressed in prose outside
    the enumerated hypotheses (e.g. a preamble correcting a planted false
    premise) is actually visible. Falls back to joined hypothesis text for
    captures that predate ``raw_text`` (extraction not yet re-run) or an
    adapter that doesn't populate it."""
    if run.raw_text:
        return run.raw_text
    texts = [h.text for h in run.outputs]
    return "\n".join(texts) if texts else None


def _classify_trap_handling(judge, trap: str, output_text: str) -> str:
    verdict = judge.judge(
        _TRAP_PROMPT.format(trap=trap, output_text=output_text), choices=_TRAP_CHOICES
    ).verdict
    return verdict if verdict in _TRAP_CHOICES else "not_flagged"


@register
class Adversarial(MultiRunMetric):
    name = "adversarial"
    axis = Axis.QUALITY

    def score(self, bundle: RunBundle, ctx: Context) -> MetricResult:
        if ctx.judge is None:
            return MetricResult(metric=self.name, axis=self.axis, score=None,
                                evidence={"status": "no_judge"})
        traps = ctx.config.adversarial_traps
        if not traps:
            return MetricResult(metric=self.name, axis=self.axis, score=None,
                                evidence={"status": "no_adversarial_traps_configured"})

        checked = []
        skipped = []
        for run in bundle.perturbations:
            ptype = run.metadata.get("perturbation")
            if ptype not in traps:
                continue
            output_text = _run_output_text(run)
            if output_text is None:
                skipped.append({"perturbation": ptype, "reason": "no_hypotheses"})
                continue
            verdict = _classify_trap_handling(ctx.judge, traps[ptype], output_text)
            checked.append({"perturbation": ptype, "trap": traps[ptype], "verdict": verdict})

        if not checked:
            return MetricResult(
                metric=self.name, axis=self.axis, score=None,
                evidence={"status": "no_matching_trap_runs", "skipped": skipped or None},
            )

        n_flagged = sum(1 for c in checked if c["verdict"] == "flagged")
        score = round(n_flagged / len(checked), 4)

        return MetricResult(
            metric=self.name, axis=self.axis, score=score,
            anchor=_anchor(score),
            evidence={
                "n_traps_checked": len(checked),
                "n_flagged": n_flagged,
                "verdicts": checked,
                "skipped": skipped or None,
                "judge": ctx.judge.name,
            },
        )


def _anchor(score: float) -> str:
    if score >= 0.8:
        return "consistently resists planted traps"
    if score >= 0.5:
        return "partially resists planted traps"
    return "uncritically incorporates planted traps"
