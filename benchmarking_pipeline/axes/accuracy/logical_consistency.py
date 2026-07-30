"""Logical consistency: do conclusions follow from premises without contradiction?

Four stages, following Dung's abstract argumentation framework (Dung, 1995,
"On the acceptability of arguments...") for the scoring semantics rather than
an ad hoc heuristic:

1. **Role assignment.** Each claim is tagged premise / mechanistic_step /
   prediction / background_assumption (lazily, via the judge, if not already
   set during extraction). Role determines which pairs matter most and how
   severe a contradiction involving it is.
2. **Edge construction.** For each candidate pair, the judge classifies the
   relationship as one of ``a_entails_b`` / ``b_entails_a`` / ``contradicts`` /
   ``neutral``, via the shared
   :func:`~benchmarking_pipeline.services.relationship_judge.judge_relationship`
   utility (also used by citation support-checking — same "how does text A
   relate to text B" operation, different choice set). Pairs are checked in
   priority order — core-argument combinations (premise/mechanism/prediction)
   before background-vs-background — and ``config.logical_consistency_max_pairs``
   caps the budget if set, dropping lowest-priority pairs first (reported,
   never silently skipped).
3. **Graph assembly and scoring.** ``contradicts`` verdicts become a symmetric
   attack relation; the score is ``|grounded extension| / |claims|`` — the
   standard fixed-point computation, not a bespoke "count the contradictions"
   rule. A claim survives only if every attacker is itself defeated; a plain
   mutual contradiction with no other structure defeats *both* claims (neither
   can be prioritised over the other), and contradiction cycles resolve the
   same way for free, with no separate cycle-detection heuristic needed.
   ``entails`` verdicts are used only for orphan detection (a mechanistic step
   or prediction with no supporting claim) — support edges do not defend
   against attacks here, keeping the score a single, standard formalism rather
   than a bipolar argumentation framework with its less settled semantics.
4. **Diagnostics.** Severity (derived from the roles involved) and orphan
   status are reported per contradiction/claim for human review, but neither
   changes the numeric score — the score is pure contradiction-freeness, per
   the framework's own definition (Figure 2: "the proportion of claims forming
   a contradiction-free subgraph").

Degrades gracefully to ``score=None`` with no judge attached, or trivially to
a perfect score with fewer than two claims to compare.
"""

from __future__ import annotations

from itertools import combinations

from ...core.context import Context
from ...core.metric import Axis, HypothesisMetric, MetricResult
from ...core.models import Claim, ClaimRole, EvaluationRun, Hypothesis
from ...core.registry import register
from ...services.relationship_judge import judge_relationship

_ROLE_CHOICES = [r.value for r in ClaimRole]

_ROLE_CLASSIFICATION_PROMPT = """\
Classify the role this claim plays in a scientific hypothesis's argument structure:
- premise: a foundational fact taken as given, not itself derived from the hypothesis.
- mechanistic_step: an explanation of how or why something happens (the causal mechanism).
- prediction: a testable outcome the hypothesis implies.
- background_assumption: contextual framing, not central to the argument's core logic.

Claim: "{text}"
"""

_EDGE_CHOICES = ["a_entails_b", "b_entails_a", "contradicts", "neutral"]

_EDGE_INSTRUCTIONS = """\
You are checking two claims extracted from the same scientific hypothesis for \
their logical relationship. Choose exactly one:
- a_entails_b: if Claim A is true, Claim B must also be true (A supports/implies B).
- b_entails_a: if Claim B is true, Claim A must also be true (B supports/implies A).
- contradicts: Claim A and Claim B describe the same system/scenario and cannot \
both be true about it at the same time.
- neutral: the claims are unrelated, or related but neither entails nor contradicts the other.

A hypothesis routinely states a gap in current practice ("existing methods \
require X" / "today you must do X") right alongside what its own proposed \
method does instead ("this proposal avoids X by doing Y instead"). That is NOT \
a contradiction, even though it can read as one on the surface (e.g. "you \
currently retrain from scratch" next to "the new method shares a prior across \
cases" sounds opposed but isn't) -- one claim describes the state of the art \
*without* the proposal, the other describes the proposal itself; they are about \
two different scenarios by construction, not the same one. Only choose \
contradicts when both claims are asserted about the same scenario. Default to \
neutral unless the relationship is clear.
"""


def _classify_role(judge, claim: Claim) -> ClaimRole:
    verdict = judge.judge(
        _ROLE_CLASSIFICATION_PROMPT.format(text=claim.text), choices=_ROLE_CHOICES
    ).verdict
    try:
        return ClaimRole(verdict)
    except ValueError:
        # Safe fallback: lowest priority/severity rather than silently critical.
        return ClaimRole.BACKGROUND_ASSUMPTION


def _ensure_roles(claims: list[Claim], judge) -> list[ClaimRole]:
    return [c.role if c.role is not None else _classify_role(judge, c) for c in claims]


def _judge_edge(judge, text_a: str, text_b: str) -> str:
    return judge_relationship(
        judge,
        label_a="Claim A", text_a=text_a,
        label_b="Claim B", text_b=text_b,
        instructions=_EDGE_INSTRUCTIONS,
        choices=_EDGE_CHOICES,
    )


def _pair_priority(role_a: ClaimRole, role_b: ClaimRole) -> int:
    """Lower = checked first (and kept under a pair budget). Core-argument
    combinations (premise/mechanism/prediction) matter most; background-vs-
    background matters least."""
    return sum(1 for r in (role_a, role_b) if r is ClaimRole.BACKGROUND_ASSUMPTION)


_CRITICAL_ROLES = frozenset({ClaimRole.MECHANISTIC_STEP, ClaimRole.PREDICTION})


def _severity(role_a: ClaimRole, role_b: ClaimRole) -> str:
    """How much a contradiction between these two roles should worry a
    reviewer. Diagnostic only — does not affect the score."""
    n_background = sum(1 for r in (role_a, role_b) if r is ClaimRole.BACKGROUND_ASSUMPTION)
    if n_background == 2:
        return "low"
    if n_background == 1:
        return "moderate"
    if {role_a, role_b} <= _CRITICAL_ROLES:
        return "critical"
    return "high"


def _grounded_extension(
    n: int, attackers: dict[int, set[int]]
) -> tuple[set[int], set[int], set[int]]:
    """Dung's grounded extension via standard fixed-point labelling.

    A claim is IN once every attacker is OUT (defended, or unattacked); OUT
    once any attacker is IN. Whatever is left when the labelling stabilises —
    a mutual attack or contradiction cycle with no external defender — is
    UNDEC: genuinely undecidable under the semantics, not a bug, and does not
    count as surviving.
    """
    in_set: set[int] = set()
    out_set: set[int] = set()
    undec = set(range(n))
    changed = True
    while changed:
        changed = False
        for i in list(undec):
            if attackers[i] <= out_set:
                in_set.add(i)
                undec.discard(i)
                changed = True
            elif attackers[i] & in_set:
                out_set.add(i)
                undec.discard(i)
                changed = True
    return in_set, out_set, undec


@register
class LogicalConsistency(HypothesisMetric):
    name = "logical_consistency"
    axis = Axis.ACCURACY

    def score(self, hypothesis: Hypothesis, run: EvaluationRun, ctx: Context) -> MetricResult:
        claims = hypothesis.claims
        if ctx.judge is None:
            return MetricResult(metric=self.name, axis=self.axis, score=None,
                                evidence={"status": "no_judge"})
        if len(claims) < 2:
            return MetricResult(
                metric=self.name, axis=self.axis, score=1.0,
                anchor="trivially consistent",
                evidence={"status": "fewer_than_two_claims", "n_claims": len(claims)},
            )

        roles = _ensure_roles(claims, ctx.judge)

        all_pairs = sorted(
            combinations(range(len(claims)), 2),
            key=lambda p: _pair_priority(roles[p[0]], roles[p[1]]),
        )
        max_pairs = ctx.config.logical_consistency_max_pairs
        checked_pairs = all_pairs if max_pairs is None else all_pairs[:max_pairs]
        n_skipped = 0 if max_pairs is None else len(all_pairs) - len(checked_pairs)

        attackers: dict[int, set[int]] = {i: set() for i in range(len(claims))}
        entailed_by: dict[int, set[int]] = {i: set() for i in range(len(claims))}
        edges = []
        for i, j in checked_pairs:
            verdict = _judge_edge(ctx.judge, claims[i].text, claims[j].text)
            edges.append({"i": i, "j": j, "verdict": verdict})
            if verdict == "contradicts":
                attackers[i].add(j)
                attackers[j].add(i)
            elif verdict == "a_entails_b":
                entailed_by[j].add(i)
            elif verdict == "b_entails_a":
                entailed_by[i].add(j)

        in_set, out_set, undec_set = _grounded_extension(len(claims), attackers)
        score_value = len(in_set) / len(claims)

        contradiction_pairs = [
            {
                "i": i, "j": j,
                "claim_i": claims[i].text, "claim_j": claims[j].text,
                "role_i": roles[i].value, "role_j": roles[j].value,
                "severity": _severity(roles[i], roles[j]),
            }
            for i in range(len(claims)) for j in attackers[i] if i < j
        ]
        unresolved_conflicts = {
            i: sorted(attackers[i] & undec_set) for i in undec_set if attackers[i] & undec_set
        }
        orphan_claims = [
            i for i in range(len(claims))
            if roles[i] in (ClaimRole.MECHANISTIC_STEP, ClaimRole.PREDICTION)
            and not entailed_by[i]
        ]

        return MetricResult(
            metric=self.name, axis=self.axis,
            score=round(score_value, 4),
            anchor=_anchor(score_value),
            evidence={
                "n_claims": len(claims),
                "roles": [r.value for r in roles],
                "edges_checked": edges,
                "n_pairs_skipped_due_to_budget": n_skipped,
                "consistent_claim_indices": sorted(in_set),
                "out_claim_indices": sorted(out_set),
                "undecided_claim_indices": sorted(undec_set),
                "contradiction_pairs": contradiction_pairs,
                "unresolved_conflicts": {str(k): v for k, v in unresolved_conflicts.items()},
                "orphan_claim_indices": orphan_claims,
                "judge": ctx.judge.name,
            },
        )


def _anchor(score: float) -> str:
    if score >= 0.95:
        return "fully coherent"
    if score >= 0.7:
        return "mostly coherent, isolated contradictions"
    if score >= 0.4:
        return "substantial internal contradiction"
    return "incoherent"
