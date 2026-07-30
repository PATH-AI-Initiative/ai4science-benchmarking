"""The Tier 1 structured audit checklist.

Implemented as a set-level metric with ``score=None``: it contributes no number
to the composite but travels through the same pipeline and report as everything
else, so a tool's descriptive profile is captured in one place. The fields
mirror the framework's Details & Features sub-sections.
"""

from __future__ import annotations

from ...core.context import Context
from ...core.metric import Axis, MetricResult, SetMetric
from ...core.models import EvaluationRun, HypothesisSet
from ...core.registry import register

# The checklist fields to capture. These are recorded from documentation,
# metadata, and testing — not computed. ``None`` means "not yet assessed".
CHECKLIST_FIELDS = {
    "identity": ["name", "version", "evaluation_date"],
    "accessibility": [
        "cost_to_useful_hypothesis",
        "compute_requirements",
        "connectivity_requirements",
        "paywall_dependence",
        "language_support",
        "time_to_useful_hypothesis",
    ],
    "interface_output": [
        "input_flexibility",  # papers / data / knowledge graph
        "interaction_mode",  # single prompt vs multi-turn
        "output_format",
        "hypotheses_per_session",
    ],
    "domain": ["subject_area"],  # life-sciences-specific vs general-purpose
    "knowledge_base": ["breadth", "recency", "non_english_coverage"],
}


@register
class DetailsChecklist(SetMetric):
    name = "details_checklist"
    axis = Axis.DETAILS

    def score(self, outputs: HypothesisSet, run: EvaluationRun, ctx: Context) -> MetricResult:
        # Seed the checklist from what we already know, leaving the rest for the
        # evaluator to fill in from documentation and testing.
        profile: dict = {section: {f: None for f in fields}
                         for section, fields in CHECKLIST_FIELDS.items()}
        profile["identity"]["name"] = run.tool.name
        profile["identity"]["version"] = run.tool.version
        profile["identity"]["evaluation_date"] = (
            run.evaluated_on.isoformat() if run.evaluated_on else None
        )
        profile["interface_output"]["hypotheses_per_session"] = len(outputs)
        for key in ("cost_to_useful_hypothesis", "time_to_useful_hypothesis"):
            if key in run.metadata:
                profile["accessibility"][key] = run.metadata[key]

        return MetricResult(
            metric=self.name, axis=self.axis, score=None, evidence={"profile": profile}
        )
