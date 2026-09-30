"""Core data model.

A tool run produces an ordered :class:`HypothesisSet`. Order is load-bearing:
the framework scores a tool's best hypotheses and checks whether they rank at
the top, so ``rank`` is preserved end to end.

Each :class:`Hypothesis` decomposes into :class:`Claim` objects, which carry
the :class:`Reference` and :class:`Entity` objects the Accuracy axis
validates against external sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class ClaimRole(str, Enum):
    """The role a claim plays in a hypothesis's argument structure.

    Used by the logical-consistency metric to prioritise which claim pairs are
    worth checking (premise/mechanism/prediction combinations matter more than
    background-vs-background) and to grade contradiction severity. Unknown
    until classified — ``Claim.role`` is ``None`` until something tags it.
    """

    PREMISE = "premise"
    MECHANISTIC_STEP = "mechanistic_step"
    PREDICTION = "prediction"
    BACKGROUND_ASSUMPTION = "background_assumption"


@dataclass
class Reference:
    """A citation attached to a claim."""

    raw: str
    title: str | None = None
    doi: str | None = None
    year: int | None = None
    authors: list[str] = field(default_factory=list)


@dataclass
class Entity:
    """A named biological entity (gene, protein, pathway, ...) mentioned in a claim."""

    name: str
    kind: str | None = None  # e.g. "gene", "protein", "pathway"
    normalized_id: str | None = None  # e.g. a UniProt or KEGG identifier once resolved


@dataclass
class Claim:
    """A single factual assertion within a hypothesis."""

    text: str
    references: list[Reference] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    role: ClaimRole | None = None


@dataclass
class Hypothesis:
    """One hypothesis produced by a tool."""

    id: str
    text: str
    rank: int | None = None  # position in the tool's output ordering (1 = top)
    category: str | None = None  # section/category grouping, for tools that organize by theme rather than flat rank
    claims: list[Claim] = field(default_factory=list)
    raw: str | None = None  # unparsed source text, kept for auditability
    # Free-form capture of tool-reported assessments (e.g. its own novelty/feasibility
    # ratings) that aren't scientific claims themselves and so don't belong in `claims` --
    # not used for scoring directly, but carried through for audit/comparison against
    # the framework's own axes (e.g. size_of_leap, tractability).
    metadata: dict = field(default_factory=dict)


@dataclass
class HypothesisSet:
    """The ordered set of hypotheses returned by a single tool run."""

    hypotheses: list[Hypothesis] = field(default_factory=list)

    def __iter__(self):
        return iter(self.hypotheses)

    def __len__(self) -> int:
        return len(self.hypotheses)

    def at_rank(self, rank: int = 1) -> "Hypothesis | None":
        """The hypothesis at the given rank (1 = the tool's top idea).

        Falls back to the first hypothesis in ranked order if that exact rank
        isn't present (e.g. the run only returned one, unranked, hypothesis).
        ``None`` if the set is empty.
        """
        for h in self.hypotheses:
            if h.rank == rank:
                return h
        ordered = sorted(
            self.hypotheses,
            key=lambda h: (h.rank is None, h.rank if h.rank is not None else 0),
        )
        return ordered[0] if ordered else None


@dataclass
class Tool:
    """Identity of the tool under evaluation (Tier 1 'Identity & framing')."""

    name: str
    version: str | None = None
    access: str | None = None  # "open" | "controlled"


@dataclass
class EvaluationRun:
    """Everything captured from one prompt submitted to one tool.

    This is the unit the pipeline evaluates. ``metadata`` holds free-form
    capture details (tokens used, wall-clock time, cost) that feed the Tier 1
    audit checklist. For a perturbation run, ``metadata["perturbation"]`` tags
    what was changed (e.g. ``"reword"``, ``"kb_removal"``).
    """

    tool: Tool
    prompt: str
    outputs: HypothesisSet
    evaluated_on: date | None = None
    metadata: dict = field(default_factory=dict)
    # The tool's full unparsed output, when the capture preserved it -- lets a
    # metric (e.g. `adversarial`) see prose outside the structured hypotheses,
    # which extraction otherwise drops entirely. `None` for older captures
    # that predate this field, or adapters that don't populate it.
    raw_text: str | None = None


@dataclass
class RunBundle:
    """A group of runs of the same tool, for metrics that are properties of a
    *set of runs* rather than a single run.

    * ``base`` — the canonical run (also what single-run metrics score).
    * ``repeats`` — the identical prompt re-run N times (reproducibility).
    * ``perturbations`` — reworded-prompt / perturbed-knowledge-base runs
      (robustness), each tagged via ``metadata["perturbation"]``.
    """

    base: EvaluationRun
    repeats: list[EvaluationRun] = field(default_factory=list)
    perturbations: list[EvaluationRun] = field(default_factory=list)
