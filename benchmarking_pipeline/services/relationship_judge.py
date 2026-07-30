"""Shared pairwise relationship classification: "how does text A relate to text B?"

Both logical consistency (claim vs. claim, within one hypothesis) and citation
support-checking (a cited source vs. the claim it's attached to) reduce to the
same operation: present two labelled pieces of text to the judge and constrain
the verdict to a fixed set of relationship labels. Factoring it out here means
both consume one prompt-construction path instead of duplicating it, while each
caller still supplies its own semantically appropriate labels, framing, and
choice set — a bidirectional claim-vs-claim check needs different choices than
a directional source-vs-claim support check.
"""

from __future__ import annotations

from .llm_judge import LLMJudge


def judge_relationship(
    judge: LLMJudge,
    *,
    label_a: str,
    text_a: str,
    label_b: str,
    text_b: str,
    instructions: str,
    choices: list[str],
) -> str:
    """Ask the judge to classify the relationship between two labelled texts.

    ``instructions`` should describe what each entry in ``choices`` means.
    By convention the safest/most conservative choice (e.g. "neutral") must be
    listed *last* in ``choices`` — it's the fallback if the judge's verdict
    isn't one of the offered choices.
    """
    prompt = f'{label_a}: "{text_a}"\n\n{label_b}: "{text_b}"\n\n{instructions}'
    verdict = judge.judge(prompt, choices=choices).verdict
    return verdict if verdict in choices else choices[-1]
