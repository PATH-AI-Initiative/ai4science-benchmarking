"""Getting tool outputs into the framework's data model.

``parsers`` turns captured text (docx, markdown, JSON) into a
:class:`~benchmarking_pipeline.core.models.HypothesisSet`. ``tool_adapters``
covers how outputs are obtained in the first place — from a pasted file today,
from a live tool API later.
"""
