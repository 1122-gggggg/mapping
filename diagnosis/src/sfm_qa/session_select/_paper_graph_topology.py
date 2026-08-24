"""Topology facade for paper-derived graph hardening."""

from ._paper_graph_community import _communities
from ._paper_graph_embedding import _component_plan, _features
from ._paper_graph_schedule import _schedule

__all__ = ["_communities", "_component_plan", "_features", "_schedule"]
