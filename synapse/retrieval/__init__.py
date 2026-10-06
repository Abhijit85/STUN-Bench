"""
Retrieval utilities for SYNAPSE clients.

Modules here provide dynamic context selection that combines vector search
with symbolic queries over the shared compendium, plus local learned-router
utilities for replacement routing experiments.
"""

from .context import RetrievalPlanner, RetrievalConfig
from .learned_router import LearnedTextRouter, RoutingExample, cross_validated_predictions, load_routing_examples

__all__ = [
    "RetrievalPlanner",
    "RetrievalConfig",
    "LearnedTextRouter",
    "RoutingExample",
    "cross_validated_predictions",
    "load_routing_examples",
]
