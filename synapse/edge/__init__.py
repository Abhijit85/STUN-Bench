"""
SYNAPSE edge tier package.

Edge aggregators cluster client submissions, deduplicate artifacts, and
relay consolidated packages toward the central server.
"""

from .aggregator import EdgeAggregator, EdgeConfig

__all__ = ["EdgeAggregator", "EdgeConfig"]
