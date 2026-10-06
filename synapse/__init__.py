"""
SYNAPSE: Structured federated knowledge exchange framework.

This package provides the core building blocks for the SYNAPSE refactor,
including client, edge, and server components as well as shared knowledge
and retrieval utilities.
"""

from . import agent, clients, edge, server, knowledge, retrieval, privacy, runtime

__all__ = [
    "agent",
    "clients",
    "edge",
    "server",
    "knowledge",
    "retrieval",
    "privacy",
    "runtime",
]
