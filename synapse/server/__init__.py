"""
SYNAPSE server tier package.

The server orchestrates global knowledge consolidation, versioning, and
distribution back to edge tiers.
"""

from .orchestrator import SynapseServer, ServerConfig

__all__ = ["SynapseServer", "ServerConfig"]
