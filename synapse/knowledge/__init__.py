"""
Knowledge abstractions for the SYNAPSE framework.

Includes data structures for artifacts, packages, and compendium management.
"""

from .compendium import KnowledgeArtifact, KnowledgePackage, SynapseCompendium

__all__ = ["KnowledgeArtifact", "KnowledgePackage", "SynapseCompendium"]
