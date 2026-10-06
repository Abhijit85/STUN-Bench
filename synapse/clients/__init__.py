"""
SYNAPSE client tier package.

Client modules handle local knowledge extraction, selective sharing,
and interactions with edge aggregators.
"""

from .client import SynapseClient, ClientMetadata
from .math_client import MathQAClient
from .science_client import ScienceQAClient
from .unified_client import UnifiedQAClient

__all__ = [
    "SynapseClient",
    "ClientMetadata",
    "MathQAClient",
    "ScienceQAClient",
    "UnifiedQAClient",
]
