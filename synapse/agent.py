from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from agenttools import ToolUsageExample
from synapse.runtime import SynapseRuntime


@dataclass
class QueryContext:
    tool_name: str
    augmented_query: str
    recommended_scenario: Optional[str]


class SynapseAgent:
    """
    High-level agent that leverages the SYNAPSE runtime to route queries
    and inject federated knowledge into downstream tools.
    """

    def __init__(self, runtime: SynapseRuntime, tool_registry: Dict[str, object]) -> None:
        self.runtime = runtime
        self.tool_registry = tool_registry

    def _choose_tool(self, query: str, data_item: Optional[dict]) -> str:
        task_type = (data_item or {}).get("task_type")
        if isinstance(task_type, str):
            normalized = task_type.strip().lower()
            if normalized == "math":
                return "mathqa"
            if normalized in {"science", "image_required"}:
                return "scienceqa"
        dataset = (data_item or {}).get("dataset") or (data_item or {}).get("domain")
        if dataset and str(dataset).lower().startswith("bbh"):
            return "mathqa"
        if data_item and data_item.get("image"):
            return "scienceqa"
        numeric_tokens = sum(token.isdigit() for token in query.split())
        if numeric_tokens >= 2 or any(keyword in query.lower() for keyword in ("calculate", "solve", "equation", "ratio")):
            return "mathqa"
        return "scienceqa"

    def _prepare_augmented_query(self, query: str, tool_name: str, dataset: Optional[str]) -> QueryContext:
        if dataset and dataset.lower().startswith("bbh"):
            artifacts = []
        else:
            artifacts = self.runtime.get_context_for_query(query, max_items=5)
        context_snippets = [
            artifact.text
            for artifact in artifacts
            if artifact.metadata.get("tool") == tool_name
        ]
        recommended = None
        if context_snippets:
            augmented = query + "\n\nRelevant SYNAPSE knowledge:\n" + "\n".join(f"- {snippet}" for snippet in context_snippets)
            for artifact in artifacts:
                if artifact.metadata.get("tool") != tool_name:
                    continue
                scenario = artifact.metadata.get("scenario")
                if scenario:
                    recommended = scenario
                    break
        else:
            augmented = query
        return QueryContext(tool_name=tool_name, augmented_query=augmented, recommended_scenario=recommended)

    def run(self, query: str, data_item: Optional[dict] = None) -> ToolUsageExample:
        dataset = (data_item or {}).get("dataset")
        tool_name = self._choose_tool(query, data_item)
        if tool_name not in self.tool_registry:
            raise ValueError(f"No registered tool named '{tool_name}'.")

        query_context = self._prepare_augmented_query(query, tool_name, dataset)
        tool = self.tool_registry[tool_name]

        if tool_name == "scienceqa":
            return tool.run(
                user_query=query_context.augmented_query,
                data_item=data_item,
                recommended_scenario=query_context.recommended_scenario,
            )
        else:
            return tool.run(
                user_query=query_context.augmented_query,
                data_item=data_item,
                recommended_scenario=query_context.recommended_scenario,
            )
