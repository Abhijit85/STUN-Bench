from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from synapse.clients.client import ClientMetadata, SynapseClient
from synapse.knowledge.compendium import KnowledgeArtifact


class MathQAClient(SynapseClient):
    """
    SYNAPSE client that curates MathQA knowledge from local compendiums
    and training datasets.
    """

    def __init__(
        self,
        metadata: ClientMetadata,
        compendium_path: Path,
        training_data_path: Path,
        privacy_policy=None,
    ) -> None:
        super().__init__(metadata, privacy_policy=privacy_policy)
        self.compendium_path = compendium_path
        self.training_data_path = training_data_path

    def _load_compendium(self) -> Dict[str, object]:
        with self.compendium_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_training_samples(self) -> List[Dict[str, object]]:
        with self.training_data_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data.get("examples", [])
        return data

    def collect_local_artifacts(self):
        compendium = self._load_compendium()
        usage_scenarios = compendium.get("Usage_Scenarios") or compendium.get("Textual_Compendium", {}).get("Usage_Scenarios", [])

        artifacts: List[KnowledgeArtifact] = []
        for scenario in usage_scenarios:
            scenario_name = scenario.get("scenario") or scenario.get("name", "Unnamed Scenario")
            context = scenario.get("context", "")
            metadata = {
                "tool": "mathqa",
                "scenario": scenario_name,
                "pii_safe": True,
                "difficulty": scenario.get("difficulty", "medium"),
            }
            artifacts.append(
                KnowledgeArtifact(
                    signature=f"mathqa::scenario::{scenario_name}",
                    text=f"{scenario_name}: {context}",
                    structured_payload={
                        "type": "usage_scenario",
                        "skills": scenario.get("skills", []),
                        "example": scenario.get("example"),
                    },
                    metadata=metadata,
                )
            )

        for sample in self._load_training_samples()[:20]:
            question = sample.get("Problem") or sample.get("question", "")
            solution = sample.get("Rationale") or sample.get("solution", "")
            options = sample.get("options")
            difficulty = sample.get("difficulty", "medium")
            structured = {
                "type": "training_example",
                "options": options,
                "answer": sample.get("Answer"),
            }
            metadata = {
                "tool": "mathqa",
                "pii_safe": True,
                "difficulty": difficulty,
            }
            artifacts.append(
                KnowledgeArtifact(
                    signature=f"mathqa::example::{hash(question)}",
                    text=f"Problem: {question}\nSolution: {solution}",
                    structured_payload=structured,
                    metadata=metadata,
                )
            )

        return artifacts
