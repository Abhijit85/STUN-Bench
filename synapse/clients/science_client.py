from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from synapse.clients.client import ClientMetadata, SynapseClient
from synapse.knowledge.compendium import KnowledgeArtifact


class ScienceQAClient(SynapseClient):
    """
    SYNAPSE client that extracts multimodal ScienceQA knowledge.
    """

    def __init__(
        self,
        metadata: ClientMetadata,
        compendium_path: Path,
        dataset_path: Path,
        privacy_policy=None,
    ) -> None:
        super().__init__(metadata, privacy_policy=privacy_policy)
        self.compendium_path = compendium_path
        self.dataset_path = dataset_path

    def _load_compendium(self) -> Dict[str, object]:
        with self.compendium_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_dataset(self) -> List[Dict[str, object]]:
        with self.dataset_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else data.get("data", [])

    def collect_local_artifacts(self):
        compendium = self._load_compendium()
        textual_section = compendium.get("Textual_Compendium", {})
        scenarios = textual_section.get("Usage_Scenarios", [])
        privacy_metadata = {"tool": "scienceqa", "pii_safe": True}

        artifacts: List[KnowledgeArtifact] = []
        for item in scenarios:
            scenario_name = item.get("scenario", "Unnamed Scenario")
            context = item.get("context", "")
            metadata = {
                **privacy_metadata,
                "scenario": scenario_name,
                "domain": item.get("domain", "science"),
                "difficulty": item.get("difficulty", "medium"),
            }
            structured_payload = {
                "type": "usage_scenario",
                "visual_skills": item.get("visual_skills", []),
                "textual_skills": item.get("skills", []),
            }
            artifacts.append(
                KnowledgeArtifact(
                    signature=f"scienceqa::scenario::{scenario_name}",
                    text=f"{scenario_name}: {context}",
                    structured_payload=structured_payload,
                    metadata=metadata,
                )
            )

        for sample in self._load_dataset()[:20]:
            question = sample.get("question", "")
            lecture = sample.get("lecture", "")
            choices = sample.get("choices", [])
            answer = sample.get("answer", "")
            has_image = bool(sample.get("image"))

            structured_payload = {
                "type": "training_example",
                "choices": choices,
                "answer": answer,
                "has_image": has_image,
            }
            metadata = {
                **privacy_metadata,
                "difficulty": sample.get("difficulty", "medium"),
                "topic": sample.get("topic", ""),
            }
            text_block = f"Question: {question}\nLecture: {lecture}"
            artifacts.append(
                KnowledgeArtifact(
                    signature=f"scienceqa::example::{hash(question)}",
                    text=text_block,
                    structured_payload=structured_payload,
                    metadata=metadata,
                )
            )

        return artifacts
