from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from third_party.textgrad import Variable
from third_party.textgrad_utils.prompt_complexity import calculate_text_complexity

from synapse.clients.client import ClientMetadata, SynapseClient
from synapse.knowledge.compendium import KnowledgeArtifact
from synapse.textgrad_support import TextGradSettings


class UnifiedQAClient(SynapseClient):
    """
    SYNAPSE client that exposes both MathQA and ScienceQA knowledge so each federated
    participant can contribute multi-domain artifacts.
    """

    def __init__(
        self,
        metadata: ClientMetadata,
        math_compendium_path: Path,
        math_training_path: Path,
        science_compendium_path: Path,
        science_dataset_path: Path,
        *,
        privacy_policy=None,
        training_sample_limit: int = 20,
        textgrad_settings: Optional[TextGradSettings] = None,
    ) -> None:
        super().__init__(metadata, privacy_policy=privacy_policy)
        self.math_compendium_path = math_compendium_path
        self.math_training_path = math_training_path
        self.science_compendium_path = science_compendium_path
        self.science_dataset_path = science_dataset_path
        env_limit = os.environ.get("SYNAPSE_TRAINING_SAMPLE_LIMIT")
        if env_limit:
            try:
                training_sample_limit = max(0, int(env_limit))
            except ValueError:
                pass
        self.training_sample_limit = training_sample_limit
        self.include_training_artifacts = os.environ.get("SYNAPSE_INCLUDE_TRAINING_ARTIFACTS", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.training_shard_mode = os.environ.get("SYNAPSE_TRAINING_SHARD_MODE", "").strip().lower()
        included_tools = os.environ.get("SYNAPSE_INCLUDED_TOOLS", "mathqa,scienceqa")
        self.included_tools = {part.strip().lower() for part in included_tools.split(",") if part.strip()}
        if not self.included_tools:
            self.included_tools = {"mathqa", "scienceqa"}
        self._textgrad_settings = textgrad_settings
        self._cached_artifacts: Optional[List[KnowledgeArtifact]] = None
        self._contradiction_rate = max(0, min(100, int(os.environ.get("SYNAPSE_CONTRADICTION_RATE", "0"))))
        self._contradiction_seed = int(os.environ.get("SYNAPSE_CONTRADICTION_SEED", "0"))

    def set_textgrad_settings(self, settings: TextGradSettings) -> None:
        self._textgrad_settings = settings
        self._cached_artifacts = None

    def _load_json_file(self, path: Path) -> Dict[str, object] | List[Dict[str, object]]:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_math_compendium(self) -> Dict[str, object]:
        data = self._load_json_file(self.math_compendium_path)
        return data if isinstance(data, dict) else {}

    def _load_math_training(self) -> Sequence[Dict[str, object]]:
        data = self._load_json_file(self.math_training_path)
        if isinstance(data, dict):
            return data.get("examples", [])
        return data

    def _load_science_compendium(self) -> Dict[str, object]:
        data = self._load_json_file(self.science_compendium_path)
        return data if isinstance(data, dict) else {}

    def _load_science_dataset(self) -> Sequence[Dict[str, object]]:
        data = self._load_json_file(self.science_dataset_path)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("data", [])
        return []

    def _client_index(self) -> int:
        text = self.metadata.client_id.rsplit("-", 1)[-1]
        try:
            return max(0, int(text) - 1)
        except ValueError:
            return 0

    def _shard_training_rows(self, rows: Sequence[Dict[str, object]]) -> Sequence[Dict[str, object]]:
        mode = self.training_shard_mode
        if mode not in {"client_stride", "stride", "client"}:
            return rows
        total_clients_text = os.environ.get("SYNAPSE_CLIENT_COUNT", "")
        try:
            total_clients = max(1, int(total_clients_text))
        except ValueError:
            total_clients = 1
        client_index = self._client_index()
        return [row for idx, row in enumerate(rows) if idx % total_clients == client_index]

    def _usage_scenarios(self, data: Dict[str, object]) -> Sequence[Dict[str, object]]:
        scenarios = data.get("Usage_Scenarios")
        if scenarios:
            return scenarios
        textual = data.get("Textual_Compendium") or {}
        return textual.get("Usage_Scenarios", [])

    def _tool_description(self, data: Dict[str, object]) -> str:
        textual = data.get("Textual_Compendium") or {}
        description = textual.get("Tool_Description") or data.get("Tool_Description") or ""
        return str(description).strip()

    def _precaution_details(self, data: Dict[str, object]) -> List[str]:
        textual = data.get("Textual_Compendium") or {}
        precautions = textual.get("Precautions") or data.get("Precautions") or []
        details: List[str] = []
        for item in precautions:
            if not isinstance(item, dict):
                continue
            detail = item.get("details") or item.get("precaution")
            if isinstance(detail, str) and detail.strip():
                details.append(detail.strip())
        return details[:6]

    def _annex_data(self, data: Dict[str, object]) -> Dict[str, List[object] | str]:
        annex = data.get("Structured_Annex") or {}
        entities = annex.get("Entities") or []
        relations = annex.get("Relations") or []
        entity_list = [str(item).strip() for item in entities if str(item).strip()][:12]
        relation_list = []
        for item in relations[:12]:
            if isinstance(item, dict):
                source = str(item.get("source") or "").strip()
                link = str(item.get("link") or "").strip()
                target = str(item.get("target") or "").strip()
                relation = {"source": source, "link": link, "target": target}
                relation_list.append(relation)
        summary_parts: List[str] = []
        if entity_list:
            summary_parts.append("entities: " + ", ".join(entity_list[:8]))
        if relation_list:
            relation_text = "; ".join(
                " ".join(part for part in (rel["source"], rel["link"], rel["target"]) if part)
                for rel in relation_list[:6]
            )
            if relation_text:
                summary_parts.append("relations: " + relation_text)
        return {
            "annex_entities": entity_list,
            "annex_relations": relation_list,
            "annex_summary": " | ".join(summary_parts),
        }

    def _scenario_fingerprint(self, scenario_name: str) -> int:
        material = f"{self._contradiction_seed}:{self.metadata.client_id}:{scenario_name}"
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % 100

    def _should_contradict(self, scenario_name: str) -> bool:
        return self._contradiction_rate > 0 and self._scenario_fingerprint(scenario_name) < self._contradiction_rate

    def _usage_neighbors(self, scenarios: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
        def tokenize(text: str) -> set[str]:
            return {token for token in text.lower().replace(':', ' ').replace(',', ' ').split() if token}

        def score(left: Dict[str, object], right: Dict[str, object]) -> tuple[int, int]:
            left_context = str(left.get("context") or "")
            right_context = str(right.get("context") or "")
            overlap = len(tokenize(left_context) & tokenize(right_context))
            return overlap, -abs(len(left_context) - len(right_context))

        neighbors: Dict[str, Dict[str, object]] = {}
        scenario_rows = [row for row in scenarios if isinstance(row, dict)]
        for row in scenario_rows:
            scenario_name = str(row.get("scenario") or row.get("name") or "").strip()
            if not scenario_name:
                continue
            candidates = [candidate for candidate in scenario_rows if candidate is not row]
            if not candidates:
                continue
            neighbors[scenario_name] = max(candidates, key=lambda candidate: score(row, candidate))
        return neighbors

    def _corrupt_scenario_payload(
        self,
        *,
        scenario_name: str,
        payload: Dict[str, object],
        neighbor: Dict[str, object] | None,
        precautions: List[str],
        annex: Dict[str, List[object] | str],
    ) -> Dict[str, object]:
        if not self._should_contradict(scenario_name) or not neighbor:
            return payload

        corrupted = dict(payload)
        neighbor_context = str(neighbor.get("context") or "").strip()
        if neighbor_context:
            corrupted["scenario_context"] = neighbor_context
        if precautions:
            corrupted["precautions"] = list(precautions)[:6]
        for key in ("annex_entities", "annex_relations", "annex_summary"):
            if annex.get(key):
                corrupted[key] = annex[key]
        corrupted["contradicted"] = True
        corrupted["contradiction_source"] = str(neighbor.get("scenario") or neighbor.get("name") or "").strip()
        return corrupted

    def _scenario_payload(
        self,
        *,
        scenario: Dict[str, object],
        tool_name: str,
        tool_description: str,
        precautions: List[str],
        annex: Dict[str, List[object] | str],
        modality_keys: Dict[str, str],
    ) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "type": "usage_scenario",
            "channel": "usage_scenario",
            "tool_description": tool_description,
            "scenario_context": str(scenario.get("context") or "").strip(),
            "precautions": precautions,
            "annex_entities": annex.get("annex_entities", []),
            "annex_relations": annex.get("annex_relations", []),
            "annex_summary": annex.get("annex_summary", ""),
        }
        for source_key, target_key in modality_keys.items():
            payload[target_key] = scenario.get(source_key, [])
        if scenario.get("example"):
            payload["example"] = scenario.get("example")
        return payload

    def collect_local_artifacts(self):
        if self._textgrad_settings and self._textgrad_settings.enabled and self._cached_artifacts is not None:
            return self._cached_artifacts

        artifacts: List[KnowledgeArtifact] = []

        if "mathqa" in self.included_tools:
            math_compendium = self._load_math_compendium()
            math_tool_description = self._tool_description(math_compendium)
            math_precautions = self._precaution_details(math_compendium)
            math_annex = self._annex_data(math_compendium)
            math_scenarios = list(self._usage_scenarios(math_compendium))
            math_neighbors = self._usage_neighbors(math_scenarios)

            for scenario in math_scenarios:
                scenario_name = scenario.get("scenario") or scenario.get("name", "Unnamed Scenario")
                context = str(scenario.get("context") or "").strip()
                metadata = {
                    "tool": "mathqa",
                    "scenario": scenario_name,
                    "pii_safe": True,
                    "difficulty": scenario.get("difficulty", "medium"),
                }
                payload = self._scenario_payload(
                    scenario=scenario,
                    tool_name="mathqa",
                    tool_description=math_tool_description,
                    precautions=math_precautions,
                    annex=math_annex,
                    modality_keys={"skills": "skills"},
                )
                payload = self._corrupt_scenario_payload(
                    scenario_name=str(scenario_name),
                    payload=payload,
                    neighbor=math_neighbors.get(str(scenario_name)),
                    precautions=math_precautions,
                    annex=math_annex,
                )
                artifacts.append(
                    self._create_textgrad_artifact(
                        signature=f"unified::math::scenario::{scenario_name}",
                        text=f"{scenario_name}: {context}",
                        metadata=metadata,
                        payload=payload,
                        role_description="structured system prompt for math QA scenarios",
                    )
                )

            if self.include_training_artifacts and self.training_sample_limit > 0:
                math_training_rows = self._shard_training_rows(self._load_math_training())
                for sample in math_training_rows[: self.training_sample_limit]:
                    question = sample.get("Problem") or sample.get("question", "")
                    solution = sample.get("Rationale") or sample.get("solution", "")
                    options = sample.get("options")
                    difficulty = sample.get("difficulty", "medium")
                    structured = {
                        "type": "training_example",
                        "options": options,
                        "answer": sample.get("Answer") or sample.get("correct"),
                    }
                    metadata = {
                        "tool": "mathqa",
                        "pii_safe": True,
                        "difficulty": difficulty,
                    }
                    artifacts.append(
                        self._create_textgrad_artifact(
                            signature=f"unified::math::example::{hash(question)}",
                            text=f"Problem: {question}\nSolution: {solution}",
                            metadata=metadata,
                            payload=structured,
                            role_description="instructional prompt describing a math QA training example",
                        )
                    )

        if "scienceqa" in self.included_tools:
            science_privacy = {"pii_safe": True}
            science_compendium = self._load_science_compendium()
            science_tool_description = self._tool_description(science_compendium)
            science_precautions = self._precaution_details(science_compendium)
            science_annex = self._annex_data(science_compendium)

            for scenario in self._usage_scenarios(science_compendium):
                scenario_name = scenario.get("scenario", "Unnamed Scenario")
                context = str(scenario.get("context") or "").strip()
                metadata = {
                    **science_privacy,
                    "tool": "scienceqa",
                    "scenario": scenario_name,
                    "domain": scenario.get("domain", "science"),
                    "difficulty": scenario.get("difficulty", "medium"),
                }
                payload = self._scenario_payload(
                    scenario=scenario,
                    tool_name="scienceqa",
                    tool_description=science_tool_description,
                    precautions=science_precautions,
                    annex=science_annex,
                    modality_keys={"visual_skills": "visual_skills", "skills": "textual_skills"},
                )
                artifacts.append(
                    self._create_textgrad_artifact(
                        signature=f"unified::science::scenario::{scenario_name}",
                        text=f"{scenario_name}: {context}",
                        metadata=metadata,
                        payload=payload,
                        role_description="structured system prompt for science QA scenarios",
                    )
                )

        if "scienceqa" in self.included_tools and self.include_training_artifacts and self.training_sample_limit > 0:
            science_privacy = {"pii_safe": True}
            science_rows = self._shard_training_rows(self._load_science_dataset())
            for sample in science_rows[: self.training_sample_limit]:
                question = sample.get("question", "")
                lecture = sample.get("lecture", "")
                choices = sample.get("choices", [])
                answer = sample.get("answer", "")
                has_image = bool(sample.get("image"))
                metadata = {
                    **science_privacy,
                    "tool": "scienceqa",
                    "difficulty": sample.get("difficulty", "medium"),
                    "topic": sample.get("topic", ""),
                }
                payload = {
                    "type": "training_example",
                    "choices": choices,
                    "answer": answer,
                    "has_image": has_image,
                }
                text_block = f"Question: {question}\nLecture: {lecture}"
                artifacts.append(
                    self._create_textgrad_artifact(
                        signature=f"unified::science::example::{hash(question)}",
                        text=text_block,
                        metadata=metadata,
                        payload=payload,
                        role_description="instructional prompt describing a science QA training example",
                    )
                )

        if self._textgrad_settings and self._textgrad_settings.enabled:
            self._cached_artifacts = artifacts

        return artifacts

    def _create_textgrad_artifact(
        self,
        *,
        signature: str,
        text: str,
        metadata: Dict[str, object],
        payload: Dict[str, object],
        role_description: str,
    ) -> KnowledgeArtifact:
        metadata_copy = dict(metadata)
        payload_copy = dict(payload)
        structured_text = self._structured_prompt(metadata_copy, payload_copy, role_description)
        textgrad_variable: Optional[Variable] = None

        if self._textgrad_settings and self._textgrad_settings.enabled:
            textgrad_variable = Variable(
                structured_text,
                requires_grad=True,
                role_description=role_description,
            )
            complexity = calculate_text_complexity(text)
            metadata_copy["textgrad_enabled"] = True
            metadata_copy["textgrad_complexity"] = complexity
            metadata_copy["textgrad_role"] = role_description

            textgrad_payload = dict(payload_copy.get("textgrad", {}))
            textgrad_payload.update(
                {
                    "role": role_description,
                    "complexity": complexity,
                    "aggregate_method": self._textgrad_settings.aggregate_method,
                }
            )
            payload_copy["textgrad"] = textgrad_payload

        return KnowledgeArtifact(
            signature=signature,
            text=structured_text,
            structured_payload=payload_copy,
            metadata=metadata_copy,
            textgrad_variable=textgrad_variable,
        )
