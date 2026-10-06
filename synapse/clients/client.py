from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Set

from synapse.knowledge.compendium import KnowledgeArtifact, KnowledgePackage
from synapse.privacy.policies import PrivacyPolicy


@dataclass
class ClientMetadata:
    """Metadata describing the client and its local environment."""

    client_id: str
    domain_tags: List[str] = field(default_factory=list)
    capabilities: Dict[str, str] = field(default_factory=dict)


class SynapseClient:
    """
    Base implementation for a SYNAPSE client.

    Responsibilities:
      * Harvest local knowledge into KnowledgeArtifact structures.
      * Apply local filters (novelty, sensitivity) before sharing.
      * Communicate filtered packages to an edge aggregator.
    """

    def __init__(self, metadata: ClientMetadata, privacy_policy: Optional[PrivacyPolicy] = None) -> None:
        self.metadata = metadata
        self._last_package: Optional[KnowledgePackage] = None
        self._shared_signatures: Set[str] = set()
        self._last_raw_artifacts: List[KnowledgeArtifact] = []
        self._last_sanitized_artifacts: List[KnowledgeArtifact] = []
        self.privacy_policy = privacy_policy or PrivacyPolicy()
        env = os.environ
        self._artifact_max_chars = max(40, int(env.get("SYNAPSE_ARTIFACT_MAX_CHARS", "280")))
        self._artifact_max_sentences = max(1, int(env.get("SYNAPSE_ARTIFACT_MAX_SENTENCES", "1")))
        self._artifact_include_skills = env.get("SYNAPSE_ARTIFACT_INCLUDE_SKILLS", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self._structured_payload_mode = env.get("SYNAPSE_STRUCTURED_PAYLOAD_MODE", "typed").strip().lower()
        self._structured_text_style = env.get("SYNAPSE_STRUCTURED_TEXT_STYLE", "paper").strip().lower()

    def _flat_payload_fields(
        self,
        metadata: Optional[Dict[str, object]],
        payload: Optional[Dict[str, object]],
    ) -> Dict[str, object]:
        metadata = metadata or {}
        payload = payload or {}
        return {
            "scenario": metadata.get("scenario"),
            "tool_description": payload.get("tool_description"),
            "scenario_context": payload.get("scenario_context"),
            "precautions": payload.get("precautions"),
            "annex_summary": payload.get("annex_summary"),
        }

    def _flat_json_string(
        self,
        metadata: Optional[Dict[str, object]],
        payload: Optional[Dict[str, object]],
    ) -> str:
        compact = {
            key: value
            for key, value in self._flat_payload_fields(metadata, payload).items()
            if value not in (None, "", [], {})
        }
        return json.dumps(compact, sort_keys=True, ensure_ascii=True)

    def collect_local_artifacts(self) -> List[KnowledgeArtifact]:
        """
        Gather raw artifacts from the local environment.

        Subclasses should override to pull data from logs, transcripts,
        or other task-specific sources.
        """
        return []

    def build_knowledge_package(self) -> KnowledgePackage:
        """
        Construct a KnowledgePackage ready for transmission to the edge tier.
        """
        artifacts = self.collect_local_artifacts()

        normalized: List[KnowledgeArtifact] = []
        for artifact in artifacts:
            signature = artifact.signature or self._derive_signature(artifact)
            payload = self._normalize_structured_payload(artifact.structured_payload)
            normalized.append(
                KnowledgeArtifact(
                    signature=signature,
                    text=artifact.text,
                    structured_payload=payload,
                    metadata=artifact.metadata,
                    textgrad_variable=artifact.textgrad_variable,
                )
            )

        package = KnowledgePackage(
            source_id=self.metadata.client_id,
            artifacts=normalized,
            created_at=datetime.utcnow(),
            metadata={
                "domain_tags": self.metadata.domain_tags,
                "capabilities": self.metadata.capabilities,
            },
        )
        self._last_package = package
        self._last_raw_artifacts = normalized
        return package

    def _derive_signature(self, artifact: KnowledgeArtifact) -> str:
        hasher = hashlib.sha256()
        hasher.update(artifact.text.encode("utf-8"))
        if artifact.structured_payload:
            hasher.update(str(sorted(artifact.structured_payload.items())).encode("utf-8"))
        return hasher.hexdigest()

    def _filter_novel_artifacts(self, artifacts: Iterable[KnowledgeArtifact]) -> List[KnowledgeArtifact]:
        novel: List[KnowledgeArtifact] = []
        for artifact in artifacts:
            if artifact.signature in self._shared_signatures:
                continue
            novel.append(artifact)
        return novel

    def _score_artifact(self, artifact: KnowledgeArtifact) -> float:
        length_score = len(artifact.text.split()) / 50.0
        metadata_bonus = 0.0
        difficulty = artifact.metadata.get("difficulty")
        if isinstance(difficulty, (int, float)):
            metadata_bonus += difficulty / 5.0
        elif isinstance(difficulty, str) and difficulty.lower() in {"hard", "advanced"}:
            metadata_bonus += 1.0
        return length_score + metadata_bonus

    def _prioritize_artifacts(self, artifacts: Iterable[KnowledgeArtifact], budget: int = 10) -> List[KnowledgeArtifact]:
        scored = sorted(
            artifacts,
            key=lambda art: self._score_artifact(art),
            reverse=True,
        )
        return scored[:budget]

    def select_for_sharing(self, package: KnowledgePackage) -> KnowledgePackage:
        novel = self._filter_novel_artifacts(package.artifacts)
        prioritized = self._prioritize_artifacts(novel)
        sanitized = self.privacy_policy.enforce(prioritized)

        for artifact in sanitized:
            self._shared_signatures.add(artifact.signature)

        self._last_sanitized_artifacts = sanitized

        return KnowledgePackage(
            source_id=package.source_id,
            artifacts=sanitized,
            created_at=package.created_at,
            metadata=package.metadata,
        )

    def get_attack_artifacts(self) -> List[Dict[str, str]]:
        raw_lookup = {artifact.signature: artifact.text for artifact in self._last_raw_artifacts}
        paired: List[Dict[str, str]] = []
        for artifact in self._last_sanitized_artifacts:
            raw_text = raw_lookup.get(artifact.signature)
            if not raw_text:
                continue
            paired.append({
                "signature": artifact.signature,
                "raw_text": raw_text,
                "observed_text": artifact.text,
            })
        return paired

    def _condense_artifact_text(
        self,
        text: str,
        metadata: Optional[Dict[str, object]] = None,
        payload: Optional[Dict[str, object]] = None,
    ) -> str:
        if not text:
            return ""
        cleaned = text.strip()
        if not cleaned:
            return ""

        segments = re.split(r"(?<=[.!?])\s+", cleaned)
        summary_parts: List[str] = []
        for segment in segments:
            if not segment:
                continue
            summary_parts.append(segment.strip())
            if len(summary_parts) >= self._artifact_max_sentences:
                break
        summary = " ".join(summary_parts) if summary_parts else cleaned.splitlines()[0].strip()

        if len(summary) > self._artifact_max_chars:
            summary = summary[: self._artifact_max_chars].rstrip() + "…"

        if self._artifact_include_skills:
            skills = []
            if metadata and isinstance(metadata.get("skills"), list):
                skills = metadata["skills"]
            elif payload:
                for key in ("skills", "visual_skills", "textual_skills"):
                    value = payload.get(key)
                    if isinstance(value, list) and value:
                        skills = value
                        break
            if skills:
                skill_text = ", ".join(str(skill) for skill in skills[:3])
                summary = f"{summary} | skills: {skill_text}"

        return summary

    def _schema_mode(self) -> str:
        mode = self._structured_payload_mode.strip().lower()
        aliases = {
            "full": "typed",
            "none": "disabled",
            "off": "disabled",
            "flat": "flat_json",
            "serialized_untyped": "flat_json",
        }
        return aliases.get(mode, mode)

    def _merge_up_payload(self, payload: Dict[str, object]) -> Dict[str, object]:
        merged = dict(payload)
        notes: List[str] = []
        scenario_context = merged.pop("scenario_context", None)
        if isinstance(scenario_context, str) and scenario_context.strip():
            notes.append(scenario_context.strip())
        precautions = merged.pop("precautions", None)
        if isinstance(precautions, list):
            notes.extend(str(item).strip() for item in precautions if str(item).strip())
        if notes:
            merged["scenario_notes"] = notes
        return merged

    def _drop_annex_payload(self, payload: Dict[str, object]) -> Dict[str, object]:
        dropped = dict(payload)
        for key in ("annex_entities", "annex_relations", "annex_summary"):
            dropped.pop(key, None)
        return dropped

    def _normalize_structured_payload(
        self,
        payload: Optional[Dict[str, object]],
    ) -> Optional[Dict[str, object]]:
        if not payload:
            return payload

        normalized = dict(payload)
        mode = self._schema_mode()
        if mode == "untyped":
            normalized.pop("type", None)
        elif mode == "flat_json":
            normalized = {
                "payload_mode": "flat_json",
                "serialized_payload": self._flat_json_string(None, normalized),
            }
        elif mode == "merge_up":
            normalized = self._merge_up_payload(normalized)
        elif mode == "drop_annex":
            normalized = self._drop_annex_payload(normalized)
        elif mode in {"none", "disabled"}:
            return None
        return normalized

    def _format_relation(self, relation: object) -> str:
        if isinstance(relation, dict):
            source = str(relation.get("source") or "").strip()
            link = str(relation.get("link") or "").strip()
            target = str(relation.get("target") or "").strip()
            return " ".join(part for part in (source, link, target) if part)
        return str(relation).strip()

    def _compact_structured_prompt(self, metadata: Dict[str, object], payload: Dict[str, object], role: str) -> str:
        template = {
            "role": role,
            "tool": metadata.get("tool"),
            "domain": metadata.get("domain") or metadata.get("scenario"),
            "scenario": metadata.get("scenario"),
            "type": payload.get("type"),
            "difficulty": metadata.get("difficulty"),
            "skills": payload.get("skills")
                or payload.get("textual_skills")
                or payload.get("visual_skills"),
        }
        compact = {key: value for key, value in template.items() if value}
        if not compact:
            compact = {"role": role}
        return json.dumps(compact, ensure_ascii=False)

    def _structured_prompt(self, metadata: Dict[str, object], payload: Dict[str, object], role: str) -> str:
        if self._schema_mode() == "flat_json":
            return self._flat_json_string(metadata, payload)

        if self._structured_text_style in {"json", "compact", "compact_json"}:
            return self._compact_structured_prompt(metadata, payload, role)

        lines: List[str] = [f"role: {role}"]
        tool = metadata.get("tool")
        scenario = metadata.get("scenario")
        domain = metadata.get("domain") or scenario
        payload_type = payload.get("type")
        difficulty = metadata.get("difficulty")
        if tool:
            lines.append(f"tool: {tool}")
        if domain:
            lines.append(f"domain: {domain}")
        if scenario:
            lines.append(f"scenario: {scenario}")
        if payload_type:
            lines.append(f"artifact_type: {payload_type}")
        if difficulty:
            lines.append(f"difficulty: {difficulty}")

        tool_description = payload.get("tool_description")
        if isinstance(tool_description, str) and tool_description.strip():
            lines.append(f"tool_description: {tool_description.strip()}")

        scenario_context = payload.get("scenario_context")
        if isinstance(scenario_context, str) and scenario_context.strip():
            lines.append(f"scenario_context: {scenario_context.strip()}")

        scenario_notes = payload.get("scenario_notes")
        if isinstance(scenario_notes, list) and scenario_notes:
            note_text = "; ".join(str(item).strip() for item in scenario_notes if str(item).strip())
            if note_text:
                lines.append(f"scenario_notes: {note_text}")

        precautions = payload.get("precautions")
        if isinstance(precautions, list) and precautions:
            precaution_text = "; ".join(str(item).strip() for item in precautions if str(item).strip())
            if precaution_text:
                lines.append(f"precautions: {precaution_text}")

        annex_summary = payload.get("annex_summary")
        if isinstance(annex_summary, str) and annex_summary.strip():
            lines.append(f"structured_annex: {annex_summary.strip()}")
        else:
            annex_entities = payload.get("annex_entities")
            annex_relations = payload.get("annex_relations")
            annex_parts: List[str] = []
            if isinstance(annex_entities, list) and annex_entities:
                annex_parts.append("entities=" + ", ".join(str(item).strip() for item in annex_entities[:8] if str(item).strip()))
            if isinstance(annex_relations, list) and annex_relations:
                rel_text = "; ".join(self._format_relation(item) for item in annex_relations[:6] if self._format_relation(item))
                if rel_text:
                    annex_parts.append("relations=" + rel_text)
            if annex_parts:
                lines.append("structured_annex: " + " | ".join(annex_parts))

        skills = payload.get("skills") or payload.get("textual_skills") or payload.get("visual_skills")
        if isinstance(skills, list) and skills:
            lines.append("skills: " + ", ".join(str(item).strip() for item in skills if str(item).strip()))

        example = payload.get("example")
        if isinstance(example, str) and example.strip():
            lines.append(f"example: {example.strip()}")

        if len(lines) <= 5:
            return self._compact_structured_prompt(metadata, payload, role)
        return "\n".join(lines)

    def prepare_for_edge(self) -> KnowledgePackage:
        raw_package = self.build_knowledge_package()
        return self.select_for_sharing(raw_package)
