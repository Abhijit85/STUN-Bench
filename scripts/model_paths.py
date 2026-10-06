#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_HF_HOME = Path("<MOUNT>/shared/shared_hf_home")
SHARED_HF_HUB = SHARED_HF_HOME / "hub"
LOCAL_MODEL_ROOT = REPO_ROOT / "external_models" / "models"


MODEL_PATHS = {
    "llama31_8b": [
        SHARED_HF_HUB / "models--meta-llama--Llama-3.1-8B-Instruct" / "local_models" / "Llama-3.1-8B-Instruct",
        SHARED_HF_HUB / "models--meta-llama--Llama-3.1-8B-Instruct" / "snapshots" / "0e9e39f249a16976918f6564b8830bc894c89659",
        LOCAL_MODEL_ROOT / "meta-llama__Llama-3.1-8B-Instruct",
    ],
    "llama32_1b": [
        SHARED_HF_HUB / "manual-models" / "meta-llama__Llama-3.2-1B-Instruct",
        LOCAL_MODEL_ROOT / "meta-llama__Llama-3.2-1B-Instruct",
        REPO_ROOT / "artifacts" / "models" / "Llama-3.2-1B-Instruct",
    ],
    "mistral7b_v03": [
        SHARED_HF_HUB / "manual-models" / "mistralai__Mistral-7B-Instruct-v0.3",
        LOCAL_MODEL_ROOT / "mistralai__Mistral-7B-Instruct-v0.3",
    ],
    "qwen25_7b": [
        SHARED_HF_HUB / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots" / "a09a35458c702b33eeacc393d103063234e8bc28",
        LOCAL_MODEL_ROOT / "Qwen__Qwen2.5-7B-Instruct",
    ],
    "llama33_70b": [
        LOCAL_MODEL_ROOT / "meta-llama__Llama-3.3-70B-Instruct",
        SHARED_HF_HUB / "models--meta-llama--Llama-3.3-70B-Instruct",
    ],
}


def resolve_model_path(model_key: str, env_var: str | None = None) -> str:
    if env_var:
        override = os.environ.get(env_var)
        if override:
            return override
    for candidate in MODEL_PATHS[model_key]:
        if candidate.exists():
            return str(candidate)
    searched = ", ".join(str(path) for path in MODEL_PATHS[model_key])
    raise FileNotFoundError(f"No local/shared path found for {model_key}. Checked: {searched}")


def default_llama31_8b_path() -> str:
    return resolve_model_path("llama31_8b", env_var="SYNAPSE_DEFAULT_LLAMA31_8B")


def default_llama32_1b_path() -> str:
    return resolve_model_path("llama32_1b", env_var="SYNAPSE_DEFAULT_LLAMA32_1B")


def default_mistral7b_v03_path() -> str:
    return resolve_model_path("mistral7b_v03", env_var="SYNAPSE_DEFAULT_MISTRAL7B_V03")


def default_qwen25_7b_path() -> str:
    return resolve_model_path("qwen25_7b", env_var="SYNAPSE_DEFAULT_QWEN25_7B")


def default_llama33_70b_path() -> str:
    return resolve_model_path("llama33_70b", env_var="SYNAPSE_DEFAULT_LLAMA33_70B")
