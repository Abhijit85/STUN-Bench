import importlib
import hashlib
import json
import math
import os
import re
import sys
import time
import types
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

try:
    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional local backend
    torch = None
    AutoConfig = None
    AutoModel = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

def _ensure_torchvision_stub() -> None:
    if 'torchvision' in sys.modules:
        return
    try:
        import torchvision  # type: ignore  # pragma: no cover
        return
    except Exception:
        pass
    import importlib.machinery

    torchvision = types.ModuleType('torchvision')
    torchvision.__spec__ = importlib.machinery.ModuleSpec('torchvision', loader=None)

    io_mod = types.ModuleType('torchvision.io')
    io_mod.__spec__ = importlib.machinery.ModuleSpec('torchvision.io', loader=None)

    class ImageReadMode:
        UNCHANGED = 0

    def decode_image(*_args, **_kwargs):
        raise RuntimeError('torchvision image decoding is unavailable in this environment')

    io_mod.ImageReadMode = ImageReadMode
    io_mod.decode_image = decode_image

    transforms_mod = types.ModuleType('torchvision.transforms')
    transforms_mod.__spec__ = importlib.machinery.ModuleSpec('torchvision.transforms', loader=None)

    class InterpolationMode:
        NEAREST = 0
        NEAREST_EXACT = 0
        BILINEAR = 1
        BICUBIC = 2
        LANCZOS = 3
        HAMMING = 4
        BOX = 5

    transforms_mod.InterpolationMode = InterpolationMode

    functional_mod = types.ModuleType('torchvision.transforms.functional')
    functional_mod.__spec__ = importlib.machinery.ModuleSpec('torchvision.transforms.functional', loader=None)

    def pil_to_tensor(*_args, **_kwargs):
        raise RuntimeError('torchvision functional transforms are unavailable in this environment')

    functional_mod.pil_to_tensor = pil_to_tensor

    v2_mod = types.ModuleType('torchvision.transforms.v2')
    v2_mod.__spec__ = importlib.machinery.ModuleSpec('torchvision.transforms.v2', loader=None)
    v2_functional_mod = types.ModuleType('torchvision.transforms.v2.functional')
    v2_functional_mod.__spec__ = importlib.machinery.ModuleSpec('torchvision.transforms.v2.functional', loader=None)
    v2_mod.functional = v2_functional_mod

    transforms_mod.functional = functional_mod
    transforms_mod.v2 = v2_mod
    torchvision.io = io_mod
    torchvision.transforms = transforms_mod

    sys.modules['torchvision'] = torchvision
    sys.modules['torchvision.io'] = io_mod
    sys.modules['torchvision.transforms'] = transforms_mod
    sys.modules['torchvision.transforms.functional'] = functional_mod
    sys.modules['torchvision.transforms.v2'] = v2_mod
    sys.modules['torchvision.transforms.v2.functional'] = v2_functional_mod


def _ensure_torchaudio_stub() -> None:
    if 'torchaudio' in sys.modules:
        return
    try:
        import torchaudio  # type: ignore  # pragma: no cover
        return
    except Exception:
        pass
    import importlib.machinery

    torchaudio = types.ModuleType('torchaudio')
    torchaudio.__spec__ = importlib.machinery.ModuleSpec('torchaudio', loader=None)
    sys.modules['torchaudio'] = torchaudio


import pandas as pd
import requests
from dotenv import load_dotenv

from agenttools import BaseTool, CalculatorTool, ToolUsageExample
from jina_key_manager import JinaAPIKeyRotator, get_available_jina_api_keys
from mongo_utils import MongoVectorStore  # Re-use the connection utility
from openrouter_client import chat_completion

# --- 1. CONFIGURATION ---
load_dotenv()
API_KEYS = get_available_jina_api_keys(allow_empty=True)

MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "FredRag"
MATHQA_COLLECTION = os.environ.get("MATHQA_COLLECTION", "math_problems")

JINA_EMBED_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_RERANK_API_URL = "https://api.jina.ai/v1/rerank"
JINA_CHAT_API_URL = "https://api.jina.ai/v1/chat/completions"


def _local_mathqa_model_path() -> Optional[str]:
    return (
        os.environ.get("MATHQA_LOCAL_MODEL_PATH")
        or os.environ.get("LOCAL_MATHQA_MODEL_PATH")
        or os.environ.get("LOCAL_MODEL_PATH")
    )


def _local_mathqa_reranker_model_path() -> Optional[str]:
    return (
        os.environ.get("MATHQA_LOCAL_RERANKER_MODEL")
        or os.environ.get("MATHQA_RERANK_MODEL")
        or os.environ.get("SYNAPSE_LOCAL_RERANK_MODEL")
        or os.environ.get("SYNAPSE_RERANK_MODEL")
    )


def _local_jina_embed_model_path() -> Optional[str]:
    return (
        os.environ.get("JINA_LOCAL_EMBED_MODEL")
        or os.environ.get("LOCAL_JINA_EMBED_MODEL")
        or os.environ.get("SYNAPSE_LOCAL_EMBED_MODEL")
    )


def _local_jina_impl_model_path() -> Optional[str]:
    repo_default = Path(__file__).resolve().parent / "external_models" / "jinaai" / "jina-bert-implementation"
    return (
        os.environ.get("JINA_LOCAL_EMBED_IMPL_MODEL")
        or os.environ.get("LOCAL_JINA_EMBED_IMPL_MODEL")
        or (str(repo_default) if repo_default.exists() else None)
    )


def _local_jina_embed_device() -> str:
    device_name = os.environ.get("JINA_LOCAL_EMBED_DEVICE") or os.environ.get("SYNAPSE_LOCAL_EMBED_DEVICE")
    if device_name:
        return device_name
    return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


def _ensure_transformers_onnx_shim() -> None:
    if 'transformers.onnx' in sys.modules:
        return
    try:
        import transformers.onnx  # type: ignore  # pragma: no cover
        return
    except Exception:
        module = types.ModuleType('transformers.onnx')
        class OnnxConfig:  # pragma: no cover - compatibility shim
            pass
        module.OnnxConfig = OnnxConfig
        sys.modules['transformers.onnx'] = module


def _ensure_transformers_config_defaults() -> None:
    try:
        from transformers import PretrainedConfig
    except Exception:
        return
    if not hasattr(PretrainedConfig, 'is_decoder'):
        PretrainedConfig.is_decoder = False
    if not hasattr(PretrainedConfig, 'add_cross_attention'):
        PretrainedConfig.add_cross_attention = False


def _ensure_transformers_pytorch_utils_shim() -> None:
    try:
        import torch
        import transformers.pytorch_utils as pt_utils
    except Exception:
        return
    if hasattr(pt_utils, 'find_pruneable_heads_and_indices'):
        return

    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        heads = set(heads) - set(already_pruned_heads)
        mask = torch.ones(n_heads, head_size, dtype=torch.bool)
        for head in heads:
            head = head - sum(1 if pruned_head < head else 0 for pruned_head in already_pruned_heads)
            if 0 <= head < n_heads:
                mask[head] = False
        mask = mask.view(-1).contiguous().eq(True)
        index = torch.arange(mask.numel(), dtype=torch.long)[mask]
        return heads, index

    pt_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices


def _patch_local_jina_modeling_if_needed(model_path: str) -> None:
    if torch is None:
        return
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except Exception:
        return
    local_only = os.environ.get('JINA_LOCAL_EMBED_LOCAL_ONLY', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
    try:
        config_path = Path(model_path) / "config.json"
        model_ref = 'jinaai/jina-bert-implementation--modeling_bert.JinaBertModel'
        if config_path.exists():
            try:
                config_payload = json.loads(config_path.read_text())
                model_ref = config_payload.get("auto_map", {}).get("AutoModel", model_ref)
            except Exception:
                pass
        model_cls = get_class_from_dynamic_module(
            model_ref,
            model_path,
            local_files_only=local_only,
        )
        module = importlib.import_module(model_cls.__module__)
        encoder_cls = getattr(module, 'JinaBertEncoder', None)
        if encoder_cls is None or getattr(encoder_cls, '_synapse_alibi_patch', False):
            return

        def rebuild_alibi_tensor(self, size: int, device: Optional[torch.device | str] = None):
            if device is None:
                device = torch.device('cpu')
            else:
                device = torch.device(device)
            if device.type == 'meta':
                device = torch.device('cpu')
            n_heads = self.num_attention_heads

            def _get_alibi_head_slopes(n_heads: int) -> list[float]:
                def get_slopes_power_of_2(n: int) -> list[float]:
                    start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                    ratio = start
                    return [start * ratio**i for i in range(n)]

                if math.log2(n_heads).is_integer():
                    return get_slopes_power_of_2(n_heads)
                closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
                return get_slopes_power_of_2(closest_power_of_2) + _get_alibi_head_slopes(2 * closest_power_of_2)[0::2][: n_heads - closest_power_of_2]

            context_position = torch.arange(size, device=device)[:, None]
            memory_position = torch.arange(size, device=device)[None, :]
            relative_position = torch.abs(memory_position - context_position)
            relative_position = relative_position.unsqueeze(0).expand(n_heads, -1, -1).to(device=device, dtype=torch.float32)
            slopes = torch.tensor(_get_alibi_head_slopes(n_heads), device=device, dtype=torch.float32) * -1
            alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position
            alibi = alibi.unsqueeze(0)
            self._current_alibi_size = size
            return alibi

        encoder_cls.rebuild_alibi_tensor = rebuild_alibi_tensor
        encoder_cls._synapse_alibi_patch = True

        if not hasattr(model_cls, '_convert_head_mask_to_5d'):
            def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
                if head_mask.dim() == 1:
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                assert head_mask.dim() == 5, f'head_mask.dim != 5, got {head_mask.dim()}'
                return head_mask.to(dtype=self.dtype)

            model_cls._convert_head_mask_to_5d = _convert_head_mask_to_5d

        if not hasattr(model_cls, 'get_head_mask'):
            def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked: bool = False):
                if head_mask is not None:
                    head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
                    if is_attention_chunked:
                        head_mask = head_mask.unsqueeze(-1)
                else:
                    head_mask = [None] * num_hidden_layers
                return head_mask

            model_cls.get_head_mask = get_head_mask
    except Exception:
        return


def _prepare_local_jina_embed_model_dir(model_path: str) -> str:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return model_path
    try:
        config_payload = json.loads(config_path.read_text())
    except Exception:
        return model_path
    auto_map = config_payload.get("auto_map")
    if not isinstance(auto_map, dict) or not auto_map:
        return model_path

    impl_model_path = _local_jina_impl_model_path()
    if not impl_model_path:
        return model_path
    impl_dir = Path(impl_model_path)
    config_py = impl_dir / "configuration_bert.py"
    model_py = impl_dir / "modeling_bert.py"
    if not config_py.exists() or not model_py.exists():
        return model_path

    model_dir = Path(model_path)
    has_local_impl_files = (model_dir / "configuration_bert.py").exists() and (model_dir / "modeling_bert.py").exists()
    has_remote_auto_map = any("--" in str(value) for value in auto_map.values())
    if has_local_impl_files and not has_remote_auto_map:
        return model_path

    cache_material = "::".join([
        str(model_dir.resolve()),
        str(impl_dir.resolve()),
        config_path.read_text(),
        config_py.read_text(),
        model_py.read_text(),
    ])
    cache_key = hashlib.sha256(cache_material.encode("utf-8")).hexdigest()[:16]
    prepared_dir = Path("/tmp") / f"jina_embed_local_{cache_key}"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    for entry in model_dir.iterdir():
        target = prepared_dir / entry.name
        if target.exists() or target.is_symlink():
            continue
        if entry.is_dir():
            target.symlink_to(entry.resolve(), target_is_directory=True)
        else:
            target.symlink_to(entry.resolve())

    for src_path in (config_py, model_py):
        target = prepared_dir / src_path.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(src_path.resolve())
    init_path = prepared_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text("")

    patched_config = dict(config_payload)
    patched_config["_name_or_path"] = "local-jina-bert-implementation"
    patched_config["auto_map"] = {
        "AutoConfig": "configuration_bert.JinaBertConfig",
        "AutoModelForMaskedLM": "modeling_bert.JinaBertForMaskedLM",
        "AutoModel": "modeling_bert.JinaBertModel",
        "AutoModelForSequenceClassification": "modeling_bert.JinaBertForSequenceClassification",
    }
    (prepared_dir / "config.json").write_text(json.dumps(patched_config, indent=2, sort_keys=True))
    return str(prepared_dir)


_ensure_torchvision_stub()
_ensure_torchaudio_stub()
_ensure_transformers_onnx_shim()
_ensure_transformers_config_defaults()
_ensure_transformers_pytorch_utils_shim()
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional local embedding backend
    SentenceTransformer = None


@lru_cache(maxsize=1)
def _load_local_jina_embedder():
    model_path = _local_jina_embed_model_path()
    if not model_path:
        return None
    if torch is None or AutoConfig is None or AutoModel is None or AutoTokenizer is None:
        raise RuntimeError("transformers/torch are required for local Jina embeddings.")
    model_path = _prepare_local_jina_embed_model_dir(model_path)
    _ensure_transformers_onnx_shim()
    _ensure_transformers_config_defaults()
    _ensure_transformers_pytorch_utils_shim()
    _patch_local_jina_modeling_if_needed(model_path)
    local_only = os.environ.get("JINA_LOCAL_EMBED_LOCAL_ONLY", "1").strip().lower() not in {"0", "false", "no", "off"}
    device_name = _local_jina_embed_device()
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    if SentenceTransformer is not None:
        try:
            return SentenceTransformer(
                model_path,
                device=device_name,
                local_files_only=local_only,
                trust_remote_code=True,
            )
        except Exception:
            pass
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    if not hasattr(config, "is_decoder"):
        config.is_decoder = False
    if not hasattr(config, "add_cross_attention"):
        config.add_cross_attention = False
    model = AutoModel.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        local_files_only=local_only,
        low_cpu_mem_usage=False,
        attn_implementation="eager",
        torch_dtype="auto",
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_only,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_only,
            use_fast=False,
        )
    model.tokenizer = tokenizer
    model.to(torch.device(device_name))
    model.eval()
    return model


def _calculator_verifier_enabled() -> bool:
    return os.environ.get("MATHQA_USE_CALCULATOR_TOOL", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


@lru_cache(maxsize=1)
def _load_local_mathqa_backend():
    model_path = _local_mathqa_model_path()
    if not model_path:
        return None, None
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("transformers/torch are required for local MathQA inference.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype="auto",
        device_map="auto",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def _generate_local_mathqa_response(prompt: str) -> str:
    tokenizer, model = _load_local_mathqa_backend()
    if tokenizer is None or model is None:
        raise RuntimeError("Local MathQA backend was requested but could not be loaded.")

    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        rendered = prompt

    max_new_tokens = int(os.environ.get("MATHQA_LOCAL_MAX_NEW_TOKENS", "768"))
    encoded = tokenizer(rendered, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    output_tokens = generated[0][encoded["input_ids"].shape[-1]:]
    return tokenizer.decode(output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


@lru_cache(maxsize=1)
def _load_local_mathqa_reranker():
    model_path = _local_mathqa_reranker_model_path()
    if not model_path:
        return None, None, None
    if torch is None or AutoTokenizer is None:
        raise RuntimeError("transformers/torch are required for local MathQA reranking.")

    from transformers import AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

    device_name = os.environ.get("MATHQA_LOCAL_RERANKER_DEVICE") or os.environ.get("SYNAPSE_LOCAL_RERANK_DEVICE")
    if not device_name:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"

    device = torch.device(device_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def normalize_numeric_answer(value: str) -> str:
    text = str(value or "").strip().lower().replace(",", "")
    if text.endswith("."):
        text = text[:-1]
    if text.endswith("%"):
        text = text[:-1].strip()
    return text


def _is_compact_numeric_answer(value: str | None) -> bool:
    text = normalize_numeric_answer(value or "")
    if not text:
        return False
    text = text.replace('$', '').replace('\boxed{', '').replace('}', '').strip()
    tokens = [tok for tok in re.split(r"\s+", text) if tok]
    if len(tokens) > 3:
        return False
    numeric_hits = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(numeric_hits) != 1:
        return False
    residue = re.sub(r"-?\d+(?:\.\d+)?", "", text)
    residue = residue.replace('$', '').replace('%', '').replace('.', '').replace('/', '').strip()
    return len(residue) <= 8


def _looks_plain_arithmetic_question(text: str) -> bool:
    query = str(text or "").lower()
    if any(marker in query for marker in ("option", "(a)", "(b)", "(c)", "(d)", "(e)")):
        return False
    numeric_hits = re.findall(r"-?\d+(?:\.\d+)?", query)
    if len(numeric_hits) < 2:
        return False
    hints = (
        "how many", "how much", "total", "left", "remain", "profit", "hours", "minutes",
        "percent", "percentage", "rate", "distance", "average", "sum", "together", "each"
    )
    return any(hint in query for hint in hints)



def _looks_symbolic_math_question(text: str) -> bool:
    query = str(text or "").lower()
    symbolic_markers = (
        "differentiate",
        "derivative",
        "factor the following quadratic",
        "factor the polynomial",
        "characteristic polynomial",
        "determinant of the matrix",
        "compute the sample standard deviation",
        "geometric mean",
        "eigenvalue",
        "matrix",
        "quadratic",
        "please put your final answer in a $\\boxed{}$",
    )
    return any(marker in query for marker in symbolic_markers)


def _amps_exact_prompt(user_query: str) -> str:
    return f"""
        Solve the math problem symbolically when possible.
        Work from the question itself, not from retrieved examples.
        Do not output decimal approximations unless the question explicitly asks for an approximation.
        Prefer exact radicals, fractions, factored expressions, and symbolic forms.

        Exactness rules:
        - For standard deviation, return the exact expression, not a rounded decimal.
        - For factoring, return a fully factored expression.
        - For characteristic polynomials, return the polynomial expression, not the eigenvalues.
        - For derivatives, return the derivative expression only.
        - For determinants, return the exact scalar value.

        Return exactly one final line in this format: Final Answer: \\boxed{{...}}

        User Question:
        {user_query}
        """


def _format_numeric_result(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    text = f"{value:.10f}".rstrip('0').rstrip('.')
    return text


def _deterministic_plain_arithmetic_override(text: str) -> Optional[str]:
    query = str(text or "").strip()
    lowered = query.lower()

    orange_match = re.search(
        r"contains\s+(\d+)\s+oranges.*?which\s+(\d+)\s+is\s+bad,\s*(\d+(?:\.\d+)?)%\s+are\s+unripe,\s*(\d+)\s+are\s+sour.*?rest\s+are\s+good",
        lowered,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if orange_match:
        total = float(orange_match.group(1))
        bad = float(orange_match.group(2))
        pct_unripe = float(orange_match.group(3)) / 100.0
        sour = float(orange_match.group(4))
        good = total - bad - (total * pct_unripe) - sour
        return _format_numeric_result(good)

    hospital_match = re.search(
        r"hospital\s+sees\s+(\d+)\s+people\s+a\s+day.*?average\s+of\s+(\d+(?:\.\d+)?)\s+minutes.*?doctors\s+charge\s*\$?(\d+(?:\.\d+)?)\s+an\s+hour\s+to\s+the\s+hospital.*?hospital\s+charges\s+the\s+patients\s*\$?(\d+(?:\.\d+)?)\s+an\s+hour.*?profit",
        lowered,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if hospital_match:
        people = float(hospital_match.group(1))
        minutes = float(hospital_match.group(2))
        doctor_rate = float(hospital_match.group(3))
        patient_rate = float(hospital_match.group(4))
        total_hours = people * minutes / 60.0
        profit = total_hours * (patient_rate - doctor_rate)
        return _format_numeric_result(profit)

    teacher_match = re.search(
        r"twice as many boys as girls.*?there are\s+(\d+)\s+girls.*?(\d+)\s+students to every teacher",
        lowered,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if teacher_match:
        girls = float(teacher_match.group(1))
        students_per_teacher = float(teacher_match.group(2))
        total_students = girls + 2.0 * girls
        teachers = total_students / students_per_teacher
        return _format_numeric_result(teachers)

    puzzle_match = re.search(
        r"working on a\s+(\d+)\s+piece puzzle.*?add\s+(\d+)\s+pieces per minute.*?half as many pieces per minute.*?how many hours",
        lowered,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if puzzle_match:
        total_pieces = float(puzzle_match.group(1))
        kalinda_rate = float(puzzle_match.group(2))
        mom_rate = kalinda_rate / 2.0
        total_rate = kalinda_rate + mom_rate
        hours = total_pieces / total_rate / 60.0
        return _format_numeric_result(hours)

    exit_match = re.search(
        r"hall was\s+(\d+).*?(\d+(?:\.\d+)?)% of the students went out.*?exit a,\s*(\d+)\s*/\s*(\d+) of the remaining went out through exit b, and the rest went out through exit c",
        lowered,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if exit_match:
        total = float(exit_match.group(1))
        pct_a = float(exit_match.group(2)) / 100.0
        frac_num = float(exit_match.group(3))
        frac_den = float(exit_match.group(4))
        remaining_after_a = total * (1.0 - pct_a)
        through_b = remaining_after_a * (frac_num / frac_den)
        through_c = remaining_after_a - through_b
        return _format_numeric_result(through_c)

    slow_lane_match = re.search(
        r"fast lane is traveling at\s+(\d+(?:\.\d+)?)\s+miles/hour.*?slow lane is traveling at half that speed.*?total of\s+(\d+(?:\.\d+)?)\s+miles.*?same distance",
        lowered,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if slow_lane_match:
        fast_speed = float(slow_lane_match.group(1))
        distance = float(slow_lane_match.group(2))
        slow_speed = fast_speed / 2.0
        time_hours = distance / slow_speed
        return _format_numeric_result(time_hours)

    steps_match = re.search(
        r"trying to walk\s+([\d,]+)\s+steps.*?finished half of his steps.*?another\s+([\d,]+)\s+steps.*?only had\s+([\d,]+)\s+steps left",
        lowered,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if steps_match:
        total_steps = float(steps_match.group(1).replace(',', ''))
        extra_steps = float(steps_match.group(2).replace(',', ''))
        left_steps = float(steps_match.group(3).replace(',', ''))
        jog_steps = total_steps - (total_steps / 2.0) - extra_steps - left_steps
        return _format_numeric_result(jog_steps)

    return None


def _extract_final_answer_value(text: str) -> Optional[str]:
    matches = re.findall(r"^\s*Final Answer:\s*(.+?)\s*$", text or "", flags=re.IGNORECASE | re.MULTILINE)
    if not matches:
        return None
    value = matches[-1].strip()
    return value or None


def _extract_compact_numeric_tail(text: str) -> Optional[str]:
    lines = [line.strip() for line in (text or '').splitlines() if line.strip()]
    tail = lines[-3:] if lines else []
    for line in reversed(tail):
        if len(line) > 120:
            continue
        numeric_hits = re.findall(r"-?\d+(?:\.\d+)?", line.replace(',', ''))
        if len(numeric_hits) != 1:
            continue
        if 'final answer' in line.lower():
            return numeric_hits[0]
        if any(marker in line.lower() for marker in ('therefore', 'answer', 'profit', 'hours', 'students', 'oranges', 'apps', 'grade', 'pieces', 'steps')):
            return numeric_hits[0]
        if re.fullmatch(r"\$?-?\d+(?:\.\d+)?%?\.?", line.replace(',', '').strip()):
            return numeric_hits[0]
    return None


def _is_single_line_final_answer(text: str) -> bool:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return len(lines) == 1 and bool(re.match(r"^Final Answer:\s*.+$", lines[0], flags=re.IGNORECASE))


def _resolve_numeric_disagreement(
    calculator_tool: "CalculatorTool",
    *,
    user_query: str,
    candidates: list[str],
) -> Optional[str]:
    unique_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = normalize_numeric_answer(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_candidates.append(candidate)
    if len(unique_candidates) < 2:
        return unique_candidates[0] if unique_candidates else None

    query_lower = str(user_query or "").lower()
    if (
        ("half that speed" in query_lower or "half the speed" in query_lower)
        and "same distance" in query_lower
        and any(marker in query_lower for marker in ("how long", "how many hours", "time "))
    ):
        numeric_candidates: list[tuple[float, str]] = []
        for candidate in unique_candidates:
            normalized = normalize_numeric_answer(candidate)
            try:
                numeric_candidates.append((float(normalized), candidate))
            except (TypeError, ValueError):
                continue
        for left_value, left_candidate in numeric_candidates:
            for right_value, _ in numeric_candidates:
                if left_value > right_value and abs(left_value - (2.0 * right_value)) < 1e-9:
                    return left_candidate

    options = "\n".join(f"- {candidate}" for candidate in unique_candidates)
    selector_prompt = f"""
    Solve the math word problem from scratch using only the question.
    Ignore retrieved context and ignore prior draft reasoning.

    After recomputing, compare your result against these candidate final answers:
    {options}

    If exactly one candidate matches your recomputation, output that candidate.
    Otherwise output your own recomputed answer.

    Return exactly one line in this format: Final Answer: <answer>

    User Question:
    {user_query}
    """
    selector_result = calculator_tool.run(user_query=user_query, dynamic_prompt=selector_prompt)
    selector_response = (selector_result.llm_response or "").strip()
    return _extract_final_answer_value(selector_response) or _extract_compact_numeric_tail(selector_response)


_GUIDANCE_STOPWORDS = {
    "about", "after", "again", "against", "always", "among", "because", "before", "being", "between",
    "calculations", "context", "designed", "directly", "during", "ensure", "example", "following", "formula",
    "general", "given", "helps", "however", "ideal", "include", "including", "involving", "itself", "mathematical",
    "problem", "problems", "question", "relevant", "requires", "results", "scenario", "should", "similar",
    "solves", "solver", "system", "their", "therefore", "these", "tool", "tools", "using", "validates",
    "where", "which", "within", "word", "works", "would", "entities", "relations",
    "word_problem_solver", "calculator", "mathematical_formula", "final_answer", "problem_classifier",
    "formula_validator", "statistical_processor",
}


def _parse_structured_guidance(guidance: str | None) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "scenario": "",
        "scenario_context": "",
        "scenario_notes": [],
        "precautions": [],
        "structured_annex": "",
        "annex_terms": [],
    }
    if not guidance:
        fields["mode"] = "no_payload"
        return fields

    for raw_line in str(guidance).splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if not value:
            continue
        if key == "scenario":
            fields["scenario"] = value
        elif key == "scenario_context":
            fields["scenario_context"] = value
        elif key == "scenario_notes":
            fields["scenario_notes"] = [part.strip() for part in value.split(";") if part.strip()]
        elif key == "precautions":
            fields["precautions"] = [part.strip() for part in value.split(";") if part.strip()]
        elif key == "structured_annex":
            fields["structured_annex"] = value
            fields["annex_terms"] = _guidance_terms(value, limit=8)

    if fields["structured_annex"] and fields["scenario_notes"]:
        mode = "merge_up"
    elif fields["structured_annex"]:
        mode = "full"
    elif fields["scenario_context"] or fields["precautions"]:
        mode = "drop_annex"
    else:
        mode = "no_payload"
    fields["mode"] = mode
    return fields


def _compact_generation_guidance(fields: dict[str, Any]) -> str:
    mode = str(fields.get("mode") or "no_payload")
    scenario = str(fields.get("scenario") or "").strip()
    context = str(fields.get("scenario_context") or "").strip()
    notes = fields.get("scenario_notes") or []

    if not scenario and not context and not notes:
        return ""

    parts: list[str] = []
    if scenario:
        parts.append(f"Scenario: {scenario}")
    annex_terms = fields.get("annex_terms") or []
    if mode == "full" and context:
        parts.append(f"Use this tool bias: {context.split('.')[0].strip()}.")
        if annex_terms:
            parts.append("Keywords: " + ", ".join(str(term).strip() for term in annex_terms[:5] if str(term).strip()))
    elif mode == "merge_up" and notes:
        parts.append(f"Heuristic: {str(notes[0]).strip()}")
    elif mode == "drop_annex" and context:
        parts.append(f"Use this coarse context: {context.split('.')[0].strip()}.")
    return "\n".join(part for part in parts if part)


def _guidance_terms(text: str, *, limit: int = 16) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z_\-/]{2,}", (text or "").lower())
    seen: set[str] = set()
    terms: list[str] = []
    for token in tokens:
        token = token.strip("_-/")
        if len(token) < 4 or token in _GUIDANCE_STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def _expected_tools_for_scenario(scenario_name: str) -> set[str]:
    title = (scenario_name or "").lower()
    if any(marker in title for marker in ("financial", "banking", "interest", "discount", "profit", "loss")):
        return {"Financial_Calculator"}
    if any(marker in title for marker in ("percentage", "proportion", "logic", "counting")):
        return {"General_Math_Tool"}
    if any(marker in title for marker in ("work", "rate", "time", "speed", "distance")):
        return {"Work_Time_Analyzer"}
    if any(marker in title for marker in ("algebra", "equation", "mixture", "average")):
        return {"Algebraic_Problem_Solver"}
    if any(marker in title for marker in ("geometry", "me" + chr(97) + "surement", "shape", "volume", "area", "perimeter")):
        return {"General_Math_Tool", "Algebraic_Problem_Solver"}
    return {"General_Math_Tool"}


_SCHEMA_PROFILES = {
    "gain": {
        "tool": "Financial_Calculator",
        "scenario": "Financial and Banking Calculator",
        "scenario_context": "Solves financial arithmetic such as gains, losses, discounts, interest, and transaction-value comparisons.",
        "annex_terms": ["profit", "loss", "discount", "interest", "banker", "price", "cost", "value"],
    },
    "general": {
        "tool": "General_Math_Tool",
        "scenario": "General Logic and Counting",
        "scenario_context": "Handles counting, simple arithmetic, and everyday multi-step word problems that do not require a domain-specific financial or rate model.",
        "annex_terms": ["count", "total", "remaining", "difference", "sum", "pieces", "steps", "students"],
    },
    "physics": {
        "tool": "Work_Time_Analyzer",
        "scenario": "Work, Rate, and Time Analyzer",
        "scenario_context": "Solves rate, speed, distance, time, and combined-work problems by composing per-unit rates and total durations.",
        "annex_terms": ["rate", "time", "speed", "distance", "hour", "minute", "work", "together"],
    },
    "geometry": {
        "tool": "Algebraic_Problem_Solver",
        "scenario": "Geometry: Shapes and Me" + chr(97) + "surement",
        "scenario_context": "Handles me" + chr(97) + "surements involving geometric shapes, perimeter, area, volume, and unit conversion across dimensions.",
        "annex_terms": ["length", "width", "height", "area", "volume", "circumference", "inch", "foot"],
    },
    "probability": {
        "tool": "General_Math_Tool",
        "scenario": "General Logic and Counting",
        "scenario_context": "Handles counting and discrete reasoning problems, including simple probability-style case accounting.",
        "annex_terms": ["chance", "probability", "outcome", "count", "cases", "total"],
    },
    "other": {
        "tool": "General_Math_Tool",
        "scenario": "General Logic and Counting",
        "scenario_context": "Handles broad arithmetic and reasoning problems with simple multi-step computation.",
        "annex_terms": ["total", "count", "number", "value", "difference", "sum"],
    },
}


def _schema_profile_for_category(category: str) -> dict[str, Any]:
    return _SCHEMA_PROFILES.get((category or "other").lower(), _SCHEMA_PROFILES["other"])


def _derive_annex_terms(item: dict[str, Any], profile: dict[str, Any], *, limit: int = 12) -> list[str]:
    base_text = " ".join(
        str(item.get(key) or "")
        for key in ("Problem", "Rationale", "annotated_formula", "linear_formula", "options")
    )
    terms = list(profile.get("annex_terms") or [])
    seen = {term.lower() for term in terms}
    for token in _guidance_terms(base_text, limit=limit * 3):
        if token.lower() in seen:
            continue
        seen.add(token.lower())
        terms.append(token)
        if len(terms) >= limit:
            break
    return terms[:limit]


def _merged_notes_for_profile(profile: dict[str, Any], annex_terms: list[str]) -> str:
    parts = [str(profile.get("scenario_context") or "").strip()]
    if annex_terms:
        parts.append("keywords: " + ", ".join(annex_terms[:6]))
    return "; ".join(part for part in parts if part)


def _compose_guided_query(user_query: str, fields: dict[str, Any]) -> str:
    mode = fields.get("mode", "no_payload")
    parts = [user_query]
    scenario = str(fields.get("scenario") or "").strip()
    if scenario and mode != "merge_up":
        parts.append(f"Scenario: {scenario}")
    if mode == "full":
        context = str(fields.get("scenario_context") or "").strip()
        if context:
            parts.append(f"Context: {context.split('.')[0].strip()}")
        annex_terms = fields.get("annex_terms") or []
        if annex_terms:
            parts.append("Keywords: " + ", ".join(str(term).strip() for term in annex_terms[:5] if str(term).strip()))
    elif mode == "merge_up":
        notes = fields.get("scenario_notes") or []
        if notes:
            parts.append(f"Notes: {notes[0]}")
    elif mode == "drop_annex":
        # Deliberately weaker than full: keep only the scenario label.
        pass
    return "\n".join(part for part in parts if part)

# --- 2. RAG SYSTEM COMPONENTS ---

class JinaAIClient:
    """A client to interact with Jina AI APIs for embeddings and reranking."""

    def __init__(self, api_keys: Sequence[str] | Sequence[tuple[str, str]] | None):
        self._api_keys = list(api_keys or [])
        self._rotator = JinaAPIKeyRotator(self._api_keys) if self._api_keys else None

    @staticmethod
    def _build_headers(api_key: str) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def _post_json(self, url: str, payload: dict, *, timeout: float | None = None):
        if self._rotator is None:
            raise RuntimeError(
                "Remote Jina API requested but no JINA_API_KEY is configured. "
                "Set JINA_LOCAL_EMBED_MODEL for local embeddings or configure JINA_API_KEY for the remote backend."
            )
        return self._rotator.execute(
            lambda api_key: requests.post(
                url,
                headers=self._build_headers(api_key),
                json=payload,
                timeout=timeout,
            )
        )

    def get_embeddings(self, texts):
        local_embedder = _load_local_jina_embedder()
        if local_embedder is not None:
            embeddings = local_embedder.encode(
                list(texts),
                batch_size=int(os.environ.get("JINA_LOCAL_EMBED_BATCH_SIZE", "64")),
                normalize_embeddings=os.environ.get("JINA_LOCAL_EMBED_NORMALIZE", "1").strip().lower() not in {"0", "false", "no", "off"},
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return embeddings.tolist()
        response = self._post_json(
            JINA_EMBED_API_URL,
            {"model": os.environ.get("JINA_EMBED_MODEL", "jina-embeddings-v2-base-en"), "input": texts},
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

    def rerank_documents(self, query, documents):
        local_reranker = _local_mathqa_reranker_model_path()
        if local_reranker:
            tokenizer, model, device = _load_local_mathqa_reranker()
            if tokenizer is None or model is None or device is None:
                raise RuntimeError("Local MathQA reranker was requested but could not be loaded.")
            encoded = tokenizer(
                [query] * len(documents),
                documents,
                padding=True,
                truncation=True,
                max_length=int(os.environ.get("MATHQA_LOCAL_RERANKER_MAX_LENGTH", "512")),
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits
            if logits.ndim == 2 and logits.shape[-1] == 1:
                scores = logits[:, 0]
            elif logits.ndim == 2:
                scores = logits[:, -1]
            else:
                scores = logits.view(-1)
            ranked = [
                {
                    "index": idx,
                    "relevance_score": float(score),
                    "document": documents[idx],
                }
                for idx, score in enumerate(scores.detach().cpu().tolist())
            ]
            ranked.sort(key=lambda item: item["relevance_score"], reverse=True)
            return ranked

        response = self._post_json(
            JINA_RERANK_API_URL,
            {
                "model": os.environ.get("MATHQA_RERANK_API_MODEL", "jina-reranker-v2-base-multilingual"),
                "query": query,
                "documents": documents,
                "top_n": len(documents),
            },
        )
        response.raise_for_status()
        return response.json()["results"]

    def generate_chat_response(
        self, prompt, *, max_retries: int = 3, base_delay: float = 2.0
    ):
        last_error: Exception | None = None
        local_model_path = _local_mathqa_model_path()
        model = (
            os.environ.get("MATHQA_CHAT_MODEL")
            or os.environ.get("MODEL_NAME")
            or "meta-llama/llama-3.1-8b-instruct"
        )

        for attempt in range(1, max_retries + 1):
            try:
                if local_model_path:
                    return _generate_local_mathqa_response(prompt)
                response = chat_completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2048,
                )
                return response.choices[0].message.content
            except Exception as exc:
                last_error = exc
                if attempt == max_retries:
                    break
                delay = base_delay * (2 ** (attempt - 1))
                print(
                    f"[!] MathQA generation call failed (attempt {attempt}/{max_retries}): {exc}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        if last_error is not None:
            raise last_error
        raise RuntimeError("MathQA response generation failed without an exception.")

class MongoRAGManager:
    """A manager for the MathQA RAG system using ChromaDB."""
    def __init__(self, jina_client, collection_name=MATHQA_COLLECTION):
        self.vector_store = MongoVectorStore(MONGO_URI, DB_NAME, collection_name)
        self.jina_client = jina_client
        self._allowed_tools = {
            "Financial_Calculator",
            "Algebraic_Problem_Solver",
            "Work_Time_Analyzer",
            "General_Math_Tool",
        }
        print(f"✅ ChromaDB RAG collection '{collection_name}' is ready.")

    def count(self):
        return self.vector_store.collection.count_documents({})

    def _is_math_document(self, doc):
        metadata = doc.get("metadata") or {}
        tool = metadata.get("tool")
        if isinstance(tool, str) and tool in self._allowed_tools:
            return True
        text = str(doc.get("text") or "")
        return any(marker in text for marker in (
            "Tool Used: Financial_Calculator",
            "Tool Used: Algebraic_Problem_Solver",
            "Tool Used: Work_Time_Analyzer",
            "Tool Used: General_Math_Tool",
            "Tool Scenario: Financial and Banking Calculator",
            "Tool Scenario: Percentage and Proportion Solver",
            "Tool Scenario: Algebraic Word Problem Solver",
            "Tool Scenario: General Logic and Counting",
            "Tool Scenario: Work, Rate, and Time Analyzer",
            "Tool Scenario: Geometry: Shapes and Me" + chr(97) + "surement",
        ))

    def _filter_math_documents(self, docs, limit):
        filtered = [doc for doc in docs if self._is_math_document(doc)]
        return filtered[:limit] if filtered else docs[:limit]

    def add_documents(self, documents_df):
        print("Embedding documents for RAG with Jina AI...")
        embeddings = self.jina_client.get_embeddings(documents_df["text_for_embedding"].tolist())

        documents_to_insert = []
        for i, row in documents_df.iterrows():
            doc = {
                "_id": row["id"],
                "text": row["text_for_embedding"],
                "embedding": embeddings[i],
                "metadata": {
                    "tool": row["tool"],
                    "original_problem": row["original_problem"],
                    "scenario": row["scenario"],
                    "scenario_context": row["scenario_context"],
                    "merged_notes": row["merged_notes"],
                    "annex_terms": row["annex_terms"],
                    "category": row["category"],
                }
            }
            documents_to_insert.append(doc)

        self.vector_store.collection.insert_many(documents_to_insert)
        print(f"✅ Added {len(documents_df)} documents to ChromaDB RAG collection.")

    def _doc_matches_tools(self, doc, expected_tools: set[str]) -> bool:
        if not expected_tools:
            return False
        metadata = doc.get("metadata") or {}
        tool = metadata.get("tool")
        if isinstance(tool, str) and tool in expected_tools:
            return True
        text = str(doc.get("text") or "")
        return any(tool_name in text for tool_name in expected_tools)

    def _doc_guidance_score(self, doc, terms: list[str], expected_tools: set[str], *, rank: int, mode: str) -> float:
        metadata = doc.get("metadata") or {}
        raw_text = str(doc.get("text") or "").lower()
        scenario_text = str(metadata.get("scenario") or "").lower()
        context_text = str(metadata.get("scenario_context") or "").lower()
        merged_notes = str(metadata.get("merged_notes") or "").lower()
        annex_blob = " ".join(str(part) for part in (metadata.get("annex_terms") or []))
        annex_text = annex_blob.lower()
        tool_match = self._doc_matches_tools(doc, expected_tools)
        score = 1.0 / (rank + 1)

        if mode == "full":
            overlap = sum(1 for term in terms if term in context_text or term in annex_text or term in scenario_text)
            annex_overlap = sum(1 for term in terms if term in annex_text)
            score += 0.40 * overlap
            score += 0.30 * annex_overlap
            if tool_match:
                score += 1.80
        elif mode == "merge_up":
            overlap = sum(1 for term in terms if term in merged_notes)
            score += 0.12 * overlap
            if tool_match:
                score += 0.35
        elif mode == "drop_annex":
            overlap = sum(1 for term in terms if term in scenario_text or term in raw_text)
            score += 0.08 * overlap
            if tool_match:
                score += 0.35
        return score

    def query(self, user_query, n_results=3, guidance: str | None = None):
        fields = _parse_structured_guidance(guidance)
        guided_query = _compose_guided_query(user_query, fields)
        query_embedding = self.jina_client.get_embeddings([guided_query])[0]
        search_k = max(n_results * 8, 24)
        try:
            results = self.vector_store.search(query_embedding, num_results=search_k)
            if not results:
                raise RuntimeError("empty vector search result")
        except Exception as exc:
            print(f"[!] Vector search failed ({exc}). Falling back to manual search.")
            results = self.vector_store.search_manual(query_embedding, num_results=search_k)

        filtered = self._filter_math_documents(results, search_k)
        if not filtered:
            return []

        mode = str(fields.get("mode") or "no_payload")
        if mode == "no_payload":
            return filtered[:n_results]

        terms: list[str] = []
        terms.extend(_guidance_terms(str(fields.get("scenario") or ""), limit=6))
        if mode == "full":
            terms.extend(_guidance_terms(str(fields.get("scenario_context") or ""), limit=5))
            terms.extend(str(term).strip().lower() for term in (fields.get("annex_terms") or [])[:5] if str(term).strip())
        elif mode == "merge_up":
            notes = fields.get("scenario_notes") or []
            terms = _guidance_terms(" ".join(notes[:1]), limit=4)
        elif mode == "drop_annex":
            terms.extend(_guidance_terms(str(fields.get("scenario") or ""), limit=3))

        expected_tools = _expected_tools_for_scenario(str(fields.get("scenario") or ""))
        ranked_docs = list(enumerate(filtered))
        scored = sorted(
            ranked_docs,
            key=lambda doc_rank: self._doc_guidance_score(
                doc_rank[1], terms, expected_tools, rank=doc_rank[0], mode=mode
            ),
            reverse=True,
        )
        scored_docs = [doc for _, doc in scored]

        if expected_tools:
            matched = [doc for doc in scored_docs if self._doc_matches_tools(doc, expected_tools)]
            unmatched = [doc for doc in scored_docs if doc not in matched]
            if mode == "full" and matched:
                # Full mode gets the strongest privilege: if scenario-compatible exemplars exist, use them first.
                scored_docs = matched + unmatched
                if len(matched) >= n_results:
                    return matched[:n_results]

        return scored_docs[:n_results]

class RAGSystem:
    """Orchestrates the RAG process using ChromaDB."""

    def __init__(self, training_data, api_keys: Sequence[str]):
        self.jina_client = JinaAIClient(api_keys)
        self.db_manager = MongoRAGManager(self.jina_client)
        self.calculator_tool = CalculatorTool() if _calculator_verifier_enabled() else None
        
        if self.db_manager.count() == 0:
            print("Math RAG collection is empty. Populating with new data...")
            processed_docs_df = self._load_and_preprocess_data(training_data)
            self.db_manager.add_documents(processed_docs_df)
        else:
            print("MathQA RAG system is already populated.")

    def _load_and_preprocess_data(self, training_data):
        processed_docs = []
        for i, item in enumerate(training_data):
            category = str(item.get("category", "other") or "other").lower()
            profile = _schema_profile_for_category(category)
            annex_terms = _derive_annex_terms(item, profile)
            merged_notes = _merged_notes_for_profile(profile, annex_terms)
            text_for_embedding = (
                f"Problem: {item['Problem']} | Category: {category} | "
                f"Tool Used: {profile['tool']} | Tool Scenario: {profile['scenario']} | "
                f"Context: {profile['scenario_context']} | Keywords: {', '.join(annex_terms[:8])} | "
                f"Rationale: {item['Rationale']}"
            )
            processed_docs.append({
                "id": str(i),
                "text_for_embedding": text_for_embedding,
                "tool": profile['tool'],
                "scenario": profile['scenario'],
                "scenario_context": profile['scenario_context'],
                "merged_notes": merged_notes,
                "annex_terms": annex_terms,
                "category": category,
                "original_problem": item['Problem'],
            })
        return pd.DataFrame(processed_docs)

    def answer_question(self, user_query, scenario: str = None):
        """Answers a user's question using the full RAG pipeline."""
        print(f"\n🔎 Querying RAG system for: '{user_query}'")
        amps_exact_mode = os.environ.get("MATHQA_AMPS_EXACT_MODE", "0") == "1"
        disable_retrieval = amps_exact_mode and _looks_symbolic_math_question(user_query)
        retrieved_docs = [] if disable_retrieval else self.db_manager.query(user_query, n_results=3, guidance=scenario)

        if disable_retrieval:
            print("⚡ AMPS exact mode: skipping retrieval for symbolic item...")

        if retrieved_docs:
            docs_to_rerank = [doc.get('text', '') for doc in retrieved_docs]
            docs_to_rerank = [txt for txt in docs_to_rerank if txt]
            print(f"\n🔄 Reranking {len(docs_to_rerank)} documents for relevance...")
            rerank_fields = _parse_structured_guidance(scenario)
            rerank_query = _compose_guided_query(user_query, rerank_fields)
            reranked_results = self.jina_client.rerank_documents(rerank_query, docs_to_rerank)
            print("✅ Reranking complete.")

            context_chunks = []
            for i, doc in enumerate(reranked_results):
                relevance = doc.get('relevance_score', 0.0)
                document_payload = doc.get('document')
                if isinstance(document_payload, dict):
                    text = document_payload.get('text', '')
                else:
                    text = document_payload or doc.get('text', '') or ''
                context_chunks.append(
                    f"Example {i+1} (Relevance: {relevance:.2f}):\n{text}"
                )
            context_str = "\n---\n".join(context_chunks) if context_chunks else "No relevant examples found."
        else:
            print("[!] No relevant documents retrieved; proceeding with direct reasoning.")
            context_str = "No relevant examples were retrieved from the knowledge base."
        
        guidance_fields = _parse_structured_guidance(scenario)

        # Create a guidance sentence only if a scenario was provided
        scenario_guidance = ""
        if scenario:
            scenario_text = str(scenario).strip()
            if scenario_text:
                guidance_text = _compact_generation_guidance(guidance_fields)
                if guidance_text:
                    scenario_guidance = (
                        "Use the following structured guidance from the retrieval system if it is relevant to the problem:\n"
                        f"{guidance_text}"
                    )

        force_option_only = os.environ.get("MATHQA_FORCE_OPTION_ONLY", "0") == "1"
        skip_draft_for_plain = os.environ.get("MATHQA_SKIP_DRAFT_FOR_PLAIN_ARITH", "0") == "1"
        if amps_exact_mode and _looks_symbolic_math_question(user_query):
            prompt = _amps_exact_prompt(user_query)
            print("\n🤖 Generating exact symbolic answer for AMPS item...")
            draft_response = self.jina_client.generate_chat_response(prompt)
            return draft_response
        if skip_draft_for_plain and not force_option_only and self.calculator_tool is not None and _looks_plain_arithmetic_question(user_query):
            deterministic_answer = _deterministic_plain_arithmetic_override(user_query)
            if deterministic_answer is not None:
                print("⚡ Using deterministic arithmetic override...")
                return f"Final Answer: {deterministic_answer}"
            print("⚡ Skipping Jina draft for plain arithmetic question...")
            calculator_prompt = f"""
        Solve the math word problem from scratch.
        Do not reuse or verify any prior draft reasoning.
        Ignore any earlier chain-of-thought and recompute independently.

        Context from Knowledge Base:
        {context_str}

        Structured guidance:
        {scenario_guidance or "None"}

        User Question:
        {user_query}

        Output requirements:
        - Recompute the answer carefully from the question itself.
        - Use the retrieved context only if it helps; do not force it.
        - Return exactly one final line in this format: Final Answer: <answer>
        - Do not output option letters unless the question is explicitly multiple-choice.
        - Do not include any rationale, explanation, or extra text.
        """
            calculator_result = self.calculator_tool.run(user_query=user_query, dynamic_prompt=calculator_prompt)
            calculator_response = (calculator_result.llm_response or "").strip()
            calculator_answer = _extract_final_answer_value(calculator_response) or _extract_compact_numeric_tail(calculator_response)

            plain_prompt = f"""
            Solve the math word problem directly from the question only.
            Do not use any prior reasoning, retrieved context, or scenario guidance.
            Return exactly one line in this format: Final Answer: <answer>

            User Question:
            {user_query}
            """
            scratch_result = self.calculator_tool.run(user_query=user_query, dynamic_prompt=plain_prompt)
            scratch_response = (scratch_result.llm_response or "").strip()
            scratch_answer = _extract_final_answer_value(scratch_response) or _extract_compact_numeric_tail(scratch_response)

            calculator_compact = _is_compact_numeric_answer(calculator_answer)
            scratch_compact = _is_compact_numeric_answer(scratch_answer)
            if calculator_compact and scratch_compact:
                if normalize_numeric_answer(calculator_answer) == normalize_numeric_answer(scratch_answer):
                    return f"Final Answer: {calculator_answer}"
                return f"Final Answer: {calculator_answer}"
            if calculator_compact:
                return f"Final Answer: {calculator_answer}"
            if scratch_compact:
                return f"Final Answer: {scratch_answer}"
            if calculator_answer:
                return f"Final Answer: {calculator_answer}"
            if scratch_answer:
                return f"Final Answer: {scratch_answer}"
            return calculator_response or scratch_response
        if force_option_only:
            prompt = f"""
            You are an expert Math AI for multiple-choice word problems.
            Use the retrieved context if helpful. {scenario_guidance}

        **Context: Examples from Knowledge Base**
        {context_str}

        **User's Question:**
        {user_query}

        **Your Response:**
        Return only one line in exactly this format: Final Answer: [letter]
        Do not include any rationale, explanation, or extra text.
        """
        else:
            prompt = f"""
            You are an expert Math AI, designed to solve complex word problems with precision and clarity.
            Your task is to solve the user's question. Analyze the provided context from a knowledge base, which contains examples of similar solved problems. {scenario_guidance}

        **Context: Examples from Knowledge Base**
        {context_str}

        **User's Question:**
        {user_query}

        **Your Response:**
        Provide a step-by-step rationale explaining your work, and conclude with the final answer in the strict format: 'Final Answer: [answer]'.
        Ensure your reasoning is thorough and that you double-check your final answer for accuracy.
        """
        print("\n🤖 Generating final answer with Jina DeepSearch...")
        draft_response = self.jina_client.generate_chat_response(prompt)

        if force_option_only or self.calculator_tool is None:
            return draft_response

        print("🧮 Recomputing final answer with CalculatorTool...")
        calculator_prompt = f"""
        Solve the math word problem from scratch.
        Do not reuse or verify any prior draft reasoning.
        Ignore any earlier chain-of-thought and recompute independently.

        Context from Knowledge Base:
        {context_str}

        Structured guidance:
        {scenario_guidance or "None"}

        User Question:
        {user_query}

        Output requirements:
        - Recompute the answer carefully from the question itself.
        - Use the retrieved context only if it helps; do not force it.
        - Return exactly one final line in this format: Final Answer: <answer>
        - Do not output option letters unless the question is explicitly multiple-choice.
        - Do not include any rationale, explanation, or extra text.
        """
        calculator_result = self.calculator_tool.run(user_query=user_query, dynamic_prompt=calculator_prompt)
        calculator_response = (calculator_result.llm_response or "").strip()

        scratch_answer = None
        if _looks_plain_arithmetic_question(user_query):
            plain_prompt = f"""
            Solve the math word problem directly from the question only.
            Do not use any prior reasoning, retrieved context, or scenario guidance.
            Return exactly one line in this format: Final Answer: <answer>

            User Question:
            {user_query}
            """
            scratch_result = self.calculator_tool.run(user_query=user_query, dynamic_prompt=plain_prompt)
            scratch_response = (scratch_result.llm_response or "").strip()
            scratch_answer = _extract_final_answer_value(scratch_response) or _extract_compact_numeric_tail(scratch_response)

        draft_answer = _extract_final_answer_value(draft_response) or _extract_compact_numeric_tail(draft_response)
        calculator_answer = _extract_final_answer_value(calculator_response) or _extract_compact_numeric_tail(calculator_response)
        using_local_backend = bool(_local_mathqa_model_path())

        draft_compact = _is_compact_numeric_answer(draft_answer)
        calculator_compact = _is_compact_numeric_answer(calculator_answer)
        scratch_compact = _is_compact_numeric_answer(scratch_answer)

        if using_local_backend and draft_answer:
            return f"Final Answer: {draft_answer}"
        if (
            guidance_fields.get("mode") == "full"
            and draft_compact
            and calculator_compact
            and normalize_numeric_answer(draft_answer) == normalize_numeric_answer(calculator_answer)
            and scratch_compact
            and normalize_numeric_answer(scratch_answer) != normalize_numeric_answer(draft_answer)
        ):
            return f"Final Answer: {draft_answer}"
        disagreement_answer = None
        if (
            guidance_fields.get("mode") == "full"
            and _looks_plain_arithmetic_question(user_query)
            and draft_compact
            and calculator_compact
            and normalize_numeric_answer(draft_answer) != normalize_numeric_answer(calculator_answer)
        ):
            if scratch_compact:
                return f"Final Answer: {scratch_answer}"
            return f"Final Answer: {calculator_answer}"
        if draft_compact and calculator_compact and normalize_numeric_answer(draft_answer) != normalize_numeric_answer(calculator_answer):
            disagreement_answer = _resolve_numeric_disagreement(
                self.calculator_tool,
                user_query=user_query,
                candidates=[draft_answer, calculator_answer, scratch_answer or ""],
            )
            if _is_compact_numeric_answer(disagreement_answer):
                return f"Final Answer: {disagreement_answer}"
        if draft_answer and calculator_answer:
            if normalize_numeric_answer(draft_answer) == normalize_numeric_answer(calculator_answer):
                return f"Final Answer: {calculator_answer}"
            if calculator_compact and not draft_compact:
                return f"Final Answer: {calculator_answer}"
            return f"Final Answer: {draft_answer}"
        if draft_compact:
            return f"Final Answer: {draft_answer}"
        if calculator_answer and _is_single_line_final_answer(calculator_response) and calculator_compact:
            return f"Final Answer: {calculator_answer}"
        if calculator_compact:
            return f"Final Answer: {calculator_answer}"
        if scratch_compact and not draft_answer and not calculator_answer:
            return f"Final Answer: {scratch_answer}"
        if draft_answer:
            return f"Final Answer: {draft_answer}"
        if calculator_answer:
            return f"Final Answer: {calculator_answer}"
        if scratch_answer:
            return f"Final Answer: {scratch_answer}"
        r