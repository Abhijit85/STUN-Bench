from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from third_party.textgrad import EngineLM, get_engine, set_backward_engine


@dataclass
class TextGradSettings:
    """
    Runtime configuration describing how TextGrad should be used inside SYNAPSE.
    """

    enabled: bool = False
    evaluation_engine_name: Optional[str] = None
    test_engine_name: Optional[str] = None
    aggregate_method: str = "summarization"
    proximal_update: bool = True
    batch_size: int = 3
    max_steps: Optional[int] = None

    evaluation_engine: Optional[EngineLM] = field(default=None, init=False)
    test_engine: Optional[EngineLM] = field(default=None, init=False)

    def ensure_engines(self) -> None:
        """
        Lazily instantiate the LLM engines required for TextGrad optimisation.
        """
        if not self.enabled:
            return

        if self.evaluation_engine is None and self.evaluation_engine_name:
            self.evaluation_engine = get_engine(engine_name=self.evaluation_engine_name)

        if self.test_engine is None and self.test_engine_name:
            if self.test_engine_name == self.evaluation_engine_name and self.evaluation_engine is not None:
                self.test_engine = self.evaluation_engine
            else:
                self.test_engine = get_engine(engine_name=self.test_engine_name)

        if self.evaluation_engine:
            set_backward_engine(self.evaluation_engine, override=True)


def textgrad_enabled_from_env() -> bool:
    """
    Helper to parse the global toggle from environment variables.
    """
    import os

    toggle = os.environ.get("SYNAPSE_TEXTGRAD_ENABLED", "").strip().lower()
    return toggle in {"1", "true", "yes", "on"}
