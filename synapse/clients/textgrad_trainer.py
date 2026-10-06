from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from third_party.textgrad import BlackboxLLM, TextualGradientDescent, Variable
from third_party.textgrad.autograd.functional import sum as tg_sum
from third_party.textgrad_utils.prompt_complexity import calculate_text_complexity

from synapse.textgrad_support import TextGradSettings


@dataclass
class BatchTrainingResult:
    """
    Lightweight record capturing the outcome of a single TextGrad batch update.
    """

    batch_loss: float
    updated_loss: float
    accepted: bool
    complexity: float


class TextGradPromptTrainer:
    """
    Helper that mirrors FedTextGrad's prompt optimisation loop for a SYNAPSE client.
    """

    def __init__(self, settings: TextGradSettings) -> None:
        self.settings = settings

    def train_batches(
        self,
        dataloader: Iterable[Tuple[Sequence[str], Sequence[int]]],
        model: BlackboxLLM,
        optimizer: TextualGradientDescent,
        eval_fn,
        system_prompt: Variable,
        validation_samples: Sequence[Tuple[str, str]] | None = None,
        *,
        total_questions: Optional[int] = None,
        progress_callback: Callable[[int, int, int, int | None, bool], None] | None = None,
    ) -> List[BatchTrainingResult]:
        """
        Execute a lightweight TextGrad training loop using proximal updates.
        """
        results: List[BatchTrainingResult] = []
        max_steps = self.settings.max_steps
        step_counter = 0

        baseline_prompt = system_prompt.get_value()
        baseline_validation_score = None
        if validation_samples:
            baseline_validation_score = self._evaluate_on_samples(model, eval_fn, validation_samples)

        questions_processed = 0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()

            batch_loss = 0.0
            losses = []
            for x, y in zip(batch_x, batch_y):
                x_var = Variable(x, requires_grad=False, role_description="client query")
                y_val = self._coerce_label(y)
                y_var = Variable(y_val, requires_grad=False, role_description="ground truth answer")
                response = model(x_var)
                try:
                    eval_output = eval_fn(inputs={"prediction": response, "ground_truth_answer": y_var})
                except Exception:
                    eval_output = eval_fn([x_var, y_var, response])

                losses.append(eval_output)
                batch_loss += self._extract_accuracy(eval_output)

            batch_loss /= max(len(batch_x), 1)

            total_loss = tg_sum(losses)
            total_loss.backward(engine=optimizer.engine)
            optimizer.step()

            updated_loss = 0.0
            for x, y in zip(batch_x, batch_y):
                x_var = Variable(x, requires_grad=False, role_description="client query")
                y_val = self._coerce_label(y)
                y_var = Variable(y_val, requires_grad=False, role_description="ground truth answer")
                response = model(x_var)
                try:
                    eval_output = eval_fn(inputs={"prediction": response, "ground_truth_answer": y_var})
                except Exception:
                    eval_output = eval_fn([x_var, y_var, response])
                updated_loss += self._extract_accuracy(eval_output)

            updated_loss /= max(len(batch_x), 1)

            accepted_update = self._accept_update(batch_loss, updated_loss)
            validation_accept = True
            if validation_samples:
                new_validation_score = self._evaluate_on_samples(model, eval_fn, validation_samples)
                if baseline_validation_score is not None and new_validation_score < baseline_validation_score:
                    validation_accept = False
                else:
                    baseline_validation_score = new_validation_score

            if not (accepted_update and validation_accept):
                system_prompt.set_value(baseline_prompt)
            else:
                baseline_prompt = system_prompt.get_value()

            complexity = calculate_text_complexity(system_prompt.get_value())
            results.append(
                BatchTrainingResult(
                    batch_loss=batch_loss,
                    updated_loss=updated_loss,
                    accepted=accepted_update,
                    complexity=complexity,
                )
            )

            questions_processed += len(batch_x)
            step_counter += 1
            reached_step_limit = max_steps is not None and step_counter >= max_steps
            if total_questions is not None and progress_callback:
                processed = min(questions_processed, total_questions)
                progress_callback(
                    processed,
                    total_questions,
                    step_counter,
                    max_steps,
                    reached_step_limit and processed < total_questions,
                )

            if reached_step_limit:
                break

        return results

    def _accept_update(self, loss: float, updated_loss: float) -> bool:
        """
        Decide whether to keep the candidate prompt update following the proximal rule.
        """
        if not self.settings.proximal_update:
            return updated_loss >= loss

        if updated_loss <= loss and updated_loss != 1.0:
            return False
        return True

    @staticmethod
    def _extract_accuracy(eval_variable: Variable) -> float:
        """
        Pull the numerical score out of the evaluation variable.
        """
        value = eval_variable.get_value()
        match = re.search(r"<ACCURACY>\s*(\d+)\s*</ACCURACY>", value)
        if match:
            return float(match.group(1))
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _coerce_label(label) -> int:
        """
        Map various dataset label formats (raw ints, numpy scalars, strings) to an integer.
        """
        if isinstance(label, (int, np.integer)):
            return int(label)
        if isinstance(label, str):
            matches = re.findall(r"-?\d+", label)
            if matches:
                return int(matches[-1])
            return label
        return label

    @staticmethod
    def _evaluate_on_samples(model: BlackboxLLM, eval_fn, samples: Sequence[Tuple[str, str]]) -> float:
        """
        Evaluate the current prompt on a set of (question, answer) pairs.
        """
        if not samples:
            return 0.0
        total = 0.0
        count = 0
        for question, answer in samples:
            x_var = Variable(question, requires_grad=False, role_description="validation query")
            y_val = TextGradPromptTrainer._coerce_label(answer)
            y_var = Variable(y_val, requires_grad=False, role_description="validation ground truth")
            response = model(x_var)
            try:
                eval_output = eval_fn(inputs={"prediction": response, "ground_truth_answer": y_var})
            except Exception:
                eval_output = eval_fn([x_var, y_var, response])
            total += TextGradPromptTrainer._extract_accuracy(eval_output)
            count += 1
        return total / max(count, 1)
