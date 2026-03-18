"""Streaming TRT Engine -- emits WebSocket events at each TRT phase."""

from __future__ import annotations
import logging
from typing import Callable, Optional

from trit_trt.trt_engine import (
    TRTEngine, TRTResult, TRTRoundResult, TRTCandidate, TextGenerator,
)
from trit_trt.config import TRTConfig
from trit_trt.knowledge_store import KnowledgeStore

logger = logging.getLogger("trit-trt.streaming")


class StreamingTRTEngine(TRTEngine):
    """TRTEngine subclass that emits events at each phase boundary."""

    def __init__(
        self,
        generator: TextGenerator,
        config: Optional[TRTConfig] = None,
        knowledge_store: Optional[KnowledgeStore] = None,
        on_event: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__(generator, config, knowledge_store)
        self._on_event = on_event or (lambda e: None)
        self._cancelled = False

    def emit(self, event: dict) -> None:
        """Send an event to the registered callback."""
        self._on_event(event)

    def cancel(self) -> None:
        """Request cancellation before the next phase."""
        self._cancelled = True

    def reset_cancel(self) -> None:
        """Clear the cancellation flag."""
        self._cancelled = False

    def run(self, prompt: str, max_new_tokens: int = 512) -> TRTResult:
        """Execute the TRT pipeline, emitting events at each phase boundary."""
        total_rounds = self.config.rounds
        result = TRTResult(text="", confidence=0.0, rounds_used=0)
        best_answer = ""
        best_confidence = 0.0

        for round_num in range(1, total_rounds + 1):
            if self._cancelled:
                self.emit({"type": "cancelled"})
                break

            # -- GENERATE (Thesis) --
            self.emit({
                "type": "status", "phase": "generating",
                "round": round_num, "total_rounds": total_rounds,
            })
            candidates = self._generate_phase(prompt, round_num, max_new_tokens)
            result.total_candidates_generated += len(candidates)
            self.emit({
                "type": "candidates",
                "count": len(candidates),
                "texts": [c.text[:300] for c in candidates],
                "round": round_num,
            })

            if self._cancelled:
                self.emit({"type": "cancelled"})
                break

            # -- SELECT (Antithesis) --
            self.emit({"type": "status", "phase": "selecting", "round": round_num, "total_rounds": total_rounds})
            selected, confidence = self._select_phase(candidates)
            self.emit({
                "type": "selected",
                "text": selected.text,
                "confidence": confidence,
                "round": round_num,
            })

            if self._cancelled:
                self.emit({"type": "cancelled"})
                break

            # -- REFLECT (Synthesis) --
            self.emit({"type": "status", "phase": "reflecting", "round": round_num, "total_rounds": total_rounds})
            insights = self._reflect_phase(prompt, candidates, selected, round_num)

            insight_confidence = {
                "minimal": 0.5, "standard": 0.6, "deep": 0.75,
            }.get(self.config.reflection_depth.value, 0.6)
            for insight in insights:
                self.emit({
                    "type": "insight",
                    "text": insight,
                    "confidence": insight_confidence,
                    "round": round_num,
                })

            # -- Record round --
            round_result = TRTRoundResult(
                round_number=round_num,
                candidates=candidates,
                selected=selected,
                confidence=confidence,
                insights=insights,
                knowledge_added=len(insights),
            )
            result.round_results.append(round_result)
            result.knowledge_log.extend(insights)

            if confidence >= best_confidence:
                best_answer = selected.text
                best_confidence = confidence
                result.rounds_used = round_num

            # -- Early stop check --
            if confidence >= self.config.early_stop_threshold:
                result.early_stopped = True
                break

        result.text = best_answer
        result.confidence = best_confidence
        result.rounds_used = len(result.round_results)

        self.emit({
            "type": "result",
            "text": result.text,
            "confidence": result.confidence,
            "rounds_used": result.rounds_used,
            "total_candidates": result.total_candidates_generated,
            "early_stopped": result.early_stopped,
            "knowledge_count": len(self.knowledge.entries),
        })
        return result
