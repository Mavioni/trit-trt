"""
TRT Engine — Dialectical Reasoning Layer
Yunis AI — TRIT-TRT

Implements the Test-time Recursive Thinking (TRT) loop from Zhuang et al.
This is the dialectical core: Generate (thesis) → Select (antithesis) → Reflect (synthesis).

The model self-improves at inference time without external feedback by:
1. Generating multiple solution candidates
2. Selecting the best via self-consistency voting
3. Reflecting on why winners won and losers lost
4. Carrying accumulated insights into the next round

This maps directly to the ternary thesis:
  -1 (antithesis) rejects weak candidates
   0 (synthesis)  reflects and extracts patterns
  +1 (thesis)     generates new, improved candidates
"""

from __future__ import annotations
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol, Optional

from .config import TRTConfig, SelectionMethod, ReflectionDepth
from .knowledge_store import KnowledgeStore, KnowledgeEntry

logger = logging.getLogger("trit-trt.trt")


# ─── Protocols ─────────────────────────────────────────────────

class TextGenerator(Protocol):
    """Any engine that can generate text (BitNet or AirLLM)."""

    def generate(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str: ...
    def generate_batch(self, prompt: str, n: int, max_new_tokens: int = 512, **kwargs) -> list[str]: ...


# ─── Result Types ──────────────────────────────────────────────

@dataclass
class TRTCandidate:
    """A single solution candidate."""
    text: str
    round_number: int
    score: float = 0.0
    is_selected: bool = False


@dataclass
class TRTRoundResult:
    """Result from one TRT round."""
    round_number: int
    candidates: list[TRTCandidate]
    selected: TRTCandidate
    confidence: float
    insights: list[str]
    knowledge_added: int = 0


@dataclass
class TRTResult:
    """Final result from the full TRT pipeline."""
    text: str                                      # Best final answer
    confidence: float                              # Self-consistency score (0-1)
    rounds_used: int                               # Total rounds executed
    round_results: list[TRTRoundResult] = field(default_factory=list)
    knowledge_log: list[str] = field(default_factory=list)
    total_candidates_generated: int = 0
    early_stopped: bool = False


# ─── TRT Engine ────────────────────────────────────────────────

class TRTEngine:
    """
    Test-time Recursive Thinking engine.

    Wraps any text generator in a dialectical loop that progressively
    improves output quality through self-directed reflection.

    The loop:
    ┌────────────────────────────────────────────────────┐
    │  Round N                                            │
    │  1. GENERATE: Produce K candidates with knowledge  │
    │  2. SELECT:   Vote on best candidate               │
    │  3. REFLECT:  Analyze success/failure patterns     │
    │  4. STORE:    Add insights to knowledge store      │
    │  5. CHECK:    If confidence >= threshold, stop      │
    │  └── else → Round N+1 with accumulated knowledge   │
    └────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        generator: TextGenerator,
        config: Optional[TRTConfig] = None,
        knowledge_store: Optional[KnowledgeStore] = None,
    ):
        self.generator = generator
        self.config = config or TRTConfig()
        self.knowledge = knowledge_store or KnowledgeStore(
            max_entries=self.config.max_knowledge_entries,
        )

    def run(
        self,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> TRTResult:
        """
        Execute the full TRT pipeline on a prompt.

        Returns the best refined answer after all rounds complete
        or early stopping triggers.
        """
        logger.info(
            f"TRT starting: {self.config.rounds} rounds, "
            f"{self.config.candidates_per_round} candidates/round"
        )

        result = TRTResult(
            text="",
            confidence=0.0,
            rounds_used=0,
        )

        best_answer = ""
        best_confidence = 0.0

        for round_num in range(1, self.config.rounds + 1):
            logger.info(f"  Round {round_num}/{self.config.rounds}")

            # ── Phase 1: GENERATE (Thesis) ───────────────────
            candidates = self._generate_phase(
                prompt=prompt,
                round_number=round_num,
                max_new_tokens=max_new_tokens,
            )
            result.total_candidates_generated += len(candidates)

            # ── Phase 2: SELECT (Antithesis) ─────────────────
            selected, confidence = self._select_phase(candidates)

            # ── Phase 3: REFLECT (Synthesis) ─────────────────
            insights = self._reflect_phase(
                prompt=prompt,
                candidates=candidates,
                selected=selected,
                round_number=round_num,
            )

            # ── Record round ─────────────────────────────────
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

            # Track best
            if confidence >= best_confidence:
                best_answer = selected.text
                best_confidence = confidence
                result.rounds_used = round_num

            logger.info(
                f"    Confidence: {confidence:.2%} | "
                f"Insights: {len(insights)} | "
                f"Knowledge store: {len(self.knowledge.entries)}"
            )

            # ── Early stop check ─────────────────────────────
            if confidence >= self.config.early_stop_threshold:
                logger.info(
                    f"  Early stop at round {round_num} "
                    f"(confidence {confidence:.2%} >= {self.config.early_stop_threshold:.2%})"
                )
                result.early_stopped = True
                break

        result.text = best_answer
        result.confidence = best_confidence
        result.rounds_used = len(result.round_results)

        logger.info(
            f"TRT complete: {result.rounds_used} rounds, "
            f"confidence {result.confidence:.2%}, "
            f"{'early stopped' if result.early_stopped else 'all rounds used'}"
        )

        return result

    # ─── Phase Implementations ─────────────────────────────────

    def _generate_phase(
        self,
        prompt: str,
        round_number: int,
        max_new_tokens: int,
    ) -> list[TRTCandidate]:
        """
        THESIS: Generate N solution candidates.
        Incorporates accumulated knowledge from previous rounds.
        """
        # Query knowledge store for relevant insights
        relevant_knowledge = self.knowledge.query(prompt, top_k=5)
        knowledge_context = self.knowledge.format_for_prompt(relevant_knowledge)

        # Build augmented prompt with knowledge
        if knowledge_context and round_number > 1:
            augmented_prompt = (
                f"{knowledge_context}\n\n"
                f"Using the insights above, provide a thorough answer to:\n\n"
                f"{prompt}"
            )
        else:
            augmented_prompt = prompt

        # Generate N candidates
        raw_outputs = self.generator.generate_batch(
            prompt=augmented_prompt,
            n=self.config.candidates_per_round,
            max_new_tokens=max_new_tokens,
        )

        return [
            TRTCandidate(
                text=text,
                round_number=round_number,
            )
            for text in raw_outputs
        ]

    def _select_phase(
        self,
        candidates: list[TRTCandidate],
    ) -> tuple[TRTCandidate, float]:
        """
        ANTITHESIS: Evaluate and select the best candidate.
        Returns (selected_candidate, confidence_score).
        """
        method = self.config.selection_method

        if method == SelectionMethod.SELF_CONSISTENCY:
            return self._select_by_consistency(candidates)
        elif method == SelectionMethod.VERIFICATION:
            return self._select_by_verification(candidates)
        else:  # HYBRID
            return self._select_hybrid(candidates)

    def _select_by_consistency(
        self,
        candidates: list[TRTCandidate],
    ) -> tuple[TRTCandidate, float]:
        """
        Majority vote: extract the final answer from each candidate,
        count frequencies, select the most common.
        """
        # Extract "final answers" — last substantive sentence
        answers = []
        for c in candidates:
            answer = self._extract_answer(c.text)
            answers.append(answer)

        # Count answer frequencies (normalized)
        counter = Counter(answers)
        most_common, count = counter.most_common(1)[0]
        confidence = count / len(candidates)

        # Find the best candidate matching the majority answer
        for i, c in enumerate(candidates):
            if answers[i] == most_common:
                c.is_selected = True
                c.score = confidence
                return c, confidence

        # Fallback: first candidate
        candidates[0].is_selected = True
        return candidates[0], 1.0 / len(candidates)

    def _select_by_verification(
        self,
        candidates: list[TRTCandidate],
    ) -> tuple[TRTCandidate, float]:
        """
        Ask the model to verify each candidate's correctness.
        Select the one with highest verification score.
        """
        best_candidate = candidates[0]
        best_score = 0.0

        for c in candidates:
            verify_prompt = (
                f"Evaluate the following answer for correctness, completeness, "
                f"and clarity. Rate it from 0 to 10.\n\n"
                f"Answer:\n{c.text[:500]}\n\n"
                f"Rating (just the number):"
            )
            try:
                response = self.generator.generate(
                    verify_prompt, max_new_tokens=16
                )
                score = self._parse_score(response)
                c.score = score / 10.0
                if score > best_score:
                    best_score = score
                    best_candidate = c
            except Exception as e:
                logger.warning(f"Verification failed: {e}")
                c.score = 0.5

        best_candidate.is_selected = True
        return best_candidate, best_candidate.score

    def _select_hybrid(
        self,
        candidates: list[TRTCandidate],
    ) -> tuple[TRTCandidate, float]:
        """Combine consistency and verification scores."""
        consistency_winner, cons_conf = self._select_by_consistency(candidates)
        verify_winner, ver_conf = self._select_by_verification(candidates)

        # Weighted average: consistency 60%, verification 40%
        cons_total = 0.6 * cons_conf
        ver_total = 0.4 * ver_conf

        if cons_total >= ver_total:
            return consistency_winner, cons_conf
        return verify_winner, ver_conf

    def _reflect_phase(
        self,
        prompt: str,
        candidates: list[TRTCandidate],
        selected: TRTCandidate,
        round_number: int,
    ) -> list[str]:
        """
        SYNTHESIS: Analyze why the selected answer was best
        and extract generalizable insights.
        """
        depth = self.config.reflection_depth

        if depth == ReflectionDepth.MINIMAL:
            return self._reflect_minimal(selected)
        elif depth == ReflectionDepth.STANDARD:
            return self._reflect_standard(prompt, candidates, selected, round_number)
        else:
            return self._reflect_deep(prompt, candidates, selected, round_number)

    def _reflect_minimal(self, selected: TRTCandidate) -> list[str]:
        """Extract one key takeaway."""
        reflect_prompt = (
            f"In one sentence, what makes this answer effective?\n\n"
            f"{selected.text[:300]}\n\n"
            f"Key insight:"
        )
        insight = self.generator.generate(
            reflect_prompt,
            max_new_tokens=128,
        )
        if insight.strip():
            self._store_insight(insight.strip(), "", selected.round_number, 0.5)
            return [insight.strip()]
        return []

    def _reflect_standard(
        self,
        prompt: str,
        candidates: list[TRTCandidate],
        selected: TRTCandidate,
        round_number: int,
    ) -> list[str]:
        """Analyze patterns across successful and failed candidates."""
        # Build a summary of candidates for reflection
        candidate_summaries = []
        for i, c in enumerate(candidates[:4]):  # Limit to 4 for context
            status = "SELECTED" if c.is_selected else "rejected"
            candidate_summaries.append(
                f"Candidate {i+1} ({status}, score={c.score:.2f}):\n"
                f"{c.text[:200]}..."
            )

        reflect_prompt = (
            f"You are analyzing solution attempts for this problem:\n"
            f"{prompt[:200]}\n\n"
            f"Here are the candidates:\n"
            f"{''.join(candidate_summaries)}\n\n"
            f"Identify 1-3 generalizable insights about what made the "
            f"winning approach better. Focus on strategies that would "
            f"apply to similar problems. Be concise.\n\n"
            f"Insights:"
        )

        response = self.generator.generate(
            reflect_prompt,
            max_new_tokens=self.config.reflection_max_tokens,
        )

        insights = self._parse_insights(response)
        problem_hash = KnowledgeStore.hash_problem(prompt)

        for insight in insights:
            self._store_insight(insight, problem_hash, round_number, 0.6)

        return insights

    def _reflect_deep(
        self,
        prompt: str,
        candidates: list[TRTCandidate],
        selected: TRTCandidate,
        round_number: int,
    ) -> list[str]:
        """Full contrastive analysis with generalization."""
        # Find a rejected candidate for contrast
        rejected = next(
            (c for c in candidates if not c.is_selected),
            candidates[-1],
        )

        reflect_prompt = (
            f"Compare these two solutions to the same problem.\n\n"
            f"Problem: {prompt[:200]}\n\n"
            f"WINNING solution (score {selected.score:.2f}):\n"
            f"{selected.text[:300]}\n\n"
            f"LOSING solution (score {rejected.score:.2f}):\n"
            f"{rejected.text[:300]}\n\n"
            f"Perform a contrastive analysis:\n"
            f"1. What specific strategy made the winner better?\n"
            f"2. What error pattern should be avoided?\n"
            f"3. What generalizable principle applies to similar problems?\n\n"
            f"Be specific and actionable."
        )

        response = self.generator.generate(
            reflect_prompt,
            max_new_tokens=self.config.reflection_max_tokens,
        )

        insights = self._parse_insights(response)
        problem_hash = KnowledgeStore.hash_problem(prompt)

        for insight in insights:
            self._store_insight(insight, problem_hash, round_number, 0.75)

        return insights

    # ─── Utility Methods ───────────────────────────────────────

    def _store_insight(
        self,
        insight: str,
        problem_hash: str,
        round_number: int,
        confidence: float,
    ) -> None:
        """Add an insight to the knowledge store."""
        entry = KnowledgeEntry(
            insight=insight,
            source_problem=problem_hash,
            round_number=round_number,
            confidence=confidence,
        )
        self.knowledge.add(entry)

    @staticmethod
    def _extract_answer(text: str) -> str:
        """
        Extract the core answer from a response.
        Used for self-consistency voting.
        """
        # Take last meaningful paragraph as the "answer"
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return text.strip()[:100]

        # Use last paragraph, normalized
        answer = paragraphs[-1].strip().lower()
        # Remove common filler
        for prefix in ["therefore", "in conclusion", "so", "thus", "hence"]:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].lstrip(", ")
        return answer[:200]

    @staticmethod
    def _parse_score(response: str) -> float:
        """Parse a numeric score from verification response."""
        import re
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", response)
        if numbers:
            score = float(numbers[0])
            return min(max(score, 0), 10)
        return 5.0  # Default middle score

    @staticmethod
    def _parse_insights(response: str) -> list[str]:
        """Parse structured insights from reflection response."""
        insights = []
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove numbering and bullet markers
            import re
            line = re.sub(r"^[\d]+[.)]\s*", "", line)
            line = re.sub(r"^[-*•]\s*", "", line)
            line = line.strip()
            if len(line) > 20:  # Skip very short lines
                insights.append(line)

        return insights[:5]  # Cap at 5 insights per round
