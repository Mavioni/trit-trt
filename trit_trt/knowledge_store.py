"""
Knowledge Store — Persistent Insight Accumulator
Yunis AI — TRIT-TRT

Stores and retrieves generalizable insights from the TRT reflection phase.
Insights persist within a session and can transfer across problems.
Inspired by the TRT MCP server pattern from Zhuang et al.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json
import hashlib


@dataclass
class KnowledgeEntry:
    """A single insight extracted from a TRT reflection round."""
    insight: str                          # The generalizable insight text
    source_problem: str                   # Hash of the problem that generated it
    round_number: int                     # Which TRT round produced it
    confidence: float                     # How reliable this insight is (0-1)
    success_count: int = 0                # Times this insight led to better solutions
    failure_count: int = 0                # Times it didn't help
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: list[str] = field(default_factory=list)

    @property
    def effectiveness(self) -> float:
        """Ratio of successful applications."""
        total = self.success_count + self.failure_count
        if total == 0:
            return self.confidence  # Prior estimate
        return self.success_count / total

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


class KnowledgeStore:
    """
    Accumulates and retrieves insights across TRT rounds and problems.

    The store acts as a session-level memory that grows smarter over time.
    Each reflection round deposits new insights; each generation round
    queries relevant insights to inform solution candidates.

    This is the dialectical memory — thesis generates, antithesis selects,
    synthesis reflects, and the knowledge store carries the synthesis forward.
    """

    def __init__(self, max_entries: int = 50):
        self.entries: list[KnowledgeEntry] = []
        self.max_entries = max_entries
        self._problem_history: list[str] = []

    def add(self, entry: KnowledgeEntry) -> None:
        """Add a new insight to the store."""
        self.entries.append(entry)
        # Prune lowest-effectiveness entries if over capacity
        if len(self.entries) > self.max_entries:
            self.entries.sort(key=lambda e: e.effectiveness, reverse=True)
            self.entries = self.entries[:self.max_entries]

    def query(
        self,
        problem: str,
        top_k: int = 5,
        min_confidence: float = 0.3,
    ) -> list[KnowledgeEntry]:
        """
        Retrieve the most relevant insights for a given problem.
        Uses effectiveness score and recency for ranking.
        """
        candidates = [
            e for e in self.entries
            if e.confidence >= min_confidence
        ]
        # Score by effectiveness (60%) + recency (40%)
        now = datetime.now()
        scored = []
        for entry in candidates:
            created = datetime.fromisoformat(entry.created_at)
            age_hours = max((now - created).total_seconds() / 3600, 0.1)
            recency_score = 1.0 / (1.0 + age_hours / 24.0)  # Decay over days
            combined = 0.6 * entry.effectiveness + 0.4 * recency_score
            scored.append((combined, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def record_outcome(self, entry: KnowledgeEntry, success: bool) -> None:
        """Record whether an insight led to a better solution."""
        if success:
            entry.success_count += 1
        else:
            entry.failure_count += 1

    def format_for_prompt(self, entries: list[KnowledgeEntry]) -> str:
        """Format knowledge entries as context for the generation prompt."""
        if not entries:
            return ""

        lines = ["[Accumulated insights from previous reasoning rounds:]"]
        for i, entry in enumerate(entries, 1):
            eff = f"{entry.effectiveness:.0%}"
            lines.append(
                f"  {i}. [{eff} effective] {entry.insight}"
            )
        return "\n".join(lines)

    @staticmethod
    def hash_problem(problem: str) -> str:
        """Create a stable hash for problem deduplication."""
        return hashlib.sha256(problem.encode()).hexdigest()[:16]

    def get_session_stats(self) -> dict:
        """Return summary statistics for the current session."""
        if not self.entries:
            return {"total": 0, "avg_confidence": 0, "avg_effectiveness": 0}
        return {
            "total": len(self.entries),
            "avg_confidence": sum(e.confidence for e in self.entries) / len(self.entries),
            "avg_effectiveness": sum(e.effectiveness for e in self.entries) / len(self.entries),
            "unique_problems": len(set(e.source_problem for e in self.entries)),
        }

    def export_json(self) -> str:
        """Serialize the store for persistence."""
        return json.dumps(
            [e.to_dict() for e in self.entries],
            indent=2,
        )

    @classmethod
    def from_json(cls, data: str, max_entries: int = 50) -> KnowledgeStore:
        """Deserialize a store from JSON."""
        store = cls(max_entries=max_entries)
        for item in json.loads(data):
            store.entries.append(KnowledgeEntry(**item))
        return store
