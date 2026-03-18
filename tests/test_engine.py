"""
Tests for TRIT-TRT components.
Run: pytest tests/test_engine.py -v
"""

import pytest
from trit_trt.knowledge_store import KnowledgeStore, KnowledgeEntry
from trit_trt.trt_engine import TRTEngine, TRTCandidate
from trit_trt.config import TRTConfig, SelectionMethod, ReflectionDepth


# ─── Knowledge Store Tests ────────────────────────────────────


class TestKnowledgeStore:
    def test_add_and_query(self):
        store = KnowledgeStore(max_entries=10)
        entry = KnowledgeEntry(
            insight="Break complex problems into sub-problems",
            source_problem="abc123",
            round_number=1,
            confidence=0.8,
        )
        store.add(entry)
        assert len(store.entries) == 1

        results = store.query("any problem", top_k=5)
        assert len(results) == 1
        assert results[0].insight == "Break complex problems into sub-problems"

    def test_max_entries_pruning(self):
        store = KnowledgeStore(max_entries=3)
        for i in range(5):
            store.add(KnowledgeEntry(
                insight=f"Insight {i}",
                source_problem=f"prob_{i}",
                round_number=1,
                confidence=i * 0.2,
            ))
        assert len(store.entries) == 3
        # Should keep highest-confidence entries
        confidences = [e.confidence for e in store.entries]
        assert min(confidences) >= 0.4  # Low-confidence ones pruned

    def test_effectiveness_tracking(self):
        entry = KnowledgeEntry(
            insight="Test insight",
            source_problem="test",
            round_number=1,
            confidence=0.5,
        )
        store = KnowledgeStore()
        store.add(entry)
        store.record_outcome(entry, success=True)
        store.record_outcome(entry, success=True)
        store.record_outcome(entry, success=False)
        assert entry.effectiveness == pytest.approx(2 / 3)

    def test_format_for_prompt(self):
        store = KnowledgeStore()
        store.add(KnowledgeEntry(
            insight="Check boundary conditions",
            source_problem="test",
            round_number=1,
            confidence=0.9,
        ))
        formatted = store.format_for_prompt(store.entries)
        assert "Check boundary conditions" in formatted
        assert "90%" in formatted

    def test_empty_format(self):
        store = KnowledgeStore()
        assert store.format_for_prompt([]) == ""

    def test_json_roundtrip(self):
        store = KnowledgeStore()
        store.add(KnowledgeEntry(
            insight="Test roundtrip",
            source_problem="test",
            round_number=1,
            confidence=0.7,
        ))
        exported = store.export_json()
        restored = KnowledgeStore.from_json(exported)
        assert len(restored.entries) == 1
        assert restored.entries[0].insight == "Test roundtrip"

    def test_hash_problem(self):
        h1 = KnowledgeStore.hash_problem("What is 2+2?")
        h2 = KnowledgeStore.hash_problem("What is 2+2?")
        h3 = KnowledgeStore.hash_problem("What is 3+3?")
        assert h1 == h2
        assert h1 != h3


# ─── TRT Engine Tests ─────────────────────────────────────────


class MockGenerator:
    """Mock text generator for testing TRT logic."""

    def __init__(self, responses: list[str] = None):
        self.responses = responses or [
            "The answer is 42.",
            "I believe the answer is 42.",
            "42 is the result.",
            "The answer is 43.",  # Minority answer
        ]
        self._call_count = 0

    def generate(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        idx = self._call_count % len(self.responses)
        self._call_count += 1
        return self.responses[idx]

    def generate_batch(self, prompt: str, n: int, max_new_tokens: int = 512, **kwargs) -> list[str]:
        return [self.generate(prompt) for _ in range(n)]


class TestTRTEngine:
    def test_basic_run(self):
        generator = MockGenerator()
        config = TRTConfig(
            rounds=1,
            candidates_per_round=4,
            selection_method=SelectionMethod.SELF_CONSISTENCY,
            reflection_depth=ReflectionDepth.MINIMAL,
        )
        engine = TRTEngine(generator, config)
        result = engine.run("What is the meaning of life?")

        assert result.text != ""
        assert result.confidence > 0
        assert result.rounds_used == 1
        assert result.total_candidates_generated == 4

    def test_extract_answer(self):
        text = "Let me think about this.\n\nTherefore, the answer is 42."
        answer = TRTEngine._extract_answer(text)
        assert "42" in answer

    def test_parse_score(self):
        assert TRTEngine._parse_score("8/10") == 8.0
        assert TRTEngine._parse_score("Rating: 7") == 7.0
        assert TRTEngine._parse_score("no number here") == 5.0

    def test_parse_insights(self):
        text = """
        1. Break problems into smaller sub-problems
        2. Verify intermediate results before proceeding
        3. Consider edge cases explicitly
        """
        insights = TRTEngine._parse_insights(text)
        assert len(insights) == 3
        assert "Break problems" in insights[0]

    def test_knowledge_accumulation(self):
        generator = MockGenerator()
        config = TRTConfig(
            rounds=2,
            candidates_per_round=4,
            reflection_depth=ReflectionDepth.MINIMAL,
            early_stop_threshold=1.0,  # Don't early stop
        )
        engine = TRTEngine(generator, config)
        engine.run("Test problem")

        # Knowledge should have been accumulated
        assert len(engine.knowledge.entries) > 0


# ─── Config Tests ──────────────────────────────────────────────


class TestConfig:
    def test_default_config(self):
        from trit_trt.config import TritTRTConfig
        config = TritTRTConfig()
        assert config.model_id == "microsoft/BitNet-b1.58-2B-4T"
        assert config.trt.rounds == 3

    def test_yaml_roundtrip(self, tmp_path):
        from trit_trt.config import TritTRTConfig
        config = TritTRTConfig(model_id="test-model")
        path = tmp_path / "test.yaml"
        config.to_yaml(path)
        loaded = TritTRTConfig.from_yaml(path)
        assert loaded.model_id == "test-model"
