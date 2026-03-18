"""Tests for the TRIT-TRT Web UI components."""

import pytest
from trit_trt.trt_engine import TRTEngine
from trit_trt.config import TRTConfig, SelectionMethod, ReflectionDepth
from tests.test_engine import MockGenerator


class TestStreamingTRTEngine:
    def test_emits_events_in_order(self):
        from ui.streaming import StreamingTRTEngine

        events = []
        generator = MockGenerator()
        config = TRTConfig(
            rounds=1,
            candidates_per_round=2,
            reflection_depth=ReflectionDepth.MINIMAL,
        )

        def on_event(event: dict):
            events.append(event)

        engine = StreamingTRTEngine(generator, config, on_event=on_event)
        result = engine.run("What is 2+2?")

        event_types = [e["type"] for e in events]
        assert "status" in event_types
        assert "candidates" in event_types
        assert "selected" in event_types
        assert "result" in event_types
        assert result.text != ""

    def test_cancel_stops_between_rounds(self):
        from ui.streaming import StreamingTRTEngine

        events = []
        generator = MockGenerator()
        config = TRTConfig(
            rounds=5,
            candidates_per_round=2,
            reflection_depth=ReflectionDepth.MINIMAL,
            early_stop_threshold=1.0,
        )

        def on_event(event: dict):
            events.append(event)

        engine = StreamingTRTEngine(generator, config, on_event=on_event)
        engine.cancel()
        result = engine.run("Test")

        event_types = [e["type"] for e in events]
        assert "cancelled" in event_types
        assert result.rounds_used <= 1
