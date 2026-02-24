"""Tests for norai_tools.rewards."""

import pytest

from norai_tools.rewards import language_reward, length_reward, accuracy_reward


def _wrap(text: str) -> list[dict]:
    """Wrap text in TRL completion format."""
    return [{"role": "assistant", "content": text}]


class TestLanguageReward:
    def test_empty_text_zero_reward(self):
        rewards = language_reward([_wrap("")])
        assert rewards == [0.0]

    def test_whitespace_only_zero_reward(self):
        rewards = language_reward([_wrap("   ")])
        assert rewards == [0.0]

    def test_norwegian_text_high_reward(self):
        text = "Blåbær og rømme er ærlig talt det beste på en vaffel."
        rewards = language_reward([_wrap(text)])
        assert rewards[0] > 0.5

    def test_english_text_base_reward(self):
        text = "This is a sentence in English without any Norwegian characters."
        rewards = language_reward([_wrap(text)])
        # Should get base reward of 0.5 (non-empty but no Norwegian chars)
        assert rewards[0] == pytest.approx(0.5)

    def test_batch_processing(self):
        completions = [
            _wrap("Norsk tekst med æøå"),
            _wrap("English text only"),
            _wrap(""),
        ]
        rewards = language_reward(completions)
        assert len(rewards) == 3
        assert rewards[0] > rewards[1]
        assert rewards[2] == 0.0


class TestLengthReward:
    def test_similar_length_full_reward(self):
        rewards = length_reward(
            [_wrap("Omtrent like lang tekst som referansen her.")],
            ["Omtrent like lang tekst som referansen her!"],
        )
        assert rewards[0] == 1.0

    def test_slightly_shorter_full_reward(self):
        ref = "Dette er en referansetekst som er ganske lang."
        comp = "Dette er en kortere referansetekst."
        rewards = length_reward([_wrap(comp)], [ref])
        assert rewards[0] >= 0.5

    def test_much_shorter_low_reward(self):
        ref = "Dette er en lang referansetekst med mange ord og setninger for testing."
        comp = "Kort."
        rewards = length_reward([_wrap(comp)], [ref])
        assert rewards[0] == 0.0

    def test_much_longer_low_reward(self):
        ref = "Kort referanse."
        comp = "x " * 200  # Very long
        rewards = length_reward([_wrap(comp)], [ref])
        assert rewards[0] == 0.0

    def test_batch_processing(self):
        completions = [_wrap("a" * 50), _wrap("b" * 10)]
        references = ["x" * 50, "y" * 50]
        rewards = length_reward(completions, references)
        assert len(rewards) == 2
        assert rewards[0] > rewards[1]


class TestAccuracyReward:
    def test_exact_match_classification(self):
        rewards = accuracy_reward(
            [_wrap("positiv")],
            ["positiv"],
            ["classification"],
        )
        assert rewards[0] == 1.0

    def test_exact_match_case_insensitive(self):
        rewards = accuracy_reward(
            [_wrap("Positiv")],
            ["positiv"],
            ["classification"],
        )
        assert rewards[0] == 1.0

    def test_partial_match(self):
        rewards = accuracy_reward(
            [_wrap("Oslo er hovedstaden")],
            ["Oslo"],
            ["qa"],
        )
        assert rewards[0] == 0.5

    def test_no_match(self):
        rewards = accuracy_reward(
            [_wrap("Bergen")],
            ["Oslo"],
            ["classification"],
        )
        assert rewards[0] == 0.0

    def test_non_verifiable_task(self):
        rewards = accuracy_reward(
            [_wrap("Et vakkert dikt om våren.")],
            ["En vakker tekst om vår."],
            ["creative"],
        )
        assert rewards[0] == 0.0

    def test_generation_task_zero(self):
        rewards = accuracy_reward(
            [_wrap("whatever")],
            ["reference"],
            ["generation"],
        )
        assert rewards[0] == 0.0

    def test_batch_processing(self):
        completions = [_wrap("Oslo"), _wrap("Bergen"), _wrap("Et dikt")]
        references = ["Oslo", "Oslo", "ref"]
        task_types = ["qa", "qa", "creative"]
        rewards = accuracy_reward(completions, references, task_types)
        assert len(rewards) == 3
        assert rewards[0] == 1.0
        assert rewards[1] == 0.0
        assert rewards[2] == 0.0

    def test_extraction_exact(self):
        rewards = accuracy_reward(
            [_wrap("Norge, Sverige, Danmark")],
            ["Norge, Sverige, Danmark"],
            ["extraction"],
        )
        assert rewards[0] == 1.0
