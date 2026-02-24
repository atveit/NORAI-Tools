"""Tests for norai_tools.prompts."""

from norai_tools.prompts import build_improvement_prompt, should_skip, IMPROVEMENT_PROMPT_TEMPLATE


class TestBuildImprovementPrompt:
    def test_contains_norwegian_text(self):
        result = build_improvement_prompt("Hei verden", "Hello world")
        assert "Hei verden" in result

    def test_contains_english_reference(self):
        result = build_improvement_prompt("Hei verden", "Hello world")
        assert "Hello world" in result

    def test_contains_instruction_header(self):
        result = build_improvement_prompt("test", "test")
        assert "Forbedre denne norske teksten" in result

    def test_contains_few_shot_examples(self):
        result = build_improvement_prompt("test", "test")
        assert "Hun bestemte seg for å dra" in result
        assert "Han ante ikke" in result

    def test_contains_no_extra_output_instruction(self):
        result = build_improvement_prompt("test", "test")
        assert "Skriv KUN den forbedrede teksten" in result

    def test_template_has_placeholders(self):
        assert "{norwegian_text}" in IMPROVEMENT_PROMPT_TEMPLATE
        assert "{english_text}" in IMPROVEMENT_PROMPT_TEMPLATE


class TestShouldSkip:
    def test_skip_empty_string(self):
        assert should_skip("", "some text") is True

    def test_skip_none(self):
        assert should_skip(None, "some text") is True

    def test_skip_whitespace_only(self):
        assert should_skip("   ", "some text") is True

    def test_skip_identical_to_english(self):
        assert should_skip("Telegram", "Telegram") is True

    def test_skip_identical_with_whitespace(self):
        assert should_skip("  Oslo  ", "Oslo") is True

    def test_skip_very_short(self):
        assert should_skip("Hei", "Hi") is True

    def test_skip_single_char(self):
        assert should_skip("A", "A") is True

    def test_no_skip_normal_text(self):
        assert should_skip("Dette er en normal setning.", "This is a normal sentence.") is False

    def test_no_skip_five_chars(self):
        assert should_skip("Hallo", "Hello") is False

    def test_no_skip_different_from_english(self):
        assert should_skip("Norsk tekst her", "English text here") is False
