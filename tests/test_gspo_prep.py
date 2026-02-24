"""Tests for norai_tools.gspo_prep."""

import pytest
from datasets import Dataset

from norai_tools.gspo_prep import (
    build_prompt,
    classify_task_type,
    prepare_gspo_dataset,
    validate_gspo_dataset,
    TASK_KEYWORDS,
)


class TestClassifyTaskType:
    def test_classification(self):
        assert classify_task_type("Classify the following animal") == "classification"

    def test_identify(self):
        assert classify_task_type("Identify the main character") == "classification"

    def test_extraction(self):
        assert classify_task_type("Extract the key points") == "extraction"

    def test_list(self):
        assert classify_task_type("List three countries in Africa") == "extraction"

    def test_generation(self):
        assert classify_task_type("Write a haiku about spring") == "generation"

    def test_create(self):
        assert classify_task_type("Create a recipe for pancakes") == "generation"

    def test_rewriting(self):
        assert classify_task_type("Summarize the following text") == "rewriting"

    def test_paraphrase(self):
        assert classify_task_type("Paraphrase this sentence") == "rewriting"

    def test_qa_what(self):
        assert classify_task_type("What is photosynthesis?") == "qa"

    def test_qa_explain(self):
        assert classify_task_type("Explain how gravity works") == "qa"

    def test_creative(self):
        assert classify_task_type("Imagine a short story about a cat") == "creative"

    def test_poem(self):
        assert classify_task_type("Compose a poem about the ocean") == "generation"

    def test_other_fallback(self):
        assert classify_task_type("Do the thing") == "other"

    def test_empty_string(self):
        assert classify_task_type("") == "other"

    def test_none_input(self):
        assert classify_task_type(None) == "other"

    def test_case_insensitive(self):
        assert classify_task_type("CLASSIFY this item") == "classification"
        assert classify_task_type("Write A Letter") == "generation"

    def test_priority_first_match(self):
        # "Write" matches generation before "story" matches creative
        result = classify_task_type("Write a story")
        assert result == "generation"


class TestBuildPrompt:
    def test_basic_prompt(self, sample_row_improved):
        prompt = build_prompt(sample_row_improved)
        assert isinstance(prompt, list)
        assert len(prompt) == 1
        assert prompt[0]["role"] == "user"
        assert "Forklar hva fotosyntese er." in prompt[0]["content"]

    def test_prompt_with_input(self):
        row = {
            "instruction_improved": "Oversett dette.",
            "input_improved": "The cat sat on the mat.",
        }
        prompt = build_prompt(row)
        assert "Oversett dette." in prompt[0]["content"]
        assert "The cat sat on the mat." in prompt[0]["content"]

    def test_prompt_empty_input(self):
        row = {
            "instruction_improved": "Skriv et dikt.",
            "input_improved": "",
        }
        prompt = build_prompt(row)
        content = prompt[0]["content"]
        assert content.strip() == "Skriv et dikt."

    def test_prompt_none_input(self):
        row = {
            "instruction_improved": "Skriv et dikt.",
            "input_improved": None,
        }
        prompt = build_prompt(row)
        assert len(prompt) == 1


class TestPrepareGspoDataset:
    def test_adds_prompt_column(self, sample_row_improved):
        ds = Dataset.from_list([sample_row_improved])
        result = prepare_gspo_dataset(ds)
        assert "prompt" in result.column_names

    def test_adds_task_type_column(self, sample_row_improved):
        ds = Dataset.from_list([sample_row_improved])
        result = prepare_gspo_dataset(ds)
        assert "task_type" in result.column_names

    def test_preserves_original_columns(self, sample_row_improved):
        ds = Dataset.from_list([sample_row_improved])
        result = prepare_gspo_dataset(ds)
        assert "instruction" in result.column_names
        assert "output_improved" in result.column_names
        assert "instruction_en" in result.column_names

    def test_prompt_is_list_of_dicts(self, sample_row_improved):
        ds = Dataset.from_list([sample_row_improved])
        result = prepare_gspo_dataset(ds)
        prompt = result[0]["prompt"]
        assert isinstance(prompt, list)
        assert prompt[0]["role"] == "user"

    def test_task_type_classified(self, sample_row_improved):
        ds = Dataset.from_list([sample_row_improved])
        result = prepare_gspo_dataset(ds)
        # "Explain what photosynthesis is." -> "qa"
        assert result[0]["task_type"] == "qa"


class TestValidateGspoDataset:
    def test_valid_dataset(self, sample_row_improved):
        ds = Dataset.from_list([sample_row_improved])
        ds = prepare_gspo_dataset(ds)
        result = validate_gspo_dataset(ds)
        assert result["is_valid"] is True
        assert result["total_rows"] == 1
        assert result["empty_prompts"] == 0
        assert result["missing_columns"] == []

    def test_missing_columns(self):
        ds = Dataset.from_list([{"some_col": "value"}])
        result = validate_gspo_dataset(ds)
        assert result["is_valid"] is False
        assert len(result["missing_columns"]) > 0

    def test_task_type_distribution(self, sample_row_improved):
        ds = Dataset.from_list([sample_row_improved])
        ds = prepare_gspo_dataset(ds)
        result = validate_gspo_dataset(ds)
        assert "qa" in result["task_type_distribution"]

    def test_empty_prompt_detected(self):
        row = {
            "instruction_improved": "",
            "input_improved": "",
            "output_improved": "some output",
            "instruction_en": "some instruction",
            "prompt": [{"role": "user", "content": ""}],
            "task_type": "other",
        }
        ds = Dataset.from_list([row])
        result = validate_gspo_dataset(ds)
        assert result["empty_prompts"] == 1
        assert result["is_valid"] is False


class TestTaskKeywords:
    def test_all_task_types_have_keywords(self):
        expected_types = {"classification", "extraction", "generation", "rewriting", "qa", "creative"}
        assert set(TASK_KEYWORDS.keys()) == expected_types

    def test_keywords_are_lowercase(self):
        for task_type, keywords in TASK_KEYWORDS.items():
            for kw in keywords:
                assert kw == kw.lower(), f"Keyword '{kw}' in {task_type} is not lowercase"
