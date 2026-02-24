"""Integration tests for norai_tools.improver with mocked model/tokenizer."""

from unittest.mock import MagicMock, patch
import torch
import pytest

from norai_tools.improver import AlpacaImprover


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer that behaves like a real one."""
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "</s>"
    tokenizer.padding_side = "left"

    def apply_chat_template(messages, tokenize=False, add_generation_prompt=True):
        return f"<user>{messages[0]['content']}</user><model>"

    tokenizer.apply_chat_template = apply_chat_template

    # Make tokenizer callable — returns dict with input_ids and attention_mask
    def tokenize_call(*args, **kwargs):
        batch_size = 1
        if isinstance(args[0], list):
            batch_size = len(args[0])
        result = MagicMock()
        result.__getitem__ = lambda self, key: {
            "input_ids": torch.ones(batch_size, 10, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 10, dtype=torch.long),
        }[key]
        result.keys = lambda: ["input_ids", "attention_mask"]
        result.to = lambda device: result
        return result

    tokenizer.side_effect = tokenize_call
    tokenizer.decode = MagicMock(return_value="Han bestemte seg for å dra fra det stedet.")
    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock model that returns tensor outputs."""
    model = MagicMock()
    model.device = "cpu"
    model.eval = MagicMock()

    def generate(**kwargs):
        batch_size = kwargs.get("input_ids", torch.ones(1, 10)).shape[0]
        return torch.ones(batch_size, 20, dtype=torch.long)

    model.generate = MagicMock(side_effect=generate)
    return model


class TestAlpacaImproverInit:
    def test_default_params(self):
        improver = AlpacaImprover()
        assert improver.model_name == "NbAiLab/borealis-4b-instruct-preview"
        assert improver.batch_size == 8
        assert improver.model is None
        assert improver.tokenizer is None

    def test_custom_params(self):
        improver = AlpacaImprover(
            model_name="test/model",
            device="cpu",
            batch_size=4,
            max_new_tokens=128,
            dtype="float32",
        )
        assert improver.model_name == "test/model"
        assert improver.device == "cpu"
        assert improver.batch_size == 4
        assert improver.max_new_tokens == 128
        assert improver.dtype == torch.float32


class TestImproveText:
    def test_skip_empty_text(self, mock_model, mock_tokenizer):
        improver = AlpacaImprover()
        improver.model = mock_model
        improver.tokenizer = mock_tokenizer
        result = improver.improve_text("", "Some english")
        assert result == ""
        mock_model.generate.assert_not_called()

    def test_skip_identical_to_english(self, mock_model, mock_tokenizer):
        improver = AlpacaImprover()
        improver.model = mock_model
        improver.tokenizer = mock_tokenizer
        result = improver.improve_text("Oslo", "Oslo")
        assert result == "Oslo"
        mock_model.generate.assert_not_called()

    def test_skip_short_text(self, mock_model, mock_tokenizer):
        improver = AlpacaImprover()
        improver.model = mock_model
        improver.tokenizer = mock_tokenizer
        result = improver.improve_text("Hei", "Hi")
        assert result == "Hei"
        mock_model.generate.assert_not_called()


class TestImproveRow:
    def test_adds_improved_columns(self, mock_model, mock_tokenizer, sample_row):
        improver = AlpacaImprover()
        improver.model = mock_model
        improver.tokenizer = mock_tokenizer
        result = improver.improve_row(sample_row)
        assert "instruction_improved" in result
        assert "input_improved" in result
        assert "output_improved" in result

    def test_preserves_original_columns(self, mock_model, mock_tokenizer, sample_row):
        improver = AlpacaImprover()
        improver.model = mock_model
        improver.tokenizer = mock_tokenizer
        result = improver.improve_row(sample_row)
        assert result["instruction"] == sample_row["instruction"]
        assert result["output"] == sample_row["output"]
        assert result["instruction_en"] == sample_row["instruction_en"]

    def test_empty_input_skipped(self, mock_model, mock_tokenizer, sample_row):
        improver = AlpacaImprover()
        improver.model = mock_model
        improver.tokenizer = mock_tokenizer
        result = improver.improve_row(sample_row)
        # Empty input should be returned as-is
        assert result["input_improved"] == ""
