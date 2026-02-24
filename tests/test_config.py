"""Tests for norai_tools.config."""

from norai_tools.config import (
    CHECKPOINT_FILE,
    COLUMN_PAIRS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    GENERATION_CONFIG,
    OUTPUT_FILE,
)


def test_default_model_is_borealis():
    assert DEFAULT_MODEL == "NbAiLab/borealis-4b-instruct-preview"


def test_default_batch_size_positive():
    assert DEFAULT_BATCH_SIZE > 0


def test_default_temperature_range():
    assert 0.0 < DEFAULT_TEMPERATURE <= 1.0


def test_default_max_new_tokens_positive():
    assert DEFAULT_MAX_NEW_TOKENS > 0


def test_default_checkpoint_every_positive():
    assert DEFAULT_CHECKPOINT_EVERY > 0


def test_default_device():
    assert DEFAULT_DEVICE == "auto"


def test_default_dtype():
    assert DEFAULT_DTYPE == "bfloat16"


def test_checkpoint_file_is_jsonl():
    assert CHECKPOINT_FILE.endswith(".jsonl")


def test_output_file_is_parquet():
    assert OUTPUT_FILE.endswith(".parquet")


def test_column_pairs_structure():
    assert len(COLUMN_PAIRS) == 3
    for no_col, en_col in COLUMN_PAIRS:
        assert isinstance(no_col, str)
        assert isinstance(en_col, str)
        assert en_col.endswith("_en")


def test_generation_config_keys():
    assert "max_new_tokens" in GENERATION_CONFIG
    assert "do_sample" in GENERATION_CONFIG
    assert "temperature" in GENERATION_CONFIG
    assert "top_p" in GENERATION_CONFIG
    assert "repetition_penalty" in GENERATION_CONFIG


def test_generation_config_values():
    assert GENERATION_CONFIG["do_sample"] is True
    assert GENERATION_CONFIG["temperature"] == DEFAULT_TEMPERATURE
    assert GENERATION_CONFIG["max_new_tokens"] == DEFAULT_MAX_NEW_TOKENS
