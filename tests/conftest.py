"""Shared test fixtures for NORAI-Tools tests."""

import pytest


@pytest.fixture
def sample_row():
    """A typical Norwegian Alpaca dataset row."""
    return {
        "instruction": "Forklar hva fotosyntese er.",
        "input": "",
        "output": "Fotosyntese er prosessen der planter gjøre mat fra sollys.",
        "instruction_en": "Explain what photosynthesis is.",
        "input_en": "",
        "output_en": "Photosynthesis is the process by which plants make food from sunlight.",
    }


@pytest.fixture
def sample_row_with_input():
    """A row where the input field is non-empty."""
    return {
        "instruction": "Oversett følgende tekst til norsk.",
        "input": "The weather is nice today.",
        "output": "Været er fint i dag.",
        "instruction_en": "Translate the following text to Norwegian.",
        "input_en": "The weather is nice today.",
        "output_en": "The weather is nice today.",
    }


@pytest.fixture
def sample_row_improved():
    """A row after Phase 1 improvement (with *_improved columns)."""
    return {
        "instruction": "Forklar hva fotosyntese er.",
        "input": "",
        "output": "Fotosyntese er prosessen der planter gjøre mat fra sollys.",
        "instruction_en": "Explain what photosynthesis is.",
        "input_en": "",
        "output_en": "Photosynthesis is the process by which plants make food from sunlight.",
        "instruction_improved": "Forklar hva fotosyntese er.",
        "input_improved": "",
        "output_improved": "Fotosyntese er prosessen der planter lager mat fra sollys.",
    }


@pytest.fixture
def sample_rows_batch():
    """A small batch of rows for testing batch operations."""
    return [
        {
            "instruction": "Skriv et dikt om våren.",
            "input": "",
            "output": "Våren kommer med blomster og sol.",
            "instruction_en": "Write a poem about spring.",
            "input_en": "",
            "output_en": "Spring comes with flowers and sun.",
        },
        {
            "instruction": "Hva er hovedstaden i Norge?",
            "input": "",
            "output": "Oslo er hovedstaden i Norge.",
            "instruction_en": "What is the capital of Norway?",
            "input_en": "",
            "output_en": "Oslo is the capital of Norway.",
        },
        {
            "instruction": "Oversett til norsk.",
            "input": "Hello",
            "output": "Hei",
            "instruction_en": "Translate to Norwegian.",
            "input_en": "Hello",
            "output_en": "Hello",
        },
    ]


@pytest.fixture
def empty_row():
    """A row with empty Norwegian fields."""
    return {
        "instruction": "",
        "input": "",
        "output": "",
        "instruction_en": "Some instruction",
        "input_en": "",
        "output_en": "Some output",
    }
