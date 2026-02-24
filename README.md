# NORAI-Tools

Norwegian AI dataset tools for improving and preparing Norwegian language datasets.

## Pipeline Architecture

NORAI-Tools implements a two-phase pipeline with two paths: a standard path using Borealis 4B, and a distilled path that adds Qwen distillation before Borealis polishing.

### Standard Path

```
NbAiLab/norwegian-alpaca (51,942 rows)
        |
        v
  [Phase 1] Borealis 4B improvement
        |   (improve_alpaca.ipynb)
        v
  [Phase 1.5] GSPO dataset preparation
        |     (prepare_gspo_dataset.ipynb)
        v
  [Phase 2] GSPO alignment training on Qwen3.5-35B-A3B
            (gspo_train.ipynb)
```

### Distilled Path

```
NbAiLab/norwegian-alpaca
        |
        v
  [Phase 1b] Qwen distillation + Borealis polish
        |    (distill_qwen_norwegian.ipynb)
        v
  [Phase 2b] GSPO alignment training on distilled dataset
             (gspo_train_distilled.ipynb)
```

## Project Structure

```
norai_tools/                         # Python library
    __init__.py                      # All exports
    config.py                        # Constants (model name, batch size, etc.)
    improver.py                      # AlpacaImprover: batch improvement engine
    loader.py                        # Dataset I/O + checkpointing
    prompts.py                       # Norwegian improvement prompt template
    validation.py                    # Response validation (preamble strip, hallucination check)
    gspo_prep.py                     # GSPO dataset preparation (prompt format, task classification)
    rewards.py                       # 4 reward functions for GSPO training

notebooks/                           # Jupyter notebooks (thin wrappers around library)
    improve_alpaca.ipynb             # Phase 1: Borealis improvement
    prepare_gspo_dataset.ipynb       # Phase 1.5: GSPO dataset prep
    gspo_train.ipynb                 # Phase 2: GSPO alignment training
    distill_qwen_norwegian.ipynb     # Phase 1b: Qwen distillation + Borealis polish
    gspo_train_distilled.ipynb       # Phase 2b: GSPO on distilled dataset

modal_app.py                         # Modal.com CLI runner for cloud GPU execution
GenerateDatasetsAndTrainingPlan.md   # Full pipeline runbook for Modal
tests/                               # 110 tests
pyproject.toml                       # Package config
```

## Installation

```bash
# Basic (improvement pipeline)
pip install -e .

# With GSPO training dependencies (TRL, vLLM, sentence-transformers, etc.)
pip install -e ".[gspo]"

# With Modal cloud runner
pip install -e ".[modal]"

# With dev/test dependencies
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Quick Start

### Notebooks

Open any notebook under `notebooks/` and run cells in order. Each notebook is a thin wrapper around the `norai_tools` library.

### Modal CLI

Run pipeline steps on cloud GPUs via [Modal](https://modal.com):

```bash
modal run modal_app.py --step improve
```

See [GenerateDatasetsAndTrainingPlan.md](GenerateDatasetsAndTrainingPlan.md) for the full Modal runbook.

### Tests

```bash
pytest tests/ -v
```

## Library API

| Export | Module | Description |
|---|---|---|
| `AlpacaImprover` | `improver` | Batch improvement engine using Borealis 4B |
| `load_alpaca` | `loader` | Load the norwegian-alpaca dataset from HuggingFace |
| `save_improved` | `loader` | Save improved dataset to Parquet with checkpointing |
| `prepare_gspo_dataset` | `gspo_prep` | Convert improved dataset to GSPO prompt format |
| `validate_gspo_dataset` | `gspo_prep` | Validate a prepared GSPO dataset |
| `classify_task_type` | `gspo_prep` | Classify instruction into task type |
| `semantic_reward` | `rewards` | Cosine similarity to reference (multilingual-e5-large) |
| `language_reward` | `rewards` | Norwegian language quality score |
| `length_reward` | `rewards` | Length appropriateness relative to reference |
| `accuracy_reward` | `rewards` | Factual accuracy score |
| `build_improvement_prompt` | `prompts` | Build the Borealis improvement prompt |
| `validate_response` | `validation` | Strip preamble and check for hallucination |

## Models Used

| Model | Role |
|---|---|
| [NbAiLab/borealis-4b-instruct-preview](https://huggingface.co/NbAiLab/borealis-4b-instruct-preview) | Phase 1 -- Norwegian text improvement (Gemma 3 4B fine-tune) |
| [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | Phase 2 -- GSPO alignment target (MoE, instruct) |

## Dataset

[NbAiLab/norwegian-alpaca](https://huggingface.co/datasets/NbAiLab/norwegian-alpaca) -- 51,942 rows with 3 Norwegian columns (`instruction`, `input`, `output`) and 3 English columns (`instruction_en`, `input_en`, `output_en`).

## License

MIT
