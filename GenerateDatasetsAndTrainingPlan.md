# Norwegian AI Dataset Generation & Training Plan

This document describes how to run the full NORAI-Tools pipeline on [Modal](https://modal.com) cloud GPUs using the `modal_app.py` CLI runner.

## Overview

The pipeline has two parallel paths, both starting from the NbAiLab/norwegian-alpaca dataset (51,942 rows):

```
Path A (Standard):
  Phase 1:   Borealis 4B improves Norwegian text quality
  Phase 1.5: Add prompt + task_type columns for training
  Phase 2:   GSPO alignment training on Qwen3.5-35B-A3B

Path B (Distilled):
  Phase 1:   Same as above
  Phase 1b:  Qwen generates new outputs + Borealis polishes them
  Phase 2b:  GSPO alignment training on the distilled dataset
```

Path B produces higher-quality reference outputs (Qwen's knowledge + Borealis's Norwegian fluency) and creates a self-distillation training signal.

## Prerequisites

### Local Setup

```bash
# Clone and install
git clone https://github.com/your-username/NORAI-Tools.git
cd NORAI-Tools
pip install -e ".[modal]"

# Set up Modal
pip install modal
modal token new
```

### Modal Secrets (optional, for HuggingFace Hub push)

```bash
modal secret create huggingface HF_TOKEN=hf_your_token_here
```

Volumes are created automatically on first run.

## Pipeline Steps

### Phase 1: Improve Norwegian Alpaca with Borealis

Uses Borealis 4B (NbAiLab/borealis-4b-instruct-preview) to rewrite the Norwegian columns into natural, fluent Bokmål.

```bash
modal run modal_app.py --step improve
```

| | |
|---|---|
| **GPU** | L4 (24 GB) |
| **Time** | ~10-12 hours |
| **Cost** | ~$7 |
| **Input** | HuggingFace Hub (NbAiLab/norwegian-alpaca) |
| **Output** | `/data/norwegian_alpaca_improved.parquet` |

Test on a small subset first:

```bash
modal run modal_app.py --step improve --subset 100
```

### Phase 1.5: Prepare GSPO Dataset

Adds `prompt` (chat format) and `task_type` (classification heuristic) columns. CPU only.

```bash
modal run modal_app.py --step prepare-gspo
```

| | |
|---|---|
| **GPU** | None (CPU) |
| **Time** | ~1 minute |
| **Cost** | ~$0.01 |
| **Input** | `/data/norwegian_alpaca_improved.parquet` |
| **Output** | `/data/norwegian_alpaca_gspo.parquet` |

### Phase 1b: Qwen Distillation + Norwegian Polish

Two-pass pipeline — run separately to use the right GPU for each:

**Pass 1: Qwen generates Norwegian outputs**

```bash
modal run modal_app.py --step distill-generate
```

| | |
|---|---|
| **GPU** | A100-80GB |
| **Time** | ~15 hours |
| **Cost** | ~$56 |
| **Input** | `/data/norwegian_alpaca_improved.parquet` |
| **Output** | `/data/norwegian_alpaca_qwen_raw.parquet` |

**Pass 2: Borealis polishes Qwen's Norwegian**

```bash
modal run modal_app.py --step distill-polish
```

| | |
|---|---|
| **GPU** | L4 (24 GB) |
| **Time** | ~10 hours |
| **Cost** | ~$6 |
| **Input** | `/data/norwegian_alpaca_qwen_raw.parquet` |
| **Output** | `/data/norwegian_alpaca_qwen_polished.parquet` |

### Phase 2: GSPO Alignment Training

GSPO (Group Sequence Policy Optimization) on Qwen3.5-35B-A3B with LoRA, using 4 reward functions (semantic similarity, language quality, length, accuracy).

```bash
modal run modal_app.py --step gspo-train
```

| | |
|---|---|
| **GPU** | 2x A100-80GB |
| **Time** | 24-72 hours |
| **Cost** | ~$180-540 |
| **Input** | `/data/norwegian_alpaca_gspo.parquet` |
| **Output** | `/models/gspo_qwen_norwegian/` (LoRA adapter) |

### Phase 2b: GSPO Training on Distilled Dataset

Same as Phase 2, but using Qwen-distilled reference outputs for self-distillation.

```bash
modal run modal_app.py --step gspo-train-distilled
```

| | |
|---|---|
| **GPU** | 2x A100-80GB |
| **Time** | 24-72 hours |
| **Cost** | ~$180-540 |
| **Input** | `/data/norwegian_alpaca_qwen_polished.parquet` |
| **Output** | `/models/gspo_qwen_norwegian_distilled/` (LoRA adapter) |

## Run Full Pipelines

```bash
# Path A: Standard (Phases 1 → 1.5 → 2)
modal run modal_app.py --step all

# Path B: Distilled (Phases 1 → 1b → 2b)
modal run modal_app.py --step all-distilled
```

## CLI Options

```
modal run modal_app.py \
  --step <step>              # Required: which step to run
  --subset <N>               # Process only first N rows (for testing)
  --batch-size <N>           # Override default batch size
  --push-to-hub              # Push dataset to HuggingFace Hub
  --hub-repo <repo>          # HuggingFace repo ID
  --num-generations <N>      # GSPO group size G (default: 8)
  --learning-rate <float>    # GSPO learning rate (default: 5e-7)
  --num-epochs <N>           # GSPO epochs (default: 1)
  --save-steps <N>           # Checkpoint frequency (default: 200)
```

## Data Flow

```
NbAiLab/norwegian-alpaca (HuggingFace Hub)
    │
    ▼
[Phase 1: improve] ─────────────────────────────────────────────
    │                                                            │
    ▼                                                            ▼
norwegian_alpaca_improved.parquet                   norwegian_alpaca_improved.parquet
    │                                                            │
    ▼                                                            ▼
[Phase 1.5: prepare-gspo]                         [Phase 1b: distill-generate]
    │                                                            │
    ▼                                                            ▼
norwegian_alpaca_gspo.parquet                     norwegian_alpaca_qwen_raw.parquet
    │                                                            │
    ▼                                                            ▼
[Phase 2: gspo-train]                             [Phase 1b: distill-polish]
    │                                                            │
    ▼                                                            ▼
/models/gspo_qwen_norwegian/                      norwegian_alpaca_qwen_polished.parquet
                                                                 │
                                                                 ▼
                                                  [Phase 2b: gspo-train-distilled]
                                                                 │
                                                                 ▼
                                                  /models/gspo_qwen_norwegian_distilled/
```

## Volume Management

### Inspect volume contents

```bash
modal volume ls norai-data
modal volume ls norai-models
```

### Download files locally

```bash
modal volume get norai-data norwegian_alpaca_improved.parquet ./local_copy.parquet
modal volume get norai-models gspo_qwen_norwegian ./local_adapter/
```

### Clean up checkpoints after a successful run

```bash
modal volume rm norai-data alpaca_improved_checkpoint.jsonl
modal volume rm norai-data qwen_generation_checkpoint.jsonl
modal volume rm norai-data borealis_polish_checkpoint.jsonl
```

## Crash Recovery

All pipeline steps save JSONL checkpoints to the `/data` volume. If a Modal function times out or crashes, re-running the same step automatically resumes from where it left off.

For GSPO training, TRL's `GRPOTrainer` saves model checkpoints to the `/models` volume at regular intervals (`--save-steps`). Training can be resumed by re-running the step.

## Cost Summary

| Step | GPU | Hours | Est. Cost |
|------|-----|-------|-----------|
| Phase 1 (improve) | L4 | ~12h | ~$7 |
| Phase 1.5 (prepare-gspo) | CPU | <1min | ~$0.01 |
| Phase 1b Pass 1 (distill-generate) | A100-80GB | ~15h | ~$56 |
| Phase 1b Pass 2 (distill-polish) | L4 | ~10h | ~$6 |
| Phase 2 (gspo-train) | 2x A100-80GB | 24-72h | ~$180-540 |
| Phase 2b (gspo-train-distilled) | 2x A100-80GB | 24-72h | ~$180-540 |
| **Path A total** | | | **~$190-550** |
| **Path B total** | | | **~$250-610** |

Costs are estimates based on Modal's on-demand pricing. Actual costs depend on batch size, dataset size, and training convergence.

## Alternative: Interactive Notebooks

For development and exploration, you can use the Jupyter notebooks directly:

- **Local/Colab:** Run the notebooks in `notebooks/` with a local GPU or Google Colab
- **Modal Notebooks:** `modal notebook --gpu L4` spins up a Jupyter instance with GPU access

The notebooks contain the same logic as the CLI steps but with interactive inspection cells (side-by-side comparisons, charts, sample outputs).
