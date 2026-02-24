# Plan: Improve Norwegian Alpaca Dataset Quality with Borealis

## Context

**Dataset:** [NbAiLab/norwegian-alpaca](https://huggingface.co/datasets/NbAiLab/norwegian-alpaca)
- 51,942 rows, single `train` split
- Bilingual: Norwegian Bokmål columns (`instruction`, `input`, `output`) + English columns (`instruction_en`, `input_en`, `output_en`)
- Originally machine-translated from Stanford Alpaca via gpt-3.5-turbo — likely contains translation artifacts, unnatural phrasing, and inconsistent quality

**Model:** [NbAiLab/borealis-4b-instruct-preview](https://huggingface.co/NbAiLab/borealis-4b-instruct-preview)
- 4B parameter Gemma 3 fine-tune, Norwegian-focused instruction model
- BF16 safetensors format (also available as GGUF and MLX)
- Chat-template compatible via `transformers`
- Preview/experimental — not safety-aligned

**Goal:** Use Borealis to rewrite/improve the three Norwegian text columns (`instruction`, `input`, `output`) so they read as natural, fluent Norwegian Bokmål rather than machine-translated text. The English columns serve as reference anchors.

---

## Architecture Overview

```
norai_tools/                  # Python library (pip-installable)
├── __init__.py
├── improver.py               # Core improvement logic
├── prompts.py                # Prompt templates (few-shot, Norwegian)
├── validation.py             # Response validation (preamble stripping, hallucination check)
├── loader.py                 # Dataset loading/saving utilities
└── config.py                 # Default configuration constants

notebooks/
└── improve_alpaca.ipynb      # Standalone Jupyter notebook

pyproject.toml                # Package definition
```

---

## Step 1: Project Setup (`pyproject.toml`)

Create a minimal `pyproject.toml` with dependencies:

- `transformers` (model loading + generation)
- `torch` (inference backend)
- `datasets` (HuggingFace dataset loading)
- `tqdm` (progress bars)
- `pandas` (optional, for notebook display)

Optional extras group `[notebook]` adding `jupyter`, `ipywidgets`.

---

## Step 2: Prompt Design (`norai_tools/prompts.py`)

### Research-Based Approach

The prompt design is informed by the [lore-lantern](https://github.com/Aanerud/lore-lantern) project — an open-source multi-agent storytelling system that uses Borealis specifically for Norwegian text refinement. Their `src/prompts/refine/no.py` module provides a battle-tested prompt pattern for making machine-generated Norwegian sound natural.

Key findings from lore-lantern and Gemma 3 prompt engineering best practices:

1. **Few-shot examples are critical for small models.** Borealis (4B) performs much better when shown concrete before/after examples of the specific transformations expected. Zero-shot "improve this text" prompts produce inconsistent results.

2. **Prompt entirely in Norwegian.** Borealis was fine-tuned on Norwegian instructions. Prompting in Norwegian (not English) produces more natural output.

3. **Target specific translation artifacts.** Machine-translated Norwegian from gpt-3.5-turbo has predictable failure patterns:
   - Unnatural collocations: "gjøre en beslutning" → "ta en beslutning" / "bestemme seg"
   - English word order: "Det var veldig interessant for ham" → "Han syntes det var spennende"
   - Overly formal vocabulary: "imidlertid" → "men", "dessuten" → "og"
   - English loanwords: "basically" → "egentlig"
   - Literal calques: "hadde ikke noen idé" → "ante ikke"

4. **Explicit "write ONLY the improved text" instruction.** Small models tend to add preambles like "Her er den forbedrede teksten:" — the prompt must explicitly prevent this.

5. **Temperature 0.3** for conservative, faithful rewrites (lore-lantern uses 0.3 for refinement; Gemma 3 defaults recommend 1.0 for creative tasks, but we want faithfulness).

6. **No system role.** Borealis inherits Gemma 3's chat template which supports only `user` and `model` roles. System-level instructions go directly into the user message.

### Prompt Template

A single unified prompt template handles all three column types. Using the Gemma 3 chat format (`tokenizer.apply_chat_template`), the user message is:

```
Forbedre denne norske teksten. Se etter:

1. Unaturlige ordvalg (f.eks. "gjøre en beslutning" → "ta en beslutning")
2. Stiv ordstilling fra engelsk (f.eks. "Det var veldig interessant for ham å se" → "Han syntes det var spennende å se")
3. Formelle ord som kan være mer muntlige (f.eks. "imidlertid" → "men", "dessuten" → "og")
4. Engelske lånord med norske alternativer (f.eks. "basically" → "egentlig")
5. Direkte oversettelser (f.eks. "hadde ikke noen idé" → "ante ikke")

Eksempler:
ORIGINAL: Hun gjorde en beslutning om å forlate stedet.
BEDRE: Hun bestemte seg for å dra.

ORIGINAL: Det var veldig interessant for ham å se dette.
BEDRE: Han syntes det var spennende å se.

ORIGINAL: Han hadde ikke noen idé om hva som skjedde.
BEDRE: Han ante ikke hva som foregikk.

Engelsk referanse (for mening, IKKE for språk): {english_text}

Skriv KUN den forbedrede teksten, ingenting annet:

{norwegian_text}
```

The English reference text is provided so Borealis can verify meaning preservation, but the prompt explicitly says it is for meaning only, not language — preventing the model from code-switching to English.

### Response Validation (`norai_tools/validation.py`)

Borrowed from lore-lantern's `validate_response()` pattern, adapted for our batch processing:

```python
PREAMBLE_INDICATORS = [
    "her er", "forbedret tekst:", "endringer:", "jeg har",
    "teksten er", "original:", "###", "bedre:", "resultat:",
]

def validate_response(response: str, original: str) -> str:
    """Validate and clean LLM response. Return original if invalid."""
    refined = response.strip()

    # Too short — probably failed
    if len(refined) < 10:
        return original

    # Strip preambles that small models add
    refined_lower = refined.lower()
    for indicator in PREAMBLE_INDICATORS:
        if refined_lower.startswith(indicator):
            # Try to extract just the text after the preamble
            lines = refined.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 30 and not any(
                    ind in line.lower() for ind in PREAMBLE_INDICATORS
                ):
                    return line
            return original  # Couldn't extract clean text

    # Hallucination check: <30% word overlap = model invented new content
    original_words = set(original.lower().split())
    refined_words = set(refined.lower().split())
    overlap = len(original_words & refined_words) / max(len(original_words), 1)
    if overlap < 0.3:
        return original

    # Length sanity: >3x original length = runaway generation
    if len(refined) > 3 * len(original) + 50:
        return original

    return refined
```

### Skip Logic

- **Empty fields:** If a Norwegian field is empty (the `input` column is often empty), skip it — don't send it to the model.
- **Identical to English:** If the Norwegian and English texts are identical (e.g., both say "Telegram", a number, or code), skip — copy as-is.
- **Very short text (<5 chars):** Single words/numbers — copy as-is.

---

## Step 3: Core Improvement Engine (`norai_tools/improver.py`)

### Class: `AlpacaImprover`

```python
class AlpacaImprover:
    def __init__(self, model_name, device, batch_size, max_new_tokens, dtype)
    def load_model(self)
    def improve_text(self, prompt: str) -> str
    def improve_row(self, row: dict) -> dict
    def improve_dataset(self, dataset, output_path, checkpoint_every) -> Dataset
```

### Efficiency strategy

Processing 51,942 rows × up to 3 columns = ~150k generations. This is the dominant cost. Key efficiency measures:

1. **Batched generation:** Group prompts into batches (default 8–16 depending on VRAM) and use `model.generate()` with left-padded batched inputs. This is the single biggest speedup — it saturates the GPU instead of running one prompt at a time.

2. **Skip empty fields:** ~40% of rows have empty `input` — skip these entirely.

3. **Short max tokens:** Most Norwegian fields are short (median ~50 tokens). Set `max_new_tokens=256` by default to avoid wasted allocation. For the `instruction` column specifically, use `max_new_tokens=128`.

4. **BF16 inference:** Load the model in `torch.bfloat16` (native format) to halve memory and speed up compute vs FP32.

5. **Checkpointing:** Save progress to a JSONL file every N rows (default 500). On restart, detect the checkpoint and resume from where processing stopped. This protects against crashes during the multi-hour run.

6. **KV cache:** Use `transformers` built-in KV cache (enabled by default in `generate()`).

7. **`torch.inference_mode()`:** Wrap all generation in `torch.inference_mode()` to disable gradient tracking.

### Generation parameters

```python
generation_config = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.3,       # Low temp for faithful rewriting
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}
```

Low temperature (0.3) keeps outputs close to the original meaning while allowing natural rephrasing. We're not looking for creative output — just fluent Norwegian.

---

## Step 4: Dataset Loading and Saving (`norai_tools/loader.py`)

```python
def load_alpaca() -> Dataset:
    """Load NbAiLab/norwegian-alpaca from HuggingFace."""

def save_improved(dataset, path: str, push_to_hub: bool = False):
    """Save improved dataset as Parquet + optionally push to HuggingFace Hub."""

def load_checkpoint(path: str) -> tuple[list[dict], int]:
    """Load existing checkpoint JSONL and return processed rows + resume index."""

def save_checkpoint(rows: list[dict], path: str):
    """Append rows to checkpoint JSONL."""
```

Output format: The improved dataset keeps all 6 original columns plus 3 new columns (`instruction_improved`, `input_improved`, `output_improved`). This preserves the originals for comparison and evaluation.

---

## Step 5: Configuration (`norai_tools/config.py`)

```python
DEFAULT_MODEL = "NbAiLab/borealis-4b-instruct-preview"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.3
DEFAULT_CHECKPOINT_EVERY = 500
DEFAULT_DEVICE = "auto"  # Let accelerate pick GPU/MPS/CPU
DEFAULT_DTYPE = "bfloat16"
CHECKPOINT_FILE = "alpaca_improved_checkpoint.jsonl"
OUTPUT_FILE = "norwegian_alpaca_improved.parquet"
```

---

## Step 6: Jupyter Notebook (`notebooks/improve_alpaca.ipynb`)

A standalone notebook that duplicates the library's logic inline so it works without installing the package. Structure:

1. **Cell 1 — Install deps:** `!pip install transformers torch datasets tqdm`
2. **Cell 2 — Configuration:** All tuneable parameters in one cell
3. **Cell 3 — Load dataset:** Load and display sample rows
4. **Cell 4 — Load model:** Load Borealis in BF16
5. **Cell 5 — Prompt templates:** Define the three prompt templates
6. **Cell 6 — Improvement functions:** `improve_batch()` and `improve_dataset()`
7. **Cell 7 — Run improvement:** Main processing loop with progress bar and checkpointing
8. **Cell 8 — Inspect results:** Side-by-side comparison of original vs improved text
9. **Cell 9 — Save output:** Save to parquet and optionally push to Hub

The notebook should be runnable top-to-bottom in a Colab/Jupyter environment with a GPU.

---

---

## Implementation Order

| # | Task | Files |
|---|------|-------|
| 1 | Create `pyproject.toml` with dependencies | `pyproject.toml` |
| 2 | Implement `config.py` with defaults | `norai_tools/config.py` |
| 3 | Implement `prompts.py` with templates | `norai_tools/prompts.py` |
| 4 | Implement `loader.py` (load, save, checkpoint) | `norai_tools/loader.py` |
| 5 | Implement `improver.py` (core engine) | `norai_tools/improver.py` |
| 6 | Wire up `__init__.py` exports | `norai_tools/__init__.py` |
| 7 | Create standalone Jupyter notebook | `notebooks/improve_alpaca.ipynb` |
| 8 | Test with a small subset (100 rows) | — |

---

## Hardware Expectations

- **GPU (recommended):** Borealis 4B in BF16 needs ~8 GB VRAM. Fits on a single RTX 3070/4070 or Colab T4 (16 GB). With batch_size=8, expect ~3–5 rows/second.
- **Apple Silicon (MPS):** Works with `device="mps"`. Use the MLX variant for better performance if desired (not covered in this library).
- **CPU:** Functional but impractically slow for 52k rows. Useful for testing on small subsets.

---

## Phase 2: GSPO Alignment on Qwen3.5-35B-A3B

### What is GSPO

**GSPO (Group Sequence Policy Optimization)** is the RL alignment algorithm developed by the Qwen team and used to train Qwen3. Published in [arXiv:2507.18071](https://arxiv.org/abs/2507.18071) (July 2025), it improves upon GRPO (DeepSeek's Group Relative Policy Optimization) in two critical ways:

1. **Sequence-level importance ratios** instead of token-level. GSPO defines the importance ratio using length-normalized sequence likelihood, then clips and optimizes at the sequence level. This eliminates the noise and instability of token-level optimization.
2. **Inherent MoE stability.** GRPO requires a "Routing Replay" strategy to stabilize MoE expert routing during RL training. GSPO completely eliminates this dependency — critical since Qwen3.5-35B-A3B is an MoE model.

In TRL, GSPO is implemented via `GRPOTrainer` with `importance_sampling_level="sequence"`. The dataset format is identical to GRPO — only the optimization algorithm changes.

Reference: [GSPO: Towards Scalable Reinforcement Learning for Language Models (Qwen blog)](https://qwenlm.github.io/blog/gspo/)

### Why GSPO on the Instruct Model

**Target model:** [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) (instruct variant, not base)

The instruct variant is preferred over [Qwen3.5-35B-A3B-Base](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base) for GSPO because:
- GSPO is a **policy optimization** method — it refines an already-capable policy, not a blank slate. The instruct model already follows instructions and generates structured responses, giving GSPO meaningful behavior to optimize.
- The base model would require SFT first to even produce coherent instruction-following outputs before GSPO could be applied, adding an entire training stage.
- Qwen3.5-35B-A3B is a sparse MoE model (35B total, 3B activated per token) — very efficient for both training and inference relative to its capability. GSPO was specifically designed for stable MoE RL training.

**Why GSPO over SFT/DPO/GRPO:**
- **SFT** just imitates the target outputs. GSPO lets the model explore and self-improve through reward signals — it can potentially surpass the quality of the training data.
- **DPO** requires pre-collected preference pairs (chosen/rejected). GSPO generates its own completions and uses reward functions to score them — no manual annotation needed.
- **GRPO** (the predecessor) suffers from instability during long training and can cause irreversible model collapse. GSPO resolves this with sequence-level optimization, achieving better training accuracy and benchmark performance under the same compute budget.
- GSPO is the algorithm behind Qwen3's improvements and is well-supported in HuggingFace TRL (≥v0.28.0).

### How GSPO Works (Relevant to Dataset Design)

For each training prompt, GSPO:
1. Generates a **group** of G completions (e.g., 8–16 per prompt)
2. Scores each completion using **reward functions**
3. Computes a **group-relative advantage** (how much better/worse each completion is vs the group mean)
4. Computes **sequence-level importance ratios** (not token-level like GRPO) and clips at the sequence level
5. Updates the policy to favor higher-reward completions

This means the dataset needs:
- A `prompt` column (the input to generate from)
- Additional columns accessible by reward functions (for scoring correctness/quality)
- **No pre-generated completions** — GSPO generates its own online

### Additional Columns Required in the Improved Dataset

The improved Alpaca dataset from Phase 1 must be extended with columns that enable reward functions to evaluate generated completions. Here is the full target schema:

| Column | Source | Purpose |
|--------|--------|---------|
| `prompt` | **New** (constructed) | Chat-formatted prompt for the model — combines the improved instruction + input |
| `instruction_improved` | Phase 1 | Improved Norwegian instruction |
| `input_improved` | Phase 1 | Improved Norwegian input context |
| `output_improved` | Phase 1 | **Reference answer** — used by reward functions to score completions |
| `instruction_en` | Original | English instruction — enables cross-lingual reward signals |
| `input_en` | Original | English input context |
| `output_en` | Original | English reference — enables cross-lingual reward signals |
| `instruction` | Original | Original Norwegian (kept for provenance) |
| `input` | Original | Original Norwegian (kept for provenance) |
| `output` | Original | Original Norwegian (kept for provenance) |
| `task_type` | **New** (classified) | Category label (e.g., `generation`, `classification`, `extraction`, `rewriting`, `qa`, `creative`) — enables task-specific reward routing |

### Constructing the `prompt` Column

The `prompt` column must be in TRL's **conversational format** (list of message dicts) to work with Qwen3.5's chat template:

```python
def build_prompt(row):
    """Construct a chat-format prompt from the improved Alpaca row."""
    user_content = row["instruction_improved"]
    if row["input_improved"]:
        user_content += f"\n\n{row['input_improved']}"
    return [{"role": "user", "content": user_content}]
```

### Classifying `task_type`

The Alpaca dataset contains a mix of task types. Different tasks need different reward strategies. We classify each row using keyword heuristics on the English instruction (more reliable than Norwegian):

| Task Type | Heuristic (on `instruction_en`) | Example |
|-----------|--------------------------------|---------|
| `classification` | Contains "classify", "identify", "which", "categorize" | "Identify the odd one out" |
| `extraction` | Contains "extract", "list", "find", "name" | "List 3 African countries" |
| `generation` | Contains "write", "generate", "create", "compose" | "Write a haiku about spring" |
| `rewriting` | Contains "rewrite", "paraphrase", "summarize", "simplify" | "Summarize the following text" |
| `qa` | Contains "what", "how", "why", "explain", "describe" | "What is photosynthesis?" |
| `creative` | Contains "story", "poem", "imagine", "invent" | "Write a short story about..." |
| `other` | Default fallback | Everything else |

This classification is a heuristic — it doesn't need to be perfect, just good enough for reward routing.

### Reward Function Design

GSPO's power comes from well-designed reward functions. For Norwegian Alpaca, we compose multiple reward signals:

#### 1. Semantic Similarity Reward (primary)

Measures whether the generated completion preserves the meaning of the reference answer.

```python
from sentence_transformers import SentenceTransformer

similarity_model = SentenceTransformer("intfloat/multilingual-e5-large")

def semantic_reward(completions, output_improved, **kwargs):
    """Reward based on cosine similarity to the reference Norwegian answer."""
    rewards = []
    for completion, reference in zip(completions, output_improved):
        emb_comp = similarity_model.encode(completion, normalize_embeddings=True)
        emb_ref = similarity_model.encode(reference, normalize_embeddings=True)
        sim = float(emb_comp @ emb_ref)
        # Scale: 0.0 at sim<=0.5, 1.0 at sim>=0.95
        reward = max(0.0, min(1.0, (sim - 0.5) / 0.45))
        rewards.append(reward)
    return rewards
```

#### 2. Language Quality Reward

Penalizes outputs that are not in Norwegian or contain excessive English.

```python
import re

def language_reward(completions, **kwargs):
    """Reward for staying in Norwegian. Penalizes English leakage."""
    rewards = []
    for completion in completions:
        # Simple heuristic: check ratio of Norwegian-specific chars (æøå)
        # and absence of common English-only patterns
        norwegian_chars = len(re.findall(r'[æøåÆØÅ]', completion))
        total_chars = max(len(completion), 1)
        # Bonus for Norwegian markers, base reward of 0.5 for non-empty
        reward = 0.5 + min(0.5, norwegian_chars / (total_chars * 0.02))
        if len(completion.strip()) == 0:
            reward = 0.0
        rewards.append(reward)
    return rewards
```

#### 3. Format/Length Reward

Penalizes excessively long or short outputs relative to the reference.

```python
def length_reward(completions, output_improved, **kwargs):
    """Reward outputs that are similar length to the reference."""
    rewards = []
    for completion, reference in zip(completions, output_improved):
        ref_len = max(len(reference), 1)
        comp_len = len(completion)
        ratio = comp_len / ref_len
        # Sweet spot: 0.7x–1.5x of reference length
        if 0.7 <= ratio <= 1.5:
            reward = 1.0
        elif ratio < 0.3 or ratio > 3.0:
            reward = 0.0
        else:
            reward = 0.5
        rewards.append(reward)
    return rewards
```

#### 4. Task-Specific Accuracy Reward (for verifiable tasks)

For classification/extraction tasks where the answer is deterministic:

```python
def accuracy_reward(completions, output_improved, task_type, **kwargs):
    """Exact/fuzzy match for tasks with verifiable answers."""
    rewards = []
    for completion, reference, task in zip(completions, output_improved, task_type):
        if task in ("classification", "extraction", "qa"):
            # Normalized comparison
            comp_norm = completion.strip().lower()
            ref_norm = reference.strip().lower()
            if comp_norm == ref_norm:
                reward = 1.0
            elif ref_norm in comp_norm or comp_norm in ref_norm:
                reward = 0.5
            else:
                reward = 0.0
        else:
            reward = None  # Not applicable — skip for this function
        rewards.append(reward)
    return rewards
```

#### Combined Reward Setup

```python
from trl import GRPOTrainer, GRPOConfig

trainer = GRPOTrainer(
    model="Qwen/Qwen3.5-35B-A3B",
    reward_funcs=[semantic_reward, language_reward, length_reward, accuracy_reward],
    reward_weights=[2.0, 1.0, 0.5, 1.5],  # Semantic similarity is most important
    train_dataset=dataset,
    args=GRPOConfig(
        # GSPO-specific: sequence-level importance sampling
        importance_sampling_level="sequence",
        loss_type="grpo",
        beta=0.04,                   # KL penalty (GSPO paper default)
        epsilon=3e-4,                # Clipping range (tighter than GRPO default)
        # General training config
        per_device_train_batch_size=2,
        num_generations=8,           # Group size G — completions per prompt
        max_completion_length=512,
        learning_rate=5e-7,
        bf16=True,
        use_vllm=True,
        vllm_mode="colocate",
    ),
    peft_config=peft_config,  # LoRA — see below
)
```

**GSPO-critical config:** The key difference from GRPO is `importance_sampling_level="sequence"`. This tells TRL to compute importance ratios at the sequence level (GSPO) rather than per-token (GRPO). The `epsilon=3e-4` is the tighter clipping range recommended in the GSPO paper — much smaller than GRPO's default because sequence-level ratios are more stable.

### Training Strategy: LoRA + GSPO

Full fine-tuning of a 35B parameter model is impractical for most setups. Use LoRA (via PEFT):

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
```

Qwen3.5-35B-A3B-Base's model card explicitly states the control tokens were trained to support efficient LoRA-style PEFT — the instruct variant inherits this. With LoRA rank 16, trainable parameters drop from 35B to ~50M.

### Hardware Requirements for GSPO Training

GSPO is more demanding than SFT because it generates G completions per prompt online:

| Setup | Configuration | Notes |
|-------|---------------|-------|
| **Minimum** | 1× A100 80GB | LoRA + 4-bit quantization + vLLM colocate, G=4 |
| **Recommended** | 2× A100 80GB | LoRA + BF16, vLLM server on 2nd GPU, G=8 |
| **Comfortable** | 4× A100 80GB | ZeRO Stage 3 + vLLM, G=16, larger batches |

Since Qwen3.5-35B-A3B only activates 3B parameters per token, memory and compute are much lower than a dense 35B model.

### Dataset Preparation Pipeline (added to `norai_tools/`)

New file: `norai_tools/gspo_prep.py`

```python
def prepare_gspo_dataset(improved_dataset) -> Dataset:
    """Transform the improved Alpaca dataset into GSPO-ready format.

    Adds:
    - prompt: Conversational format [{"role": "user", "content": ...}]
    - task_type: Classified task category
    Keeps all original + improved columns for reward function access.
    """

def classify_task_type(instruction_en: str) -> str:
    """Heuristic classification of instruction into task category."""

def validate_gspo_dataset(dataset) -> dict:
    """Sanity checks: no empty prompts, task_type distribution, etc."""
```

### Updated Architecture Overview

```
norai_tools/
├── __init__.py
├── improver.py               # Phase 1: Borealis improvement engine
├── prompts.py                # Few-shot prompt templates (Norwegian)
├── validation.py             # Response validation (preamble stripping, hallucination check)
├── loader.py                 # Dataset loading/saving/checkpointing
├── config.py                 # Configuration constants
└── gspo_prep.py              # Phase 2: GSPO dataset preparation

notebooks/
├── improve_alpaca.ipynb      # Phase 1: Standalone improvement notebook
└── gspo_train.ipynb          # Phase 2: GSPO training notebook

pyproject.toml
```

### Updated Implementation Order

| # | Task | Files | Phase |
|---|------|-------|-------|
| 1 | Create `pyproject.toml` with dependencies | `pyproject.toml` | 1 |
| 2 | Implement `config.py` with defaults | `norai_tools/config.py` | 1 |
| 3 | Implement `prompts.py` with few-shot templates | `norai_tools/prompts.py` | 1 |
| 4 | Implement `validation.py` (preamble stripping, hallucination check) | `norai_tools/validation.py` | 1 |
| 5 | Implement `loader.py` (load, save, checkpoint) | `norai_tools/loader.py` | 1 |
| 6 | Implement `improver.py` (core engine) | `norai_tools/improver.py` | 1 |
| 7 | Wire up `__init__.py` exports | `norai_tools/__init__.py` | 1 |
| 8 | Create standalone improvement notebook | `notebooks/improve_alpaca.ipynb` | 1 |
| 9 | Test improvement with a small subset (100 rows) | — | 1 |
| 10 | Run full dataset improvement | — | 1 |
| 11 | Implement `gspo_prep.py` (prompt construction, task classification) | `norai_tools/gspo_prep.py` | 2 |
| 12 | Create GSPO training notebook with reward functions | `notebooks/gspo_train.ipynb` | 2 |
| 13 | Validate GSPO dataset (distributions, spot checks) | — | 2 |
| 14 | Run GSPO training | — | 2 |

### Dependencies Added for Phase 2

Additional packages in `pyproject.toml` under `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
gspo = [
    "trl>=0.28.0",
    "peft>=0.15.0",
    "accelerate",
    "sentence-transformers",
    "vllm",
    "bitsandbytes",        # For 4-bit quantization if needed
]
```
