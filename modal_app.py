"""NORAI-Tools Modal runner — run the full pipeline on Modal cloud GPUs.

Usage:
    modal run modal_app.py --step improve
    modal run modal_app.py --step prepare-gspo
    modal run modal_app.py --step distill-generate
    modal run modal_app.py --step distill-polish
    modal run modal_app.py --step gspo-train
    modal run modal_app.py --step gspo-train-distilled
    modal run modal_app.py --step all              # Phases 1 → 1.5 → 2
    modal run modal_app.py --step all-distilled    # Phases 1 → 1b → 2b
"""

import modal

# ── App + Volumes ──────────────────────────────────────────────
app = modal.App("norai-tools")

data_vol = modal.Volume.from_name("norai-data", create_if_missing=True)
models_vol = modal.Volume.from_name("norai-models", create_if_missing=True)

VOLUMES = {"/data": data_vol, "/models": models_vol}

# ── Secrets (optional — for HuggingFace Hub push) ─────────────
hf_secret = modal.Secret.from_name("huggingface", required=False)

# ── Data paths on the /data volume ────────────────────────────
DATA_DIR = "/data"
IMPROVED_PARQUET = f"{DATA_DIR}/norwegian_alpaca_improved.parquet"
GSPO_PARQUET = f"{DATA_DIR}/norwegian_alpaca_gspo.parquet"
QWEN_RAW_PARQUET = f"{DATA_DIR}/norwegian_alpaca_qwen_raw.parquet"
QWEN_POLISHED_PARQUET = f"{DATA_DIR}/norwegian_alpaca_qwen_polished.parquet"

IMPROVE_CHECKPOINT = f"{DATA_DIR}/alpaca_improved_checkpoint.jsonl"
QWEN_GEN_CHECKPOINT = f"{DATA_DIR}/qwen_generation_checkpoint.jsonl"
BOREALIS_POLISH_CHECKPOINT = f"{DATA_DIR}/borealis_polish_checkpoint.jsonl"

GSPO_OUTPUT_DIR = "/models/gspo_qwen_norwegian"
GSPO_DISTILLED_OUTPUT_DIR = "/models/gspo_qwen_norwegian_distilled"

# ── Image definitions ──────────────────────────────────────────
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "tqdm",
        "pandas",
        "huggingface_hub",
    )
    .add_local_python_source("norai_tools")
)

gspo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "tqdm",
        "pandas",
        "huggingface_hub",
        "trl>=0.28.0",
        "peft>=0.15.0",
        "accelerate",
        "sentence-transformers",
        "vllm",
        "bitsandbytes",
    )
    .add_local_python_source("norai_tools")
)


# ── Phase 1: Improve Norwegian Alpaca with Borealis ───────────
@app.function(
    gpu="L4",
    image=base_image,
    volumes=VOLUMES,
    timeout=43200,  # 12 hours
    secrets=[hf_secret],
)
def improve_alpaca(
    push_to_hub: bool = False,
    hub_repo: str = "",
    batch_size: int = 8,
    subset: int = 0,
):
    """Phase 1: Improve Norwegian Alpaca dataset with Borealis 4B."""
    from norai_tools import AlpacaImprover, load_alpaca, save_improved

    dataset = load_alpaca()
    if subset > 0:
        dataset = dataset.select(range(min(subset, len(dataset))))
    print(f"Dataset: {len(dataset)} rows")

    improver = AlpacaImprover(batch_size=batch_size)
    improver.load_model()
    print(f"Model loaded: {improver.model_name}")

    improved = improver.improve_dataset(
        dataset,
        checkpoint_path=IMPROVE_CHECKPOINT,
        checkpoint_every=500,
    )

    save_improved(
        improved,
        path=IMPROVED_PARQUET,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo or None,
    )
    data_vol.commit()
    print(f"Saved {len(improved)} rows to {IMPROVED_PARQUET}")


# ── Phase 1.5: Prepare GSPO dataset ───────────────────────────
@app.function(
    image=base_image,
    volumes=VOLUMES,
    timeout=600,  # 10 minutes
)
def prepare_gspo(input_path: str = ""):
    """Phase 1.5: Add prompt + task_type columns for GSPO training."""
    from datasets import load_dataset
    from norai_tools import prepare_gspo_dataset, validate_gspo_dataset

    src = input_path or IMPROVED_PARQUET
    data_vol.reload()
    dataset = load_dataset("parquet", data_files=src, split="train")
    print(f"Loaded: {len(dataset)} rows from {src}")

    gspo_dataset = prepare_gspo_dataset(dataset)
    validation = validate_gspo_dataset(gspo_dataset)

    print(f"Validation: valid={validation['is_valid']}, "
          f"empty_prompts={validation['empty_prompts']}")
    print(f"Task types: {validation['task_type_distribution']}")

    if not validation["is_valid"]:
        raise RuntimeError(f"GSPO dataset validation failed: {validation}")

    gspo_dataset.to_parquet(GSPO_PARQUET)
    data_vol.commit()
    print(f"Saved {len(gspo_dataset)} rows to {GSPO_PARQUET}")


# ── Phase 1b Pass 1: Qwen generation ──────────────────────────
@app.function(
    gpu="A100-80GB",
    image=base_image,
    volumes=VOLUMES,
    timeout=86400,  # 24 hours
    secrets=[hf_secret],
)
def distill_generate(
    batch_size: int = 4,
    subset: int = 0,
):
    """Phase 1b Pass 1: Generate Norwegian outputs with Qwen3.5-35B-A3B."""
    import gc
    import json
    import os

    import torch
    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data_vol.reload()
    dataset = load_dataset("parquet", data_files=IMPROVED_PARQUET, split="train")
    if subset > 0:
        dataset = dataset.select(range(min(subset, len(dataset))))
    print(f"Dataset: {len(dataset)} rows")

    # Resume from checkpoint
    qwen_outputs = []
    if os.path.exists(QWEN_GEN_CHECKPOINT):
        with open(QWEN_GEN_CHECKPOINT, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    qwen_outputs.append(json.loads(line.strip())["output"])
        print(f"Resumed: {len(qwen_outputs)} rows already done.")

    resume_idx = len(qwen_outputs)
    total = len(dataset)

    if resume_idx >= total:
        print("All rows already generated!")
    else:
        print(f"Generating rows {resume_idx}–{total - 1} ({total - resume_idx} remaining)")

        model_name = "Qwen/Qwen3.5-35B-A3B"
        system_prompt = "Du er en hjelpsom assistent. Svar alltid på norsk bokmål."

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model.eval()
        print("Qwen model loaded.")

        for start in tqdm(range(resume_idx, total, batch_size), desc="Generating"):
            end = min(start + batch_size, total)
            batch = dataset.select(range(start, end))

            formatted_prompts = []
            for row in batch:
                user_content = row["instruction_improved"]
                input_text = row.get("input_improved", "") or ""
                if input_text.strip():
                    user_content += f"\n\n{input_text}"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                formatted_prompts.append(formatted)

            inputs = tokenizer(
                formatted_prompts, return_tensors="pt",
                padding=True, truncation=True, max_length=2048,
            ).to(model.device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs, max_new_tokens=512,
                    do_sample=True, temperature=0.7, top_p=0.9,
                )

            padded_len = inputs["input_ids"].shape[1]
            batch_texts = []
            for i in range(len(formatted_prompts)):
                text = tokenizer.decode(
                    outputs[i][padded_len:], skip_special_tokens=True
                )
                batch_texts.append(text)

            qwen_outputs.extend(batch_texts)

            with open(QWEN_GEN_CHECKPOINT, "a", encoding="utf-8") as f:
                for text in batch_texts:
                    f.write(json.dumps({"output": text}, ensure_ascii=False) + "\n")

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print("Qwen model unloaded.")

    # Save intermediate parquet
    dataset = dataset.add_column("output_qwen_raw", qwen_outputs)
    dataset.to_parquet(QWEN_RAW_PARQUET)
    data_vol.commit()
    print(f"Saved {len(dataset)} rows to {QWEN_RAW_PARQUET}")


# ── Phase 1b Pass 2: Borealis polish ──────────────────────────
@app.function(
    gpu="L4",
    image=base_image,
    volumes=VOLUMES,
    timeout=43200,  # 12 hours
)
def distill_polish(
    batch_size: int = 8,
    subset: int = 0,
):
    """Phase 1b Pass 2: Polish Qwen outputs with Borealis."""
    import json
    import os

    from datasets import load_dataset
    from norai_tools import AlpacaImprover
    from tqdm import tqdm

    data_vol.reload()
    dataset = load_dataset("parquet", data_files=QWEN_RAW_PARQUET, split="train")
    if subset > 0:
        dataset = dataset.select(range(min(subset, len(dataset))))
    print(f"Dataset: {len(dataset)} rows")

    # Resume from checkpoint
    polished_outputs = []
    if os.path.exists(BOREALIS_POLISH_CHECKPOINT):
        with open(BOREALIS_POLISH_CHECKPOINT, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    polished_outputs.append(json.loads(line.strip())["output"])
        print(f"Resumed: {len(polished_outputs)} rows done.")

    resume_idx = len(polished_outputs)

    if resume_idx >= len(dataset):
        print("All rows already polished!")
    else:
        print(f"Polishing rows {resume_idx}–{len(dataset) - 1}")

        improver = AlpacaImprover(batch_size=batch_size)
        improver.load_model()
        print(f"Borealis loaded: {improver.model_name}")

        for start in tqdm(range(resume_idx, len(dataset), batch_size), desc="Polishing"):
            end = min(start + batch_size, len(dataset))
            batch_pairs = [
                (dataset[i]["output_qwen_raw"], dataset[i]["instruction_en"])
                for i in range(start, end)
            ]
            polished = improver.improve_batch(batch_pairs)
            polished_outputs.extend(polished)

            with open(BOREALIS_POLISH_CHECKPOINT, "a", encoding="utf-8") as f:
                for text in polished:
                    f.write(json.dumps({"output": text}, ensure_ascii=False) + "\n")

    dataset = dataset.add_column("output_qwen_polished", polished_outputs)
    dataset.to_parquet(QWEN_POLISHED_PARQUET)
    data_vol.commit()
    print(f"Saved {len(dataset)} rows to {QWEN_POLISHED_PARQUET}")


# ── Phase 2: GSPO alignment training ──────────────────────────
@app.function(
    gpu="A100-80GB:2",
    image=gspo_image,
    volumes=VOLUMES,
    timeout=259200,  # 72 hours
    secrets=[hf_secret],
)
def gspo_train(
    num_generations: int = 8,
    learning_rate: float = 5e-7,
    num_epochs: int = 1,
    save_steps: int = 200,
):
    """Phase 2: GSPO alignment training on Qwen3.5-35B-A3B."""
    from datasets import load_dataset
    from norai_tools import (
        accuracy_reward,
        language_reward,
        length_reward,
        semantic_reward,
    )
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    data_vol.reload()
    dataset = load_dataset("parquet", data_files=GSPO_PARQUET, split="train")
    print(f"Dataset: {len(dataset)} rows")

    # Pre-load semantic similarity model
    print("Loading semantic similarity model...")
    _ = semantic_reward(
        completions=[[{"content": "test"}]], output_improved=["test"]
    )
    print("Loaded.")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=GSPO_OUTPUT_DIR,
        importance_sampling_level="sequence",
        loss_type="grpo",
        beta=0.04,
        epsilon=3e-4,
        num_generations=num_generations,
        max_completion_length=512,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        bf16=True,
        use_vllm=True,
        vllm_mode="colocate",
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=3,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model="Qwen/Qwen3.5-35B-A3B",
        reward_funcs=[semantic_reward, language_reward, length_reward, accuracy_reward],
        reward_weights=[2.0, 1.0, 0.5, 1.5],
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
    )

    print("Starting GSPO training...")
    result = trainer.train()
    trainer.save_model(GSPO_OUTPUT_DIR)
    models_vol.commit()
    print(f"Training complete. Loss: {result.training_loss:.4f}")
    print(f"LoRA adapter saved to {GSPO_OUTPUT_DIR}")


# ── Phase 2b: GSPO training on distilled dataset ──────────────
@app.function(
    gpu="A100-80GB:2",
    image=gspo_image,
    volumes=VOLUMES,
    timeout=259200,  # 72 hours
    secrets=[hf_secret],
)
def gspo_train_distilled(
    num_generations: int = 8,
    learning_rate: float = 5e-7,
    num_epochs: int = 1,
    save_steps: int = 200,
):
    """Phase 2b: GSPO training on Qwen-distilled dataset."""
    from datasets import load_dataset
    from norai_tools import (
        accuracy_reward,
        language_reward,
        length_reward,
        prepare_gspo_dataset,
        semantic_reward,
        validate_gspo_dataset,
    )
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    data_vol.reload()
    dataset = load_dataset("parquet", data_files=QWEN_POLISHED_PARQUET, split="train")
    print(f"Dataset: {len(dataset)} rows")

    # Map output_qwen_polished → output_improved for reward functions
    if "output_improved" in dataset.column_names:
        dataset = dataset.rename_column("output_improved", "output_improved_phase1")
    dataset = dataset.rename_column("output_qwen_polished", "output_improved")
    print("Renamed: output_qwen_polished → output_improved")

    dataset = prepare_gspo_dataset(dataset)
    validation = validate_gspo_dataset(dataset)
    if not validation["is_valid"]:
        raise RuntimeError(f"Validation failed: {validation}")
    print(f"Task types: {validation['task_type_distribution']}")

    # Pre-load semantic model
    _ = semantic_reward(
        completions=[[{"content": "test"}]], output_improved=["test"]
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=GSPO_DISTILLED_OUTPUT_DIR,
        importance_sampling_level="sequence",
        loss_type="grpo",
        beta=0.04,
        epsilon=3e-4,
        num_generations=num_generations,
        max_completion_length=512,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        bf16=True,
        use_vllm=True,
        vllm_mode="colocate",
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=3,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model="Qwen/Qwen3.5-35B-A3B",
        reward_funcs=[semantic_reward, language_reward, length_reward, accuracy_reward],
        reward_weights=[2.0, 1.0, 0.5, 1.5],
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
    )

    print("Starting GSPO training (distilled)...")
    result = trainer.train()
    trainer.save_model(GSPO_DISTILLED_OUTPUT_DIR)
    models_vol.commit()
    print(f"Training complete. Loss: {result.training_loss:.4f}")
    print(f"LoRA adapter saved to {GSPO_DISTILLED_OUTPUT_DIR}")


# ── CLI entry point ────────────────────────────────────────────
@app.local_entrypoint()
def main(
    step: str = "improve",
    push_to_hub: bool = False,
    hub_repo: str = "",
    batch_size: int = 0,
    subset: int = 0,
    num_generations: int = 8,
    learning_rate: float = 5e-7,
    num_epochs: int = 1,
    save_steps: int = 200,
):
    """Run NORAI-Tools pipeline steps on Modal cloud GPUs.

    Steps:
        improve              Phase 1: Borealis improves Norwegian Alpaca
        prepare-gspo         Phase 1.5: Add prompt + task_type columns
        distill-generate     Phase 1b Pass 1: Qwen generates Norwegian outputs
        distill-polish       Phase 1b Pass 2: Borealis polishes Qwen outputs
        gspo-train           Phase 2: GSPO alignment training
        gspo-train-distilled Phase 2b: GSPO on distilled dataset
        all                  Run Phases 1 → 1.5 → 2 sequentially
        all-distilled        Run Phases 1 → 1b → 2b sequentially
    """
    steps = {
        "improve": lambda: improve_alpaca.remote(
            push_to_hub=push_to_hub,
            hub_repo=hub_repo,
            batch_size=batch_size or 8,
            subset=subset,
        ),
        "prepare-gspo": lambda: prepare_gspo.remote(),
        "distill-generate": lambda: distill_generate.remote(
            batch_size=batch_size or 4,
            subset=subset,
        ),
        "distill-polish": lambda: distill_polish.remote(
            batch_size=batch_size or 8,
            subset=subset,
        ),
        "gspo-train": lambda: gspo_train.remote(
            num_generations=num_generations,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            save_steps=save_steps,
        ),
        "gspo-train-distilled": lambda: gspo_train_distilled.remote(
            num_generations=num_generations,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            save_steps=save_steps,
        ),
    }

    if step == "all":
        print("Running full standard pipeline: improve → prepare-gspo → gspo-train")
        steps["improve"]()
        steps["prepare-gspo"]()
        steps["gspo-train"]()
    elif step == "all-distilled":
        print("Running full distilled pipeline: improve → distill → gspo-train-distilled")
        steps["improve"]()
        steps["distill-generate"]()
        steps["distill-polish"]()
        steps["gspo-train-distilled"]()
    elif step in steps:
        print(f"Running step: {step}")
        steps[step]()
    else:
        valid = list(steps.keys()) + ["all", "all-distilled"]
        raise ValueError(f"Unknown step: {step}. Valid steps: {', '.join(valid)}")

    print("Done.")
