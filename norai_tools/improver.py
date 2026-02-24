"""Core improvement engine for Norwegian Alpaca dataset."""

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from norai_tools.config import (
    COLUMN_PAIRS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL,
    GENERATION_CONFIG,
)
from norai_tools.loader import load_checkpoint, save_checkpoint
from norai_tools.prompts import build_improvement_prompt, should_skip
from norai_tools.validation import validate_response


class AlpacaImprover:
    """Batch-based Norwegian text improver using Borealis.

    Args:
        model_name: HuggingFace model ID.
        device: Device to use ("auto", "cuda", "mps", "cpu").
        batch_size: Number of prompts to process at once.
        max_new_tokens: Maximum tokens to generate per prompt.
        dtype: Model dtype ("bfloat16" or "float32").
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        dtype: str = DEFAULT_DTYPE,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
        )
        self.model.eval()

    def improve_text(self, norwegian_text: str, english_text: str) -> str:
        """Improve a single Norwegian text using the model.

        Args:
            norwegian_text: The Norwegian text to improve.
            english_text: The English reference text.

        Returns:
            The improved Norwegian text, or the original if improvement fails.
        """
        if should_skip(norwegian_text, english_text):
            return norwegian_text

        prompt = build_improvement_prompt(norwegian_text, english_text)
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        gen_config = {**GENERATION_CONFIG, "max_new_tokens": self.max_new_tokens}

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_config)

        # Decode only the new tokens (exclude the prompt)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return validate_response(response, norwegian_text)

    def improve_batch(self, batch: list[tuple[str, str]]) -> list[str]:
        """Improve a batch of Norwegian texts.

        Args:
            batch: List of (norwegian_text, english_text) tuples.

        Returns:
            List of improved texts.
        """
        # Separate texts that need improvement from those that should be skipped
        to_process = []
        results = [None] * len(batch)

        for i, (no_text, en_text) in enumerate(batch):
            if should_skip(no_text, en_text):
                results[i] = no_text
            else:
                to_process.append((i, no_text, en_text))

        if not to_process:
            return results

        # Build prompts and tokenize
        prompts = []
        for _, no_text, en_text in to_process:
            prompt = build_improvement_prompt(no_text, en_text)
            messages = [{"role": "user", "content": prompt}]
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(input_text)

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        gen_config = {**GENERATION_CONFIG, "max_new_tokens": self.max_new_tokens}

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_config)

        # Decode each output, stripping the prompt
        for j, (idx, no_text, _) in enumerate(to_process):
            prompt_len = inputs["attention_mask"][j].sum().item()
            new_tokens = outputs[j][prompt_len:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            results[idx] = validate_response(response, no_text)

        return results

    def improve_row(self, row: dict) -> dict:
        """Improve all Norwegian columns in a single dataset row.

        Args:
            row: A dataset row dict with Norwegian and English columns.

        Returns:
            The row dict with added *_improved columns.
        """
        improved = dict(row)
        for no_col, en_col in COLUMN_PAIRS:
            no_text = row.get(no_col, "") or ""
            en_text = row.get(en_col, "") or ""
            improved_text = self.improve_text(no_text, en_text)
            improved[f"{no_col}_improved"] = improved_text
        return improved

    def improve_dataset(
        self,
        dataset: Dataset,
        checkpoint_path: str | None = None,
        checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    ) -> Dataset:
        """Improve an entire dataset with checkpointing.

        Args:
            dataset: The input dataset.
            checkpoint_path: Path to save/load checkpoints.
            checkpoint_every: Save checkpoint every N rows.

        Returns:
            A new Dataset with added *_improved columns.
        """
        # Load existing checkpoint if available
        processed_rows, start_idx = load_checkpoint(checkpoint_path)

        rows = list(dataset)
        for i in tqdm(range(start_idx, len(rows), self.batch_size), desc="Improving"):
            batch_rows = rows[i : i + self.batch_size]

            for no_col, en_col in COLUMN_PAIRS:
                batch_pairs = [
                    (r.get(no_col, "") or "", r.get(en_col, "") or "")
                    for r in batch_rows
                ]
                improved_texts = self.improve_batch(batch_pairs)
                for j, text in enumerate(improved_texts):
                    batch_rows[j][f"{no_col}_improved"] = text

            processed_rows.extend(batch_rows)

            # Checkpoint periodically
            if len(processed_rows) % checkpoint_every < self.batch_size:
                save_checkpoint(batch_rows, checkpoint_path)

        return Dataset.from_list(processed_rows)
