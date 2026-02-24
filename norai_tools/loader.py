"""Dataset loading, saving, and checkpoint management."""

import json
from pathlib import Path

from datasets import Dataset, load_dataset

from norai_tools.config import CHECKPOINT_FILE, OUTPUT_FILE


def load_alpaca() -> Dataset:
    """Load the NbAiLab/norwegian-alpaca dataset from HuggingFace.

    Returns:
        The train split of the norwegian-alpaca dataset.
    """
    ds = load_dataset("NbAiLab/norwegian-alpaca", split="train")
    return ds


def save_improved(dataset: Dataset, path: str | None = None, push_to_hub: bool = False, hub_repo: str | None = None) -> None:
    """Save the improved dataset as Parquet and optionally push to HuggingFace Hub.

    Args:
        dataset: The improved dataset to save.
        path: Output file path. Defaults to OUTPUT_FILE constant.
        push_to_hub: Whether to push the dataset to HuggingFace Hub.
        hub_repo: The Hub repository ID (e.g., "username/dataset-name").
    """
    path = path or OUTPUT_FILE
    dataset.to_parquet(path)
    if push_to_hub:
        if not hub_repo:
            raise ValueError("hub_repo must be specified when push_to_hub=True")
        dataset.push_to_hub(hub_repo)


def load_checkpoint(path: str | None = None) -> tuple[list[dict], int]:
    """Load an existing checkpoint JSONL file.

    Args:
        path: Path to the checkpoint JSONL file. Defaults to CHECKPOINT_FILE.

    Returns:
        A tuple of (processed_rows, resume_index) where resume_index is
        the number of rows already processed.
    """
    path = path or CHECKPOINT_FILE
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return [], 0

    rows = []
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows, len(rows)


def save_checkpoint(rows: list[dict], path: str | None = None) -> None:
    """Append rows to a checkpoint JSONL file.

    Args:
        rows: List of row dicts to append.
        path: Path to the checkpoint JSONL file. Defaults to CHECKPOINT_FILE.
    """
    path = path or CHECKPOINT_FILE
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
