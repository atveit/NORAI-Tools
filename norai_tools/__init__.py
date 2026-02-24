"""norai-tools — Norwegian AI dataset tools."""

from norai_tools.config import (
    DEFAULT_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    CHECKPOINT_FILE,
    OUTPUT_FILE,
)
from norai_tools.prompts import build_improvement_prompt, should_skip
from norai_tools.validation import validate_response
from norai_tools.loader import load_alpaca, save_improved, load_checkpoint, save_checkpoint
from norai_tools.improver import AlpacaImprover
from norai_tools.gspo_prep import prepare_gspo_dataset, classify_task_type, validate_gspo_dataset
from norai_tools.rewards import semantic_reward, language_reward, length_reward, accuracy_reward

__all__ = [
    # config
    "DEFAULT_MODEL",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_CHECKPOINT_EVERY",
    "DEFAULT_DEVICE",
    "DEFAULT_DTYPE",
    "CHECKPOINT_FILE",
    "OUTPUT_FILE",
    # prompts
    "build_improvement_prompt",
    "should_skip",
    # validation
    "validate_response",
    # loader
    "load_alpaca",
    "save_improved",
    "load_checkpoint",
    "save_checkpoint",
    # improver
    "AlpacaImprover",
    # gspo_prep
    "prepare_gspo_dataset",
    "classify_task_type",
    "validate_gspo_dataset",
    # rewards
    "semantic_reward",
    "language_reward",
    "length_reward",
    "accuracy_reward",
]
