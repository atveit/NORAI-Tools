"""Default constants for the NORAI-Tools pipeline."""

DEFAULT_MODEL = "NbAiLab/borealis-4b-instruct-preview"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.3
DEFAULT_CHECKPOINT_EVERY = 500
DEFAULT_DEVICE = "auto"
DEFAULT_DTYPE = "bfloat16"
CHECKPOINT_FILE = "alpaca_improved_checkpoint.jsonl"
OUTPUT_FILE = "norwegian_alpaca_improved.parquet"

COLUMN_PAIRS = [
    ("instruction", "instruction_en"),
    ("input", "input_en"),
    ("output", "output_en"),
]

GENERATION_CONFIG = {
    "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
    "do_sample": True,
    "temperature": DEFAULT_TEMPERATURE,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}
