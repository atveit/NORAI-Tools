"""GSPO dataset preparation — transform improved Alpaca data for alignment training."""

from datasets import Dataset


# Task classification keywords — matched against English instruction (more reliable)
TASK_KEYWORDS = {
    "classification": ["classify", "identify", "which", "categorize", "determine", "label"],
    "extraction": ["extract", "list", "find", "name", "enumerate", "mention"],
    "generation": ["write", "generate", "create", "compose", "draft", "produce"],
    "rewriting": ["rewrite", "paraphrase", "summarize", "simplify", "rephrase", "translate"],
    "qa": ["what", "how", "why", "explain", "describe", "define"],
    "creative": ["story", "poem", "imagine", "invent", "fiction", "creative"],
}


def classify_task_type(instruction_en: str) -> str:
    """Classify an instruction into a task category using keyword heuristics.

    Args:
        instruction_en: The English instruction text.

    Returns:
        A task type string: one of "classification", "extraction", "generation",
        "rewriting", "qa", "creative", or "other".
    """
    if not instruction_en:
        return "other"

    instruction_lower = instruction_en.lower()
    for task_type, keywords in TASK_KEYWORDS.items():
        for keyword in keywords:
            if keyword in instruction_lower:
                return task_type
    return "other"


def build_prompt(row: dict) -> list[dict]:
    """Construct a chat-format prompt from an improved Alpaca row.

    The prompt uses TRL's conversational format (list of message dicts)
    compatible with Qwen3.5's chat template.

    Args:
        row: A dataset row with improved columns.

    Returns:
        A list of message dicts for the chat template.
    """
    user_content = row.get("instruction_improved", "") or ""
    input_text = row.get("input_improved", "") or ""
    if input_text.strip():
        user_content += f"\n\n{input_text}"
    return [{"role": "user", "content": user_content}]


def prepare_gspo_dataset(improved_dataset: Dataset) -> Dataset:
    """Transform the improved Alpaca dataset into GSPO-ready format.

    Adds:
    - prompt: Conversational format [{"role": "user", "content": ...}]
    - task_type: Classified task category

    Keeps all original + improved columns for reward function access.

    Args:
        improved_dataset: The dataset from Phase 1 with *_improved columns.

    Returns:
        A new Dataset with prompt and task_type columns added.
    """
    def add_gspo_columns(row):
        row["prompt"] = build_prompt(row)
        row["task_type"] = classify_task_type(row.get("instruction_en", ""))
        return row

    return improved_dataset.map(add_gspo_columns)


def validate_gspo_dataset(dataset: Dataset) -> dict:
    """Run sanity checks on a GSPO-ready dataset.

    Args:
        dataset: The GSPO-prepared dataset.

    Returns:
        A dict with validation results:
        - total_rows: Number of rows
        - empty_prompts: Count of rows with empty prompt content
        - task_type_distribution: Dict of task_type -> count
        - missing_columns: List of required columns that are missing
        - is_valid: Whether all checks pass
    """
    required_columns = [
        "prompt", "task_type", "output_improved",
        "instruction_improved", "instruction_en",
    ]

    missing = [col for col in required_columns if col not in dataset.column_names]

    empty_prompts = 0
    task_distribution = {}

    for row in dataset:
        # Check for empty prompts
        prompt = row.get("prompt", [])
        if not prompt or not prompt[0].get("content", "").strip():
            empty_prompts += 1

        # Count task types
        task_type = row.get("task_type", "other")
        task_distribution[task_type] = task_distribution.get(task_type, 0) + 1

    is_valid = len(missing) == 0 and empty_prompts == 0

    return {
        "total_rows": len(dataset),
        "empty_prompts": empty_prompts,
        "task_type_distribution": task_distribution,
        "missing_columns": missing,
        "is_valid": is_valid,
    }
