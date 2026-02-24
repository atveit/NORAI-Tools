"""Reward functions for GSPO alignment training."""

import re


def semantic_reward(completions: list[list[dict]], output_improved: list[str], **kwargs) -> list[float]:
    """Reward based on cosine similarity to the reference Norwegian answer.

    Uses a multilingual sentence transformer model for embedding comparison.
    Lazy-loads the model on first call.

    Args:
        completions: List of completion message lists from the model.
            Each completion is a list of message dicts with "content" key.
        output_improved: List of reference Norwegian texts.

    Returns:
        List of reward scores in [0.0, 1.0].
    """
    if not hasattr(semantic_reward, "_model"):
        from sentence_transformers import SentenceTransformer
        semantic_reward._model = SentenceTransformer("intfloat/multilingual-e5-large")

    model = semantic_reward._model
    rewards = []
    for completion, reference in zip(completions, output_improved):
        # Extract text from completion messages
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        emb_comp = model.encode(text, normalize_embeddings=True)
        emb_ref = model.encode(reference, normalize_embeddings=True)
        sim = float(emb_comp @ emb_ref)
        # Scale: 0.0 at sim<=0.5, 1.0 at sim>=0.95
        reward = max(0.0, min(1.0, (sim - 0.5) / 0.45))
        rewards.append(reward)
    return rewards


def language_reward(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward for staying in Norwegian. Penalizes English leakage.

    Uses a simple heuristic: checks ratio of Norwegian-specific characters
    (æøå) and penalizes empty outputs.

    Args:
        completions: List of completion message lists from the model.

    Returns:
        List of reward scores in [0.0, 1.0].
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        if len(text.strip()) == 0:
            rewards.append(0.0)
            continue
        norwegian_chars = len(re.findall(r"[æøåÆØÅ]", text))
        total_chars = max(len(text), 1)
        reward = 0.5 + min(0.5, norwegian_chars / (total_chars * 0.02))
        rewards.append(reward)
    return rewards


def length_reward(completions: list[list[dict]], output_improved: list[str], **kwargs) -> list[float]:
    """Reward outputs that are similar length to the reference.

    Sweet spot: 0.7x–1.5x of reference length gets full reward.

    Args:
        completions: List of completion message lists from the model.
        output_improved: List of reference Norwegian texts.

    Returns:
        List of reward scores in {0.0, 0.5, 1.0}.
    """
    rewards = []
    for completion, reference in zip(completions, output_improved):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        ref_len = max(len(reference), 1)
        comp_len = len(text)
        ratio = comp_len / ref_len
        if 0.7 <= ratio <= 1.5:
            reward = 1.0
        elif ratio < 0.3 or ratio > 3.0:
            reward = 0.0
        else:
            reward = 0.5
        rewards.append(reward)
    return rewards


def accuracy_reward(completions: list[list[dict]], output_improved: list[str], task_type: list[str], **kwargs) -> list[float]:
    """Exact/fuzzy match reward for tasks with verifiable answers.

    Only applies to classification, extraction, and qa tasks.
    For other task types, returns 0.0 (neutral).

    Args:
        completions: List of completion message lists from the model.
        output_improved: List of reference Norwegian texts.
        task_type: List of task type strings.

    Returns:
        List of reward scores in {0.0, 0.5, 1.0}.
    """
    rewards = []
    for completion, reference, task in zip(completions, output_improved, task_type):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        if task in ("classification", "extraction", "qa"):
            comp_norm = text.strip().lower()
            ref_norm = reference.strip().lower()
            if comp_norm == ref_norm:
                reward = 1.0
            elif ref_norm in comp_norm or comp_norm in ref_norm:
                reward = 0.5
            else:
                reward = 0.0
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards
