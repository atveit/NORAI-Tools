"""Response validation for LLM-generated Norwegian text improvements."""

PREAMBLE_INDICATORS = [
    "her er",
    "forbedret tekst:",
    "endringer:",
    "jeg har",
    "teksten er",
    "original:",
    "###",
    "bedre:",
    "resultat:",
]


def validate_response(response: str, original: str) -> str:
    """Validate and clean an LLM response. Returns original if invalid.

    Applies the following checks in order:
    1. Too short (< 10 chars) — probably a failed generation
    2. Preamble stripping — small models often prepend "Her er..." etc.
    3. Hallucination check — < 30% word overlap means the model invented new content
    4. Length sanity — > 3x original length means runaway generation

    Args:
        response: The raw LLM response text.
        original: The original Norwegian text (fallback if validation fails).

    Returns:
        The cleaned response text, or the original if validation fails.
    """
    refined = response.strip()

    # Too short — probably failed
    if len(refined) < 10:
        return original

    # Strip preambles that small models add
    refined_lower = refined.lower()
    for indicator in PREAMBLE_INDICATORS:
        if refined_lower.startswith(indicator):
            # Try to extract just the text after the preamble
            lines = refined.split("\n")
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
