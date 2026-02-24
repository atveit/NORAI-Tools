"""Prompt templates for Norwegian text improvement using Borealis."""

IMPROVEMENT_PROMPT_TEMPLATE = """Forbedre denne norske teksten. Se etter:

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

{norwegian_text}"""


def build_improvement_prompt(norwegian_text: str, english_text: str) -> str:
    """Build a prompt for improving Norwegian text using Borealis.

    Args:
        norwegian_text: The Norwegian text to improve.
        english_text: The English reference text (for meaning verification).

    Returns:
        The formatted prompt string.
    """
    return IMPROVEMENT_PROMPT_TEMPLATE.format(
        norwegian_text=norwegian_text,
        english_text=english_text,
    )


def should_skip(norwegian_text: str, english_text: str) -> bool:
    """Determine if a text field should be skipped (not sent to the model).

    Skip conditions:
    - Empty or whitespace-only Norwegian text
    - Norwegian text identical to English text (e.g., numbers, proper nouns, code)
    - Very short text (< 5 characters)

    Args:
        norwegian_text: The Norwegian text to check.
        english_text: The corresponding English text.

    Returns:
        True if the field should be skipped (copied as-is).
    """
    if not norwegian_text or not norwegian_text.strip():
        return True
    if norwegian_text.strip() == english_text.strip():
        return True
    if len(norwegian_text.strip()) < 5:
        return True
    return False
