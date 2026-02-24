"""Tests for norai_tools.validation."""

from norai_tools.validation import validate_response, PREAMBLE_INDICATORS


class TestValidateResponse:
    def test_valid_response_returned(self):
        original = "Han gjorde en beslutning om å reise."
        response = "Han bestemte seg for å reise."
        assert validate_response(response, original) == "Han bestemte seg for å reise."

    def test_too_short_returns_original(self):
        original = "En lang norsk setning som er ganske bra."
        response = "Kort"
        assert validate_response(response, original) == original

    def test_empty_response_returns_original(self):
        original = "Noe tekst her."
        assert validate_response("", original) == original

    def test_whitespace_response_returns_original(self):
        original = "Noe tekst her."
        assert validate_response("   ", original) == original

    def test_preamble_stripped_with_clean_line(self):
        original = "Han hadde ikke noen idé om hva som skjedde."
        response = "Her er den forbedrede teksten:\nHan ante ikke hva som foregikk med situasjonen."
        result = validate_response(response, original)
        assert result == "Han ante ikke hva som foregikk med situasjonen."

    def test_preamble_no_clean_line_returns_original(self):
        original = "Noe tekst her som er ganske lang."
        response = "Her er teksten:\nKort"
        assert validate_response(response, original) == original

    def test_hallucination_returns_original(self):
        original = "Katten satt på matta og sov."
        response = "Universet ekspanderer med akselererende hastighet i alle retninger."
        assert validate_response(response, original) == original

    def test_runaway_generation_returns_original(self):
        original = "Kort tekst."
        response = "A " * 500
        assert validate_response(response, original) == original

    def test_acceptable_length_expansion(self):
        original = "Han gjorde en beslutning om å forlate stedet fordi det var nødvendig."
        response = "Han bestemte seg for å dra fordi det var helt nødvendig for ham."
        assert validate_response(response, original) == response

    def test_strips_whitespace(self):
        original = "Han gjorde en beslutning om å forlate stedet."
        response = "  Han bestemte seg for å dra fra stedet.  "
        assert validate_response(response, original) == "Han bestemte seg for å dra fra stedet."

    def test_overlap_at_boundary(self):
        # Exactly 30% overlap should pass
        original = "en to tre fire fem seks sju åtte ni ti"
        # 3 out of 10 words overlap = 30%
        response = "en to tre alpha beta gamma delta epsilon zeta theta"
        result = validate_response(response, original)
        assert result == response

    def test_overlap_below_boundary(self):
        original = "en to tre fire fem seks sju åtte ni ti"
        # 2 out of 10 = 20% overlap — below threshold
        response = "en to alpha beta gamma delta epsilon zeta theta kappa"
        result = validate_response(response, original)
        assert result == original


class TestPreambleIndicators:
    def test_indicators_are_lowercase(self):
        for indicator in PREAMBLE_INDICATORS:
            assert indicator == indicator.lower()

    def test_indicators_not_empty(self):
        assert len(PREAMBLE_INDICATORS) > 0
        for indicator in PREAMBLE_INDICATORS:
            assert len(indicator) > 0
