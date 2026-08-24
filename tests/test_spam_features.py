"""Tests for the 12 spam features and the redaction filter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.spam_features import (
    FEATURE_ORDER,
    all_caps_word_count,
    build_feature_frame,
    char_length,
    currency_symbol_count,
    digit_count,
    digit_ratio,
    exclamation_count,
    has_long_digit_run,
    has_url_like,
    mean_word_length,
    non_alnum_count,
    uppercase_ratio,
    word_count,
)
from src.utils.redaction import redact, redact_series, safe_sample

SPAMMY = "WINNER!! Claim your £900 prize now, call 09061701461 www.dbuk.net"
HAMMY = "Ok lar... Joking wif u oni"


class TestFeatureCount:
    def test_exactly_twelve_features(self) -> None:
        assert len(FEATURE_ORDER) == 12

    def test_names_are_unique(self) -> None:
        assert len(set(FEATURE_ORDER)) == 12


class TestIndividualFeatures:
    def test_char_length(self) -> None:
        assert char_length("hello world") == 11
        assert char_length("") == 0

    def test_word_count(self) -> None:
        assert word_count("hello   big  world") == 3
        assert word_count("   ") == 0

    def test_mean_word_length(self) -> None:
        assert mean_word_length("ab abcd") == 3.0
        assert mean_word_length("") == 0.0  # 0.0, not NaN

    def test_digit_count(self) -> None:
        assert digit_count("a1b22c333") == 6
        assert digit_count("none") == 0

    def test_digit_ratio(self) -> None:
        assert digit_ratio("ab12") == 0.5
        assert digit_ratio("") == 0.0

    def test_digit_ratio_differs_from_digit_count(self) -> None:
        """The two carry different information -- that is why both are kept."""
        short = "07123456789"
        long = "hello there this is a long friendly message 07123456789 ok"
        assert digit_count(short) == digit_count(long)
        assert digit_ratio(short) > digit_ratio(long)

    def test_uppercase_ratio_is_over_letters_not_length(self) -> None:
        # 4 letters, all upper; the digits must not dilute the ratio.
        assert uppercase_ratio("FREE1234567890") == 1.0
        assert uppercase_ratio("Free") == 0.25
        assert uppercase_ratio("1234") == 0.0

    def test_all_caps_word_count(self) -> None:
        assert all_caps_word_count("WIN a FREE prize") == 2
        assert all_caps_word_count("no shouting here") == 0

    def test_all_caps_ignores_single_letter_i(self) -> None:
        """'I' would otherwise fire on a large share of ordinary ham."""
        assert all_caps_word_count("I think I will go") == 0
        assert all_caps_word_count("OK I will") == 1

    def test_exclamation_count(self) -> None:
        assert exclamation_count("wow!! really!") == 3
        assert exclamation_count("calm") == 0

    def test_currency_symbol_count(self) -> None:
        assert currency_symbol_count("£100 and $50") == 2
        assert currency_symbol_count("100 pounds") == 0

    def test_currency_matches_symbols_not_codes(self) -> None:
        """Codes are words; word identity is exactly what this project avoids."""
        assert currency_symbol_count("GBP 100") == 0
        assert currency_symbol_count("£100") == 1

    def test_non_alnum_count_excludes_whitespace(self) -> None:
        assert non_alnum_count("a b!") == 1
        assert non_alnum_count("a, b. c!") == 3

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("visit http://x.com now", 1),
            ("go to www.dbuk.net", 1),
            ("WWW.MOVIETRIVIA.TV", 1),      # case-insensitive
            ("see example.com", 1),          # bare domain, no scheme
            ("no links here", 0),
            ("costs 3.50 total", 0),         # a decimal is not a domain
        ],
    )
    def test_has_url_like(self, text: str, expected: int) -> None:
        assert has_url_like(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("call 09061701461", 1),   # 11 digits
            ("call 12345678", 1),      # exactly 8
            ("call 1234567", 0),       # 7, below threshold
            ("on 12/05/2024 at 3pm", 0),  # a date is not a phone number
            ("no digits", 0),
        ],
    )
    def test_has_long_digit_run(self, text: str, expected: int) -> None:
        assert has_long_digit_run(text) == expected


class TestRobustness:
    @pytest.mark.parametrize(
        "value", [None, np.nan, float("nan"), 12345, ["not", "a", "string"]]
    )
    def test_every_feature_survives_bad_input(self, value: object) -> None:
        """One malformed row must not fail the whole column."""
        from src.features.spam_features import FEATURE_FUNCTIONS

        for name, func in FEATURE_FUNCTIONS.items():
            out = func(value)
            assert isinstance(out, (int, float)), name
            assert not pd.isna(out), name

    def test_no_feature_returns_nan_on_empty_string(self) -> None:
        from src.features.spam_features import FEATURE_FUNCTIONS

        for name, func in FEATURE_FUNCTIONS.items():
            assert func("") == 0, name


class TestBuildFeatureFrame:
    @pytest.fixture
    def frame(self) -> pd.DataFrame:
        return build_feature_frame(pd.Series([SPAMMY, HAMMY, "", None]))

    def test_shape_and_column_order(self, frame: pd.DataFrame) -> None:
        assert frame.shape == (4, 12)
        assert list(frame.columns) == list(FEATURE_ORDER)

    def test_all_numeric_and_finite(self, frame: pd.DataFrame) -> None:
        assert frame.dtypes.eq(float).all()
        assert np.isfinite(frame.to_numpy()).all()

    def test_index_is_preserved_for_concatenation(self) -> None:
        messages = pd.Series([SPAMMY, HAMMY], index=[17, 42])
        assert list(build_feature_frame(messages).index) == [17, 42]

    def test_spam_scores_higher_on_the_shape_features(
        self, frame: pd.DataFrame
    ) -> None:
        spam, ham = frame.iloc[0], frame.iloc[1]
        assert spam["all_caps_word_count"] > ham["all_caps_word_count"]
        assert spam["currency_symbol_count"] > ham["currency_symbol_count"]
        assert spam["has_url_like"] > ham["has_url_like"]
        assert spam["has_long_digit_run"] > ham["has_long_digit_run"]

    def test_rejects_a_dataframe(self) -> None:
        with pytest.raises(TypeError, match="Series"):
            build_feature_frame(pd.DataFrame({"message": [SPAMMY]}))


class TestRedaction:
    def test_phone_numbers_are_masked(self) -> None:
        assert "09061701461" not in redact("call 09061701461 now", max_chars=200)
        assert "<PHONE>" in redact("call 09061701461 now", max_chars=200)

    def test_urls_and_emails_are_masked(self) -> None:
        out = redact("mail me at a.b@c.co.uk or www.dbuk.net", max_chars=200)
        assert "<EMAIL>" in out and "<URL>" in out
        assert "a.b@c.co.uk" not in out and "dbuk" not in out

    def test_money_and_bare_numbers_are_masked(self) -> None:
        out = redact("won £900 of 12 prizes", max_chars=200)
        assert "<MONEY>" in out and "<NUM>" in out
        assert "900" not in out

    def test_names_are_masked_but_common_words_are_not(self) -> None:
        out = redact("Hey Priyanka are you coming Tomorrow", max_chars=200)
        assert "Priyanka" not in out
        assert "<NAME>" in out
        assert "Hey" in out and "Tomorrow" in out

    def test_truncation_applies_after_redaction(self) -> None:
        """Truncating first could leave half a phone number in the output."""
        out = redact("x" * 20 + " 09061701461 " + "y" * 200, max_chars=40)
        assert len(out) <= 40
        assert "0906" not in out

    def test_nan_and_none_become_empty(self) -> None:
        assert redact(None) == ""
        assert redact(np.nan) == ""

    def test_redact_series_leaves_the_input_untouched(self) -> None:
        original = pd.Series(["call 09061701461"])
        out = redact_series(original)
        assert original.iloc[0] == "call 09061701461"
        assert "<PHONE>" in out.iloc[0]

    def test_safe_sample_redacts_and_does_not_mutate(self) -> None:
        frame = pd.DataFrame(
            {
                "label": ["spam", "ham"] * 10,
                "message": ["Win £900 call 09061701461", "ok see you"] * 10,
            }
        )
        out = safe_sample(frame, n=4, random_state=0)
        assert len(out) == 4
        assert not out["message"].str.contains("09061701461").any()
        assert frame["message"].str.contains("09061701461").any()

    def test_safe_sample_can_balance_by_group(self) -> None:
        frame = pd.DataFrame(
            {
                "label": ["spam"] * 5 + ["ham"] * 30,
                "message": ["Win £900 now"] * 5 + ["ok"] * 30,
            }
        )
        out = safe_sample(frame, n=3, group_by="label", random_state=0)
        assert out["label"].value_counts().to_dict() == {"spam": 3, "ham": 3}

    def test_no_digit_run_survives_redaction(self) -> None:
        """The property that actually matters for a committed notebook."""
        import re

        samples = [
            "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. "
            "Text FA to 87121 to receive entry question(std txt rate)",
            "URGENT! You have won a 1 week FREE membership in our £100,000 Prize "
            "Jackpot! Txt the word: CLAIM to No: 81010",
            "Call me on 07734396839, its Priya",
        ]
        for sample in samples:
            out = redact(sample, max_chars=500)
            assert not re.search(r"\d{3,}", out), out
