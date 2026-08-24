"""Twelve shape-of-message features, engineered from raw text.

Each feature is a standalone function taking one message and returning one
number, applied column-wise by :func:`build_feature_frame`. They are written
with Python string operations and no vectoriser: the point of this project is
that the classifier never sees word identity, only the shape of the message.
What that buys and what it costs is written up in the notebook's LIMITATIONS
section.

Every function tolerates NaN and non-string input by treating it as an empty
message, so one malformed row cannot fail the whole column.
"""

from __future__ import annotations

import re
from typing import Callable

import pandas as pd

__all__ = [
    "FEATURE_FUNCTIONS",
    "FEATURE_ORDER",
    "build_feature_frame",
    "char_length",
    "word_count",
    "mean_word_length",
    "digit_count",
    "digit_ratio",
    "uppercase_ratio",
    "all_caps_word_count",
    "exclamation_count",
    "currency_symbol_count",
    "non_alnum_count",
    "has_url_like",
    "has_long_digit_run",
]

# Symbols, not currency codes: "GBP"/"USD" are words and would be caught by a
# word-identity model, which is exactly what this feature set is avoiding.
CURRENCY_SYMBOLS: frozenset[str] = frozenset("$£€¥₹¢₩₽₪₦฿")

# Deliberately loose. Spam in this corpus writes "www.dbuk.net" and
# "WWW.MOVIETRIVIA.TV" far more often than a well-formed http:// URL, so an
# anchored URL regex would miss most of the positives.
_URL_RE = re.compile(
    r"(https?://|www\.|\b[\w-]+\.(?:com|net|org|co\.uk|uk|tv|info|biz|ru|de)\b)",
    re.IGNORECASE,
)

# 8 is the shortest run that reliably indicates a UK phone number or shortcode
# in this corpus. Lower thresholds start matching prices, dates and PIN codes;
# the sensitivity of this choice is checked in the notebook.
_LONG_DIGIT_RUN_RE = re.compile(r"\d{8,}")


def _as_text(message: object) -> str:
    """Coerce anything to a string, mapping NaN/None to the empty message."""
    if message is None or (isinstance(message, float) and pd.isna(message)):
        return ""
    return message if isinstance(message, str) else str(message)


# ---------------------------------------------------------------- 1 -- 3 ---
def char_length(message: object) -> int:
    """Total characters, whitespace included."""
    return len(_as_text(message))


def word_count(message: object) -> int:
    """Whitespace-delimited tokens."""
    return len(_as_text(message).split())


def mean_word_length(message: object) -> float:
    """Mean token length. 0.0 for an empty message rather than NaN.

    Returning 0.0 keeps the column numeric and avoids handing the imputer a
    missing value that never came from missing data.
    """
    words = _as_text(message).split()
    if not words:
        return 0.0
    return sum(len(word) for word in words) / len(words)


# ---------------------------------------------------------------- 4 -- 6 ---
def digit_count(message: object) -> int:
    """Digit characters. Shortcodes and prize amounts push this up."""
    return sum(char.isdigit() for char in _as_text(message))


def digit_ratio(message: object) -> float:
    """Digits as a share of all characters.

    Carries different information from the raw count: a short message that is
    mostly a phone number scores high here and low on `digit_count`.
    """
    text = _as_text(message)
    if not text:
        return 0.0
    return sum(char.isdigit() for char in text) / len(text)


def uppercase_ratio(message: object) -> float:
    """Uppercase letters as a share of *letters*, not of all characters.

    Dividing by letters rather than length keeps the feature comparable across
    messages with different amounts of punctuation and digits -- otherwise a
    message padded with numbers would look less shouty than it is.
    """
    letters = [char for char in _as_text(message) if char.isalpha()]
    if not letters:
        return 0.0
    return sum(char.isupper() for char in letters) / len(letters)


# ---------------------------------------------------------------- 7 -- 9 ---
def all_caps_word_count(message: object) -> int:
    """Fully-capitalised words of 2+ letters ("FREE", "URGENT", "WINNER").

    The 2-letter floor drops the pronoun "I", which otherwise fires on a large
    share of perfectly ordinary ham.
    """
    count = 0
    for word in _as_text(message).split():
        letters = [char for char in word if char.isalpha()]
        if len(letters) >= 2 and all(char.isupper() for char in letters):
            count += 1
    return count


def exclamation_count(message: object) -> int:
    """Exclamation marks."""
    return _as_text(message).count("!")


def currency_symbol_count(message: object) -> int:
    """Occurrences of a currency symbol. See CURRENCY_SYMBOLS."""
    return sum(char in CURRENCY_SYMBOLS for char in _as_text(message))


# --------------------------------------------------------------- 10 -- 12 ---
def non_alnum_count(message: object) -> int:
    """Characters that are neither alphanumeric nor whitespace.

    Punctuation density: spam here leans on '!', '*', '&' and bare URLs.
    """
    return sum(
        (not char.isalnum()) and (not char.isspace())
        for char in _as_text(message)
    )


def has_url_like(message: object) -> int:
    """1 if the message contains something URL-shaped, else 0."""
    return int(bool(_URL_RE.search(_as_text(message))))


def has_long_digit_run(message: object) -> int:
    """1 if a run of 8+ consecutive digits appears -- a phone number or shortcode."""
    return int(bool(_LONG_DIGIT_RUN_RE.search(_as_text(message))))


# ---------------------------------------------------------------------------
# Registry. Order is fixed so that feature-importance output lines up with the
# columns across runs.
FEATURE_FUNCTIONS: dict[str, Callable[[object], float]] = {
    "char_length": char_length,
    "word_count": word_count,
    "mean_word_length": mean_word_length,
    "digit_count": digit_count,
    "digit_ratio": digit_ratio,
    "uppercase_ratio": uppercase_ratio,
    "all_caps_word_count": all_caps_word_count,
    "exclamation_count": exclamation_count,
    "currency_symbol_count": currency_symbol_count,
    "non_alnum_count": non_alnum_count,
    "has_url_like": has_url_like,
    "has_long_digit_run": has_long_digit_run,
}

FEATURE_ORDER: tuple[str, ...] = tuple(FEATURE_FUNCTIONS)

assert len(FEATURE_ORDER) == 12, "The brief specifies exactly 12 features."


def build_feature_frame(messages: pd.Series) -> pd.DataFrame:
    """Apply every feature function column-wise to `messages`.

    Returns a DataFrame with one column per feature, in FEATURE_ORDER, indexed
    like the input so it can be concatenated back onto the source frame.
    """
    if not isinstance(messages, pd.Series):
        raise TypeError(
            f"Expected a Series of messages, got {type(messages).__name__}."
        )
    return pd.DataFrame(
        {name: messages.apply(func) for name, func in FEATURE_FUNCTIONS.items()},
        index=messages.index,
    ).astype(float)
