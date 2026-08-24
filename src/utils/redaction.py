"""Redaction for sample rows printed into a committed notebook.

The SMS Spam Collection is real traffic. It contains working phone numbers,
shortcodes, email addresses, URLs and first names, and a notebook that prints
``df.head()`` copies all of that into a file that gets committed and shared.
Notebook outputs are stored in the ``.ipynb``, so this survives long after the
cell is re-run.

The functions here replace the identifying spans with type placeholders and
truncate what remains, which keeps sample rows readable for the purpose they
actually serve -- showing the *shape* of a message -- without republishing the
contents. That shape is all the features in this project use anyway.

This is not a general-purpose de-identifier: it is a display filter for
exploratory output. The underlying frame is never modified.
"""

from __future__ import annotations

import re

import pandas as pd

__all__ = ["redact", "redact_series", "safe_sample"]

# Ordered: the most specific patterns run first so an email address is not
# half-consumed by the URL rule, and a phone number inside a URL stays part of
# the URL.
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<EMAIL>"),
    (re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE), "<URL>"),
    # 5+ digits, allowing internal spaces/dashes: covers UK mobiles, premium
    # numbers and the 5-digit shortcodes this corpus is full of.
    (re.compile(r"\+?\d[\d\s-]{4,}\d"), "<PHONE>"),
    (re.compile(r"[£$€¥₹]\s?\d+(?:[.,]\d+)?"), "<MONEY>"),
    (re.compile(r"\b\d+(?:[.,]\d+)?\b"), "<NUM>"),
)

# Capitalised words are the main name-leak route. An allowlist of common
# sentence-initial and proper-noun-but-not-personal words keeps the output
# readable instead of reducing every message to a row of placeholders.
_NOT_NAMES: frozenset[str] = frozenset(
    """
    I I'm I'll I've A An The This That These Those There Here It Its
    You Your Yours We Our Us They Their He She Him Her His Hers My Mine Me
    And But Or So If Then Than When While Because As At By For From In Into
    Of On To With Without Up Down Out Off Over Under Again Just Now Not No
    Yes Ok Okay Hi Hey Hello Bye Thanks Thank Please Sorry Call Text Send
    Free Win Won Winner Urgent Claim Prize Cash Offer Reply Stop Mobile
    Phone Msg Message Txt Sms Congratulations Congrats Guaranteed Award
    Customer Service Network Contact Ltd Box Po Pobox Valid Only New Latest
    Do Don't Can Can't Will Won't Would Should Could Have Has Had Am Are Is
    Was Were Be Been Being Get Got Go Going Come Coming See Know Let Make
    Good Great Nice Love Like Want Need Time Day Today Tomorrow Tonight
    Monday Tuesday Wednesday Thursday Friday Saturday Sunday
    """.split()
)

_CAPITALISED_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")


def _mask_names(text: str) -> str:
    """Replace capitalised words that are not on the allowlist with <NAME>."""

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        return word if word in _NOT_NAMES else "<NAME>"

    return _CAPITALISED_RE.sub(replace, text)


def redact(
    message: object,
    *,
    max_chars: int = 70,
    mask_names: bool = True,
) -> str:
    """Redact identifying spans in `message` and truncate to `max_chars`.

    Parameters
    ----------
    message:
        The raw message. Non-strings and NaN become ``""``.
    max_chars:
        Truncation budget applied *after* redaction, so a long phone number
        cannot survive by being cut in half.
    mask_names:
        Replace capitalised non-allowlisted words with ``<NAME>``.

    Returns
    -------
    str
        A display-safe rendering, e.g.
        ``"<NAME> call <PHONE> to claim your <MONEY> prize now! ..."``.
    """
    if message is None or (isinstance(message, float) and pd.isna(message)):
        return ""
    text = message if isinstance(message, str) else str(message)

    for pattern, placeholder in _PATTERNS:
        text = pattern.sub(placeholder, text)
    if mask_names:
        text = _mask_names(text)

    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def redact_series(
    messages: pd.Series, *, max_chars: int = 70, mask_names: bool = True
) -> pd.Series:
    """Vectorised :func:`redact`. Returns a new Series; the input is untouched."""
    return messages.apply(
        lambda m: redact(m, max_chars=max_chars, mask_names=mask_names)
    )


def safe_sample(
    frame: pd.DataFrame,
    message_column: str = "message",
    *,
    n: int = 5,
    random_state: int = 42,
    max_chars: int = 70,
    group_by: str | None = None,
) -> pd.DataFrame:
    """A sample of `frame` with the message column redacted, ready to display.

    Use this anywhere a notebook would otherwise call ``.head()`` or
    ``.sample()`` on the raw corpus.

    Parameters
    ----------
    group_by:
        Take `n` rows from each level of this column (e.g. `n` spam and `n`
        ham) rather than `n` overall.
    """
    if message_column not in frame.columns:
        raise KeyError(f"No column {message_column!r} in frame.")

    if group_by is not None:
        # Explicit concat rather than groupby().apply(): apply's handling of the
        # grouping column changes across pandas 2.x and emits a deprecation
        # warning either way, and this keeps the group column in the output.
        pieces = [
            group.sample(min(n, len(group)), random_state=random_state)
            for _, group in frame.groupby(group_by, sort=True)
        ]
        sampled = pd.concat(pieces).reset_index(drop=True)
    else:
        sampled = frame.sample(min(n, len(frame)), random_state=random_state)

    out = sampled.copy()
    out[message_column] = redact_series(out[message_column], max_chars=max_chars)
    return out
