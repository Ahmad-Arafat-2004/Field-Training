"""Encoding-safe CSV loading.

The motivating case is the UCI SMS Spam Collection, which is Latin-1 encoded.
Reading it as UTF-8 raises ``UnicodeDecodeError``; reading it with
``errors="replace"`` succeeds but silently substitutes U+FFFD for every
non-ASCII byte, so the corruption only surfaces much later as odd feature
values. Neither is acceptable, so this module tries a ladder of encodings and
*reports which one worked* instead of guessing quietly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# Ordered most-strict first. UTF-8 is strict enough to reject most non-UTF-8
# byte sequences, which is what makes it a usable probe. Latin-1 is last on
# purpose: it maps all 256 byte values to characters and therefore *never*
# fails, so anything after it would be unreachable.
DEFAULT_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


@dataclass(frozen=True)
class LoadResult:
    """A loaded frame plus the provenance of how it was decoded."""

    frame: pd.DataFrame
    encoding: str
    path: Path
    attempted: tuple[str, ...]

    @property
    def used_fallback(self) -> bool:
        """True when the first candidate encoding did not work."""
        return self.encoding != self.attempted[0]

    def describe(self) -> str:
        rows, cols = self.frame.shape
        note = "" if not self.used_fallback else (
            f" (fell back from {', '.join(self.attempted[:-1])})"
        )
        return (
            f"Loaded {self.path.name}: {rows:,} rows x {cols} cols "
            f"using encoding '{self.encoding}'{note}"
        )


class EncodingDetectionError(RuntimeError):
    """Raised when every candidate encoding failed to decode the file."""


def read_csv_safe(
    path: str | Path,
    encodings: Sequence[str] = DEFAULT_ENCODINGS,
    *,
    verbose: bool = True,
    **read_csv_kwargs: Any,
) -> LoadResult:
    """Read a CSV, trying `encodings` in order, and report the one that worked.

    Parameters
    ----------
    path:
        CSV (or other delimited text) file to read.
    encodings:
        Candidate encodings, most-strict first. Defaults to
        ``("utf-8", "utf-8-sig", "cp1252", "latin-1")``.
    verbose:
        Print a one-line provenance summary. The result also carries the
        encoding programmatically, so this is only for humans.
    **read_csv_kwargs:
        Forwarded to :func:`pandas.read_csv` (``sep``, ``names``, ``header``,
        ``quoting``, ...). Passing ``encoding`` here is an error: it would
        defeat the whole point of this function.

    Returns
    -------
    LoadResult
        The frame plus the encoding actually used.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If `encodings` is empty, or `encoding` was passed via kwargs.
    EncodingDetectionError
        If every candidate encoding failed.
    """
    path = Path(path)
    if "encoding" in read_csv_kwargs:
        raise ValueError(
            "Do not pass `encoding=` to read_csv_safe -- it detects the "
            "encoding for you. Pass a one-element `encodings=(...)` sequence "
            "instead if you want to force one."
        )
    encodings = tuple(encodings)
    if not encodings:
        raise ValueError("`encodings` must contain at least one candidate.")
    if not path.is_file():
        raise FileNotFoundError(
            f"No such file: {path}. If this is a dataset, run "
            f"`python scripts/download_data.py` first."
        )

    failures: list[str] = []
    for candidate in encodings:
        try:
            # errors="strict" is pandas' default and is load-bearing here: a
            # lenient error handler would make every candidate "succeed".
            frame = pd.read_csv(path, encoding=candidate, **read_csv_kwargs)
        except (UnicodeDecodeError, UnicodeError) as exc:
            failures.append(f"{candidate}: {type(exc).__name__}: {exc}")
            logger.debug("Encoding %s failed for %s: %s", candidate, path, exc)
            continue

        result = LoadResult(
            frame=frame, encoding=candidate, path=path, attempted=encodings
        )
        if result.used_fallback:
            logger.warning(
                "%s is not %s; decoded as %s instead.",
                path.name,
                encodings[0],
                candidate,
            )
        if verbose:
            print(result.describe())
        return result

    raise EncodingDetectionError(
        f"Could not decode {path} with any of {encodings}.\n"
        + "\n".join(f"  - {f}" for f in failures)
    )


def read_text_safe(
    path: str | Path,
    encodings: Sequence[str] = DEFAULT_ENCODINGS,
    *,
    verbose: bool = False,
) -> tuple[str, str]:
    """Read a whole text file with the same encoding ladder.

    Returns ``(text, encoding_used)``. Used by the log parser so that it, too,
    never depends on the platform default encoding.
    """
    path = Path(path)
    encodings = tuple(encodings)
    if not encodings:
        raise ValueError("`encodings` must contain at least one candidate.")
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")

    failures: list[str] = []
    for candidate in encodings:
        try:
            text = path.read_text(encoding=candidate)
        except (UnicodeDecodeError, UnicodeError) as exc:
            failures.append(f"{candidate}: {exc}")
            continue
        if verbose:
            print(f"Read {path.name} using encoding '{candidate}'")
        return text, candidate

    raise EncodingDetectionError(
        f"Could not decode {path} with any of {encodings}.\n"
        + "\n".join(f"  - {f}" for f in failures)
    )


def summarise_duplicates(
    frame: pd.DataFrame, subset: str | Iterable[str] | None = None
) -> str:
    """One-line report of exact-duplicate rows, for use before a train/test split.

    Duplicates that survive into the split can place identical rows on both
    sides of it, which inflates held-out scores without any real generalisation.
    """
    n_dupes = int(frame.duplicated(subset=subset).sum())
    scope = "rows" if subset is None else f"values of {subset}"
    pct = 100 * n_dupes / len(frame) if len(frame) else 0.0
    return f"{n_dupes:,} duplicate {scope} ({pct:.1f}% of {len(frame):,})"


__all__ = [
    "DEFAULT_ENCODINGS",
    "EncodingDetectionError",
    "LoadResult",
    "read_csv_safe",
    "read_text_safe",
    "summarise_duplicates",
]
