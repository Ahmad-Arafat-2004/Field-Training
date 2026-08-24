"""CLI log parser: line count, level counts, and the top recurring error messages.

Usage
-----
    python -m src.utils.log_parser path/to/app.log
    python -m src.utils.log_parser app.log --top 20 --encoding utf-8
    python -m src.utils.log_parser app.log --json

Encoding is always passed explicitly at ``open()``. On Windows the platform
default is still a legacy ANSI codepage in many setups, so an unqualified
``open()`` decodes the same file differently depending on where it runs -- and
fails loudly on the first non-ASCII byte in a stack trace. The default here is
the same detection ladder the CSV loader uses.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# Support both `python -m src.utils.log_parser` and `python src/utils/log_parser.py`.
if __package__ in (None, ""):  # pragma: no cover - exercised only as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.loader import DEFAULT_ENCODINGS, read_text_safe  # noqa: E402

# Levels ordered by severity so the summary reads top-down rather than
# alphabetically (CRITICAL before DEBUG).
KNOWN_LEVELS: tuple[str, ...] = (
    "CRITICAL",
    "FATAL",
    "ERROR",
    "WARNING",
    "WARN",
    "INFO",
    "DEBUG",
    "TRACE",
    "NOTSET",
)
ERROR_LEVELS: frozenset[str] = frozenset({"CRITICAL", "FATAL", "ERROR"})

# Word-boundary match so "ERROR" inside a payload like "/api/errors" or a
# message such as "no ERRORS found" is not counted as a level token.
_LEVEL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(KNOWN_LEVELS) + r")(?![A-Za-z0-9_])"
)

# A dotted logger/component name sitting between the level and the message
# ("app.service.db - Connection timeout ..."). It is dropped before grouping:
# otherwise one failure mode reported by three components counts as three
# separate errors, and the real top error never surfaces.
_LOGGER_PREFIX_RE = re.compile(
    r"^[A-Za-z_][\w.$-]*(?:\.[\w.$-]+)+\s*[-:|]\s+"
)

# Volatile substrings are masked before grouping error messages. Without this,
# "timeout connecting to 10.0.0.4:5432" and "...to 10.0.0.9:5432" look like two
# distinct errors and neither reaches the top of the list.
_NORMALISERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b"), "<TS>"),
    (re.compile(r"\b(?:0x)?[0-9a-fA-F]{8,}\b"), "<HEX>"),
    (re.compile(r"\b\d+(?:\.\d+){3}\b"), "<IP>"),
    # Identifiers with a glued numeric tail ("user u12", "job-4"). Handled
    # before the bare-number rule so the alphabetic stem survives.
    (re.compile(r"\b([A-Za-z_]+)\d+\b"), r"\1<N>"),
    # Bare numbers, including ones glued to a unit ("30000ms" -> "<N>ms").
    # A lookbehind rather than \b so the unit suffix does not block the match.
    (re.compile(r"(?<![A-Za-z_0-9])\d+(?:\.\d+)?"), "<N>"),
    (re.compile(r"[\'\"][^\'\"]{0,80}[\'\"]"), "<STR>"),
    (re.compile(r"\s+"), " "),
)


@dataclass
class LogSummary:
    """Aggregate statistics for one log file."""

    path: str
    encoding: str
    total_lines: int = 0
    blank_lines: int = 0
    unlevelled_lines: int = 0
    level_counts: dict[str, int] = field(default_factory=dict)
    top_errors: list[tuple[str, int, str]] = field(default_factory=list)

    @property
    def error_lines(self) -> int:
        return sum(n for lvl, n in self.level_counts.items() if lvl in ERROR_LEVELS)


def normalise_message(message: str) -> str:
    """Collapse volatile parts of a message so repeats group together."""
    out = message.strip()
    for pattern, replacement in _NORMALISERS:
        out = pattern.sub(replacement, out)
    return out.strip()


def extract_level(line: str) -> str | None:
    """Return the first recognised level token in `line`, or None."""
    match = _LEVEL_RE.search(line)
    if match is None:
        return None
    level = match.group(1)
    # Treat the common aliases as their canonical form so counts do not split.
    return {"WARN": "WARNING", "FATAL": "CRITICAL"}.get(level, level)


def strip_prefix(line: str, level: str) -> str:
    """Drop everything up to and including the level token, plus separators.

    Timestamps and logger names live before the level in every common layout,
    and they are exactly the parts that differ between two occurrences of the
    same error.
    """
    idx = line.find(level)
    tail = line[idx + len(level):] if idx >= 0 else line
    tail = tail.lstrip(" \t:|-]}>")
    return _LOGGER_PREFIX_RE.sub("", tail, count=1)


def parse_lines(lines: Iterable[str], top: int = 10) -> tuple[
    int, int, int, Counter, list[tuple[str, int, str]]
]:
    """Core aggregation. Separated from I/O so it is directly testable."""
    total = 0
    blank = 0
    unlevelled = 0
    level_counts: Counter = Counter()
    error_groups: Counter = Counter()
    # Keep one verbatim example per normalised group -- the masked form alone is
    # hard to read when you are actually trying to debug something.
    examples: dict[str, str] = {}

    for line in lines:
        total += 1
        stripped = line.strip()
        if not stripped:
            blank += 1
            continue

        level = extract_level(stripped)
        if level is None:
            unlevelled += 1
            continue

        level_counts[level] += 1
        if level in ERROR_LEVELS:
            message = strip_prefix(stripped, level) or stripped
            key = normalise_message(message)
            if not key:
                continue
            error_groups[key] += 1
            examples.setdefault(key, message.strip())

    top_errors = [
        (key, count, examples.get(key, key))
        for key, count in error_groups.most_common(top)
    ]
    return total, blank, unlevelled, level_counts, top_errors


def summarise_log(
    path: str | Path,
    *,
    top: int = 10,
    encoding: str | None = None,
) -> LogSummary:
    """Read `path` and build a :class:`LogSummary`.

    `encoding` is passed explicitly to the reader. When None, the loader's
    detection ladder (utf-8 -> ... -> latin-1) is used and the encoding that
    worked is recorded on the summary.
    """
    path = Path(path)
    encodings: Sequence[str] = (encoding,) if encoding else DEFAULT_ENCODINGS
    text, used = read_text_safe(path, encodings=encodings)

    total, blank, unlevelled, level_counts, top_errors = parse_lines(
        text.splitlines(), top=top
    )
    ordered = {
        lvl: level_counts[lvl]
        for lvl in KNOWN_LEVELS
        if lvl in level_counts
    }
    # Anything matched but not in the canonical ordering (shouldn't happen, but
    # keeps the counts honest if KNOWN_LEVELS changes).
    ordered.update({k: v for k, v in level_counts.items() if k not in ordered})

    return LogSummary(
        path=str(path),
        encoding=used,
        total_lines=total,
        blank_lines=blank,
        unlevelled_lines=unlevelled,
        level_counts=ordered,
        top_errors=top_errors,
    )


def render(summary: LogSummary, *, width: int = 100) -> str:
    """Human-readable report."""
    lines: list[str] = []
    rule = "=" * width
    lines.append(rule)
    lines.append(f"LOG SUMMARY  {summary.path}")
    lines.append(rule)
    lines.append(f"  encoding used      : {summary.encoding}")
    lines.append(f"  total lines        : {summary.total_lines:,}")
    lines.append(f"  blank lines        : {summary.blank_lines:,}")
    lines.append(f"  lines w/o a level  : {summary.unlevelled_lines:,}")
    lines.append("")

    lines.append("LEVEL COUNTS")
    if not summary.level_counts:
        lines.append("  (no recognised level tokens found)")
    else:
        levelled = sum(summary.level_counts.values())
        widest = max(len(k) for k in summary.level_counts)
        for level, count in summary.level_counts.items():
            share = 100 * count / levelled
            bar = "#" * max(0, int(round(share / 2.5)))
            lines.append(
                f"  {level:<{widest}}  {count:>7,}  {share:>5.1f}%  {bar}"
            )
    lines.append("")

    lines.append(f"TOP ERROR MESSAGES  ({summary.error_lines:,} error-level lines)")
    if not summary.top_errors:
        lines.append("  (none)")
    else:
        for rank, (_key, count, example) in enumerate(summary.top_errors, start=1):
            snippet = example if len(example) <= width - 12 else example[: width - 15] + "..."
            lines.append(f"  {rank:>2}. [{count:>5,}x]  {snippet}")
    lines.append(rule)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="log_parser",
        description=(
            "Summarise a log file: line count, counts per level, and the most "
            "frequent error messages (grouped after masking timestamps, IDs "
            "and numbers)."
        ),
    )
    parser.add_argument("logfile", type=Path, help="Path to the log file.")
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many distinct error messages to show (default: 10).",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help=(
            "Force a specific encoding. Omit to try "
            f"{' -> '.join(DEFAULT_ENCODINGS)} in order. Never falls back to "
            "the platform default."
        ),
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit the summary as JSON instead of a formatted report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top < 1:
        print("--top must be >= 1", file=sys.stderr)
        return 2
    try:
        summary = summarise_log(
            args.logfile, top=args.top, encoding=args.encoding
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except UnicodeError as exc:
        print(f"error: could not decode {args.logfile}: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        payload = asdict(summary)
        payload["error_lines"] = summary.error_lines
        payload["top_errors"] = [
            {"pattern": k, "count": c, "example": ex}
            for k, c, ex in summary.top_errors
        ]
        print(json.dumps(payload, indent=2))
    else:
        print(render(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
