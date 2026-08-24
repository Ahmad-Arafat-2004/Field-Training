"""Tests for the CLI log parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.loader import EncodingDetectionError
from src.utils.log_parser import (
    extract_level,
    main,
    normalise_message,
    parse_lines,
    render,
    strip_prefix,
    summarise_log,
)


class TestLevelExtraction:
    @pytest.mark.parametrize(
        "line,expected",
        [
            ("2026-08-24 12:00:00 [ERROR] boom", "ERROR"),
            ("2026-08-24 12:00:00 INFO started", "INFO"),
            ("WARN: retrying", "WARNING"),          # alias folded
            ("FATAL: out of memory", "CRITICAL"),   # alias folded
            ("    at com.example.Handler(Handler.java:1)", None),
            ("", None),
        ],
    )
    def test_extract_level(self, line: str, expected: str | None) -> None:
        assert extract_level(line) == expected

    def test_level_token_inside_a_word_is_not_a_level(self) -> None:
        """Without word boundaries, a URL path inflates the ERROR count."""
        assert extract_level("INFO GET /api/errors returned 200") == "INFO"
        assert extract_level("GET /v1/ERRORLOG") is None

    def test_aliases_do_not_split_the_counts(self) -> None:
        _, _, _, counts, _ = parse_lines(
            ["WARN a", "WARNING b", "FATAL c", "CRITICAL d"]
        )
        assert counts["WARNING"] == 2
        assert counts["CRITICAL"] == 2


class TestNormalisation:
    def test_volatile_parts_are_masked(self) -> None:
        assert normalise_message(
            "timeout to 10.0.0.4:5432 after 30000ms"
        ) == "timeout to <IP>:<N> after <N>ms"

    def test_two_occurrences_normalise_to_one_key(self) -> None:
        a = normalise_message("connect to db-4 at 10.0.0.4 failed")
        b = normalise_message("connect to db-91 at 10.0.0.9 failed")
        assert a == b

    def test_timestamps_and_hex_ids_are_masked(self) -> None:
        out = normalise_message(
            "job 2026-08-24T11:02:03,123 trace=9f3ab21c77de failed"
        )
        assert "<TS>" in out and "<HEX>" in out

    def test_quoted_payloads_are_masked(self) -> None:
        a = normalise_message("rejected file 'a.csv'")
        b = normalise_message("rejected file 'b.csv'")
        assert a == b == "rejected file <STR>"

    def test_logger_prefix_is_dropped(self) -> None:
        """One failure reported by three components is one error, not three."""
        assert strip_prefix(
            "12:00 [ERROR] app.service.db - disk full", "ERROR"
        ) == "disk full"
        assert strip_prefix(
            "12:00 [ERROR] app.service.api - disk full", "ERROR"
        ) == "disk full"


class TestParseLines:
    LINES = [
        "2026-08-24 10:00:00 [INFO] app.api - started",
        "2026-08-24 10:00:01 [ERROR] app.db - connect to 10.0.0.1 failed",
        "",
        "2026-08-24 10:00:02 [ERROR] app.api - connect to 10.0.0.7 failed",
        "    at Handler.java:9",
        "2026-08-24 10:00:03 [WARNING] app.api - slow",
        "2026-08-24 10:00:04 [CRITICAL] app.db - disk full",
    ]

    def test_counts(self) -> None:
        total, blank, unlevelled, counts, _ = parse_lines(self.LINES)
        assert total == 7
        assert blank == 1
        assert unlevelled == 1
        assert counts["ERROR"] == 2
        assert counts["CRITICAL"] == 1

    def test_identical_errors_group_into_one_entry(self) -> None:
        _, _, _, _, top = parse_lines(self.LINES)
        patterns = {key: count for key, count, _ in top}
        assert patterns["connect to <IP> failed"] == 2

    def test_top_is_ordered_by_frequency(self) -> None:
        _, _, _, _, top = parse_lines(self.LINES)
        counts = [count for _, count, _ in top]
        assert counts == sorted(counts, reverse=True)

    def test_top_limit_is_respected(self) -> None:
        # Distinguished by letters, not digits: numbers are masked to <N>, so
        # twenty numbered messages would correctly collapse into one group.
        lines = [
            f"[ERROR] failure kind {chr(97 + i)}{chr(97 + i)} here"
            for i in range(20)
        ]
        _, _, _, _, top = parse_lines(lines, top=3)
        assert len(top) == 3

    def test_each_group_keeps_a_verbatim_example(self) -> None:
        _, _, _, _, top = parse_lines(self.LINES)
        _, _, example = top[0]
        assert "10.0.0.1" in example  # not the masked form

    def test_empty_input(self) -> None:
        total, blank, unlevelled, counts, top = parse_lines([])
        assert (total, blank, unlevelled) == (0, 0, 0)
        assert not counts and not top


class TestSummariseLog:
    def test_reads_a_latin1_log_without_the_platform_default(
        self, sample_log: Path
    ) -> None:
        summary = summarise_log(sample_log, top=5)
        assert summary.encoding == "latin-1"
        assert summary.total_lines > 900
        assert summary.level_counts["ERROR"] > 0

    def test_non_ascii_message_survives(self, sample_log: Path) -> None:
        summary = summarise_log(sample_log, top=10)
        examples = " ".join(ex for _, _, ex in summary.top_errors)
        assert "financière" in examples
        assert "�" not in examples

    def test_error_lines_equals_grouped_totals(self, sample_log: Path) -> None:
        """Every error-level line must land in exactly one group."""
        summary = summarise_log(sample_log, top=100)
        assert sum(c for _, c, _ in summary.top_errors) == summary.error_lines

    def test_levels_are_ordered_by_severity_not_alphabetically(
        self, sample_log: Path
    ) -> None:
        levels = list(summarise_log(sample_log).level_counts)
        assert levels.index("CRITICAL") < levels.index("ERROR")
        assert levels.index("ERROR") < levels.index("INFO")

    def test_forced_encoding_is_honoured(self, tmp_path: Path) -> None:
        path = tmp_path / "u.log"
        path.write_text("[ERROR] café unavailable\n", encoding="utf-8")
        assert summarise_log(path, encoding="utf-8").encoding == "utf-8"

    def test_forced_wrong_encoding_fails_loudly(self, tmp_path: Path) -> None:
        """Better to fail than to decode into mojibake."""
        path = tmp_path / "l.log"
        path.write_bytes(b"[ERROR] caf\xe9 unavailable\n")
        with pytest.raises(EncodingDetectionError):
            summarise_log(path, encoding="utf-8")

    def test_cli_reports_a_decode_failure_cleanly(
        self, tmp_path: Path, capsys
    ) -> None:
        """A forced-wrong encoding should exit 1 with a message, not traceback."""
        path = tmp_path / "l.log"
        path.write_bytes(b"[ERROR] caf\xe9 unavailable\n")
        assert main([str(path), "--encoding", "utf-8"]) == 1
        assert "could not decode" in capsys.readouterr().err

    def test_render_produces_a_report(self, sample_log: Path) -> None:
        text = render(summarise_log(sample_log, top=3))
        assert "LOG SUMMARY" in text
        assert "LEVEL COUNTS" in text
        assert "TOP ERROR MESSAGES" in text


class TestCli:
    def test_exit_zero_and_prints_report(self, sample_log: Path, capsys) -> None:
        assert main([str(sample_log), "--top", "3"]) == 0
        assert "LEVEL COUNTS" in capsys.readouterr().out

    def test_json_output_is_valid_json(self, sample_log: Path, capsys) -> None:
        assert main([str(sample_log), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["encoding"] == "latin-1"
        assert payload["total_lines"] > 900
        assert isinstance(payload["top_errors"], list)
        assert set(payload["top_errors"][0]) == {"pattern", "count", "example"}

    def test_missing_file_exits_one(self, tmp_path: Path, capsys) -> None:
        assert main([str(tmp_path / "nope.log")]) == 1
        assert "error:" in capsys.readouterr().err

    def test_bad_top_exits_two(self, sample_log: Path, capsys) -> None:
        assert main([str(sample_log), "--top", "0"]) == 2
        assert "--top" in capsys.readouterr().err
