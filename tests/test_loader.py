"""Tests for the encoding-safe loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.utils.loader import (
    DEFAULT_ENCODINGS,
    EXTENDED_ENCODINGS,
    EncodingDetectionError,
    read_csv_safe,
    read_text_safe,
    summarise_duplicates,
)


class TestEncodingDetection:
    def test_reports_utf8_when_utf8_works(self, utf8_csv: Path) -> None:
        result = read_csv_safe(utf8_csv, verbose=False)
        assert result.encoding == "utf-8"
        assert result.attempted == DEFAULT_ENCODINGS
        assert not result.used_fallback

    def test_reports_the_fallback_when_utf8_fails(self, latin1_csv: Path) -> None:
        result = read_csv_safe(latin1_csv, verbose=False)
        assert result.encoding == "latin-1"
        assert result.used_fallback

    def test_describe_mentions_the_fallback(self, latin1_csv: Path) -> None:
        text = read_csv_safe(latin1_csv, verbose=False).describe()
        assert "latin-1" in text
        assert "fell back" in text

    def test_ascii_only_file_reads_as_utf8(self, tmp_path: Path) -> None:
        path = tmp_path / "plain.csv"
        path.write_text("a,b\n1,2\n", encoding="ascii")
        assert read_csv_safe(path, verbose=False).encoding == "utf-8"

    def test_characters_survive_the_round_trip(self, latin1_csv: Path) -> None:
        frame = read_csv_safe(latin1_csv, verbose=False).frame
        assert frame.loc[0, "message"] == "Win £1000 cash now"
        assert frame.loc[1, "message"] == "Café at 6 then?"
        assert frame.loc[2, "message"] == "Señor García says hi"

    def test_extended_ladder_prefers_cp1252(self, tmp_path: Path) -> None:
        """0x92 is a curly apostrophe in cp1252 and a control code in latin-1."""
        path = tmp_path / "win.csv"
        path.write_bytes(b"a\nit\x92s here\n")

        default = read_csv_safe(path, verbose=False)
        assert default.encoding == "latin-1"

        extended = read_csv_safe(path, encodings=EXTENDED_ENCODINGS, verbose=False)
        assert extended.encoding == "cp1252"
        assert extended.frame.iloc[0, 0] == "it’s here"


class TestLoaderContract:
    def test_missing_file_raises_with_a_useful_hint(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="download_data"):
            read_csv_safe(tmp_path / "nope.csv")

    def test_passing_encoding_kwarg_is_rejected(self, utf8_csv: Path) -> None:
        """Silently honouring it would disable the detection this exists for."""
        with pytest.raises(ValueError, match="Do not pass"):
            read_csv_safe(utf8_csv, encoding="utf-8")

    def test_empty_encoding_list_is_rejected(self, utf8_csv: Path) -> None:
        with pytest.raises(ValueError, match="at least one"):
            read_csv_safe(utf8_csv, encodings=())

    def test_all_candidates_failing_raises(self, latin1_csv: Path) -> None:
        with pytest.raises(EncodingDetectionError, match="Could not decode"):
            read_csv_safe(latin1_csv, encodings=("utf-8", "ascii"), verbose=False)

    def test_read_csv_kwargs_are_forwarded(self, tmp_path: Path) -> None:
        path = tmp_path / "tsv.txt"
        path.write_bytes(b"spam\tWin \xa31000\nham\thello\n")
        result = read_csv_safe(
            path, sep="\t", names=["label", "message"], header=None, verbose=False
        )
        assert list(result.frame.columns) == ["label", "message"]
        assert result.encoding == "latin-1"
        assert result.frame.loc[0, "message"] == "Win £1000"

    def test_verbose_prints_provenance(self, latin1_csv: Path, capsys) -> None:
        read_csv_safe(latin1_csv, verbose=True)
        assert "latin-1" in capsys.readouterr().out


class TestReadTextSafe:
    def test_latin1_text_file(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_bytes(b"co\xfbt total: 12\xa3\n")
        text, encoding = read_text_safe(path)
        assert encoding == "latin-1"
        assert "coût" in text

    def test_utf8_text_file(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("coût total\n", encoding="utf-8")
        text, encoding = read_text_safe(path)
        assert encoding == "utf-8"
        assert "coût" in text

    def test_forced_single_encoding(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_bytes(b"caf\xe9\n")
        with pytest.raises(EncodingDetectionError):
            read_text_safe(path, encodings=("utf-8",))


class TestSummariseDuplicates:
    def test_counts_full_row_duplicates(self) -> None:
        frame = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        assert "1 duplicate rows" in summarise_duplicates(frame)

    def test_counts_subset_duplicates(self) -> None:
        frame = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "z", "y"]})
        assert summarise_duplicates(frame, subset="a").startswith("1 duplicate")
        assert summarise_duplicates(frame).startswith("0 duplicate")

    def test_empty_frame_does_not_divide_by_zero(self) -> None:
        assert "0 duplicate" in summarise_duplicates(pd.DataFrame({"a": []}))
