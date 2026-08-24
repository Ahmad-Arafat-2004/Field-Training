"""Shared fixtures.

`pythonpath = ["."]` in pyproject.toml puts the repo root on sys.path, so
`from src...` works without installing the package.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20240824)


@pytest.fixture
def mixed_frame(rng: np.random.Generator) -> pd.DataFrame:
    """Numeric + categorical, with missing values in both kinds of column."""
    n = 300
    frame = pd.DataFrame(
        {
            "length": rng.normal(100.0, 25.0, n),
            "ratio": rng.beta(2.0, 5.0, n),
            "count": rng.poisson(4.0, n).astype(float),
            "colour": rng.choice(["red", "green", "blue"], n),
            "size": rng.choice(["S", "M", "L"], n),
        }
    )
    frame.loc[5:15, "length"] = np.nan
    frame.loc[20:25, "colour"] = np.nan
    return frame


@pytest.fixture
def binary_target(rng: np.random.Generator) -> pd.Series:
    """Imbalanced target, roughly 13% positive -- like the spam corpus."""
    return pd.Series(
        (rng.random(300) < 0.13).astype(int), name="label"
    )


@pytest.fixture
def latin1_csv(tmp_path: Path) -> Path:
    """A CSV that is valid Latin-1 and invalid UTF-8.

    0xA3 (GBP) and 0xE9 (e-acute) are single bytes that no valid UTF-8
    sequence starts with in this position, so a strict UTF-8 read must fail.
    """
    path = tmp_path / "latin1.csv"
    rows = [
        b"label,message",
        b"spam,Win \xa31000 cash now",
        b"ham,Caf\xe9 at 6 then?",
        b"ham,Se\xf1or Garc\xeda says hi",
    ]
    path.write_bytes(b"\n".join(rows) + b"\n")
    return path


@pytest.fixture
def utf8_csv(tmp_path: Path) -> Path:
    path = tmp_path / "utf8.csv"
    path.write_text(
        "label,message\nspam,Win £1000 cash now\nham,Café at 6 then?\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def sample_log() -> Path:
    return FIXTURE_DIR / "sample.log"
