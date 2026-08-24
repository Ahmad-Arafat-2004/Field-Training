"""Cleaning for the used-car listings.

Four columns arrive as unit-suffixed strings -- ``"23.4 kmpl"``, ``"1248 CC"``,
``"74 bhp"``, ``"190Nm@ 2000rpm"`` -- so pandas types them as ``object`` and
the preprocessing template would one-hot them into thousands of columns
instead of scaling four numbers. This module turns them into floats.

This is dataset-specific cleaning, kept out of the template on purpose: the
template's job is the generic impute/encode/scale contract, and baking a
``kmpl`` parser into it would make it a used-car tool.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

__all__ = [
    "CAR_TARGET",
    "clean_car_frame",
    "parse_measurement",
    "parse_torque_nm",
    "parse_owner_rank",
]

CAR_TARGET = "selling_price"

_LEADING_NUMBER_RE = re.compile(r"^\s*([-+]?\d*\.?\d+)")
_TORQUE_RE = re.compile(r"([-+]?\d*\.?\d+)\s*(nm|kgm)?", re.IGNORECASE)

# 1 kgm = 9.80665 Nm. About a third of the torque column is quoted in kgm, and
# leaving both units in one column would put a 12.7 next to a 190 meaning
# roughly the same thing.
_KGM_TO_NM = 9.80665

# Ordinal, not categorical: "First Owner" < "Second Owner" < ... is a real
# ordering that a one-hot encoding would throw away.
_OWNER_RANKS: dict[str, float] = {
    "test drive car": 0.0,
    "first owner": 1.0,
    "second owner": 2.0,
    "third owner": 3.0,
    "fourth & above owner": 4.0,
}


def parse_measurement(value: object) -> float:
    """Pull the leading number out of a unit-suffixed string.

    ``"23.4 kmpl"`` -> 23.4, ``"1248 CC"`` -> 1248.0, ``"74 bhp"`` -> 74.0.
    Blanks and unparseable entries become NaN, which is what the imputer is
    for -- roughly 220 of the 8,128 rows are genuinely missing these.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    match = _LEADING_NUMBER_RE.match(str(value))
    return float(match.group(1)) if match else np.nan


def parse_torque_nm(value: object) -> float:
    """Normalise the torque column to Newton-metres.

    The column mixes formats: ``"190Nm@ 2000rpm"``, ``"12.7@ 2,700(kgm@ rpm)"``,
    ``"250Nm at 1500-2500rpm"``. Only the torque magnitude is taken; the rpm
    band it is measured at is a separate quantity and is dropped rather than
    silently averaged in.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan

    match = _TORQUE_RE.search(text)
    if match is None:
        return np.nan
    magnitude = float(match.group(1))

    # The unit can appear after the rpm figure -- "12.7@ 2,700(kgm@ rpm)" --
    # so the whole string is checked, not just the span next to the number.
    lowered = text.lower()
    if "kgm" in lowered and "nm" not in lowered:
        return magnitude * _KGM_TO_NM
    # A bare number under 50 is almost certainly kgm: petrol/diesel cars in
    # this dataset run 60-800 Nm, and 4-60 kgm.
    if not match.group(2) and magnitude < 50:
        return magnitude * _KGM_TO_NM
    return magnitude


def parse_owner_rank(value: object) -> float:
    """Map the ownership history to an ordinal rank. Unknown labels -> NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    return _OWNER_RANKS.get(str(value).strip().lower(), np.nan)


def clean_car_frame(
    frame: pd.DataFrame, *, reference_year: int = 2021
) -> pd.DataFrame:
    """Return a modelling-ready copy of the raw listings frame.

    Steps, in order:

    1. Parse ``mileage``, ``engine`` and ``max_power`` to floats.
    2. Normalise ``torque`` to Nm.
    3. Turn ``owner`` into an ordinal rank.
    4. Derive ``age`` from ``year``.
    5. Reduce ``name`` to its make (first token).
    6. Drop the raw string columns that have been replaced.

    Parameters
    ----------
    reference_year:
        Year the listings were collected, used for ``age``. The corpus was
        scraped in 2020-2021 and its newest listings are 2020, so 2021 keeps
        age non-negative.
    """
    out = frame.copy()

    for column in ("mileage", "engine", "max_power"):
        if column in out.columns:
            out[column] = out[column].map(parse_measurement)

    if "torque" in out.columns:
        out["torque"] = out["torque"].map(parse_torque_nm)

    if "owner" in out.columns:
        out["owner_rank"] = out["owner"].map(parse_owner_rank)
        out = out.drop(columns=["owner"])

    if "year" in out.columns:
        out["age"] = reference_year - out["year"]
        out = out.drop(columns=["year"])

    if "name" in out.columns:
        # 2,058 distinct full names would one-hot into a matrix wider than the
        # dataset is tall. The make (first token) is ~32 levels and carries
        # most of the brand signal; trim level is largely recoverable from
        # engine size and max power, which are already columns.
        out["make"] = out["name"].astype(str).str.split().str[0].str.title()
        out = out.drop(columns=["name"])

    return out
