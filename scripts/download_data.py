"""Fetch the two datasets into the gitignored data/ directory.

    python scripts/download_data.py            # fetch anything missing
    python scripts/download_data.py --force    # re-fetch everything

data/ is not committed. The SMS corpus contains real phone numbers and names
from the original collection, and both files are reproducible from their
public sources, so the repo carries the recipe rather than the payload.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

SMS_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SMS_MEMBER = "SMSSpamCollection"
SMS_TARGET = DATA_DIR / "SMSSpamCollection.tsv"

CARS_URL = (
    "https://raw.githubusercontent.com/swetaswarupa/Car-Price-Prediction/"
    "main/Car%20details%20v3.csv"
)
CARS_TARGET = DATA_DIR / "car_details_v3.csv"

USER_AGENT = "training-portfolio/0.1 (dataset fetch for coursework)"


@dataclass
class Fetched:
    name: str
    path: Path
    n_bytes: int
    skipped: bool


def _get(url: str, timeout: int = 90) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_sms(force: bool = False) -> Fetched:
    """UCI SMS Spam Collection v.1: 5,574 messages, 747 spam / 4,827 ham.

    Ships as a zip containing a tab-separated file with no header, and the
    payload is Latin-1 encoded. It is written to disk verbatim -- decoding is
    the loader's job, and re-encoding here would hide the very case the
    encoding-safe loader exists to handle.
    """
    if SMS_TARGET.exists() and not force:
        return Fetched("SMS Spam Collection", SMS_TARGET,
                       SMS_TARGET.stat().st_size, skipped=True)

    payload = _get(SMS_ZIP_URL)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if SMS_MEMBER not in names:
            raise RuntimeError(
                f"'{SMS_MEMBER}' not found in the archive; it contains {names}."
            )
        raw = archive.read(SMS_MEMBER)

    SMS_TARGET.write_bytes(raw)
    return Fetched("SMS Spam Collection", SMS_TARGET, len(raw), skipped=False)


def fetch_cars(force: bool = False) -> Fetched:
    """CarDekho used-car listings: 8,128 rows, 13 columns.

    Mixed numeric and categorical with unit-suffixed strings ('23.4 kmpl',
    '1248 CC'), which is what makes it a fair exercise for the preprocessing
    template rather than a frame that is already clean.
    """
    if CARS_TARGET.exists() and not force:
        return Fetched("Used-car listings", CARS_TARGET,
                       CARS_TARGET.stat().st_size, skipped=True)

    raw = _get(CARS_URL)
    CARS_TARGET.write_bytes(raw)
    return Fetched("Used-car listings", CARS_TARGET, len(raw), skipped=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if present."
    )
    parser.add_argument(
        "--only",
        choices=("sms", "cars"),
        default=None,
        help="Fetch just one dataset.",
    )
    args = parser.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    jobs = {"sms": fetch_sms, "cars": fetch_cars}
    if args.only:
        jobs = {args.only: jobs[args.only]}

    failures = 0
    for key, job in jobs.items():
        try:
            result = job(force=args.force)
        except (urllib.error.URLError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
            print(f"  FAILED  {key}: {exc}", file=sys.stderr)
            failures += 1
            continue
        state = "already present" if result.skipped else "downloaded"
        print(
            f"  {state:>15}  {result.name}: {result.path.name} "
            f"({result.n_bytes:,} bytes)"
        )

    if failures:
        print(
            f"\n{failures} dataset(s) could not be fetched. "
            "Check the network, or download them manually into data/.",
            file=sys.stderr,
        )
        return 1
    print(f"\nData ready in {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
