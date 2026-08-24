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

# Candidate corpora examined in notebook 04 when considering a larger spam
# dataset. Not needed for notebooks 01-03; fetched only with `--only audit`
# because they total ~734 MB.
HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
AUDIT_SOURCES: dict[str, tuple[str, str, str]] = {
    "audit_big_300k.csv": (
        "locuoco/the-biggest-spam-ham-phish-email-dataset-300000",
        "df.csv",
        "365k emails, 3 classes -- the one notebook 04 rejects as contaminated",
    ),
    "audit_enron_spam.csv": (
        "SetFit/enron_spam",
        "enron_spam_data.csv",
        "33.7k Enron emails, balanced spam/ham",
    ),
    "audit_phishing.csv": (
        "zefang-liu/phishing-email-dataset",
        "Phishing_Email.csv",
        "18.6k emails labelled safe/phishing",
    ),
}

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


def fetch_audit(force: bool = False) -> list[Fetched]:
    """The candidate corpora compared in notebook 04.

    Optional: notebooks 01-03 do not touch these, and they total ~734 MB.
    Notebook 04 skips any that are absent rather than failing.
    """
    results: list[Fetched] = []
    for filename, (repo, path, description) in AUDIT_SOURCES.items():
        target = DATA_DIR / filename
        if target.exists() and not force:
            results.append(
                Fetched(description, target, target.stat().st_size, skipped=True)
            )
            continue
        raw = _get(HF.format(repo=repo, path=path), timeout=600)
        target.write_bytes(raw)
        results.append(Fetched(description, target, len(raw), skipped=False))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if present."
    )
    parser.add_argument(
        "--only",
        choices=("sms", "cars", "audit"),
        default=None,
        help=(
            "Fetch just one group. 'audit' pulls the ~734 MB of candidate "
            "corpora that notebook 04 compares; it is not needed for "
            "notebooks 01-03."
        ),
    )
    args = parser.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # The audit corpora are large and optional, so they are excluded from a
    # bare run and fetched only when asked for by name.
    jobs = {"sms": fetch_sms, "cars": fetch_cars}
    if args.only == "audit":
        jobs = {"audit": fetch_audit}
    elif args.only:
        jobs = {args.only: jobs[args.only]}

    failures = 0
    for key, job in jobs.items():
        try:
            result = job(force=args.force)
        except (urllib.error.URLError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
            print(f"  FAILED  {key}: {exc}", file=sys.stderr)
            failures += 1
            continue
        for item in result if isinstance(result, list) else [result]:
            state = "already present" if item.skipped else "downloaded"
            print(
                f"  {state:>15}  {item.name}: {item.path.name} "
                f"({item.n_bytes:,} bytes)"
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
