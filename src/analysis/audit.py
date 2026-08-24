"""Checks to run on a corpus *before* trusting a score computed from it.

Three questions this module answers, all by measurement:

1. **Is the label recoverable from a surface artefact?** If one trivial marker
   separates the classes almost perfectly, the corpus may be recording how it
   was assembled rather than what the classes are.
2. **Does a feature have any variance here?** A feature that reads the same
   value on every row is a dead input, and no quantity of rows revives it.
3. **Is the model actually short of data?** A flat learning curve says the
   ceiling is the feature set, not the row count.

The first check needs care, and the care is the point. A high single-marker
AUC is *evidence of nothing on its own* -- it means "this marker separates the
classes", which is exactly what a genuinely strong feature also does. The SMS
corpus scores 0.896 on "contains a digit" because spam has to carry a number
to call; that is signal. A different corpus scores 0.906 on the same marker
because its two classes were preprocessed by different pipelines; that is
contamination. The number is nearly identical and the meaning is opposite.

What separates them is a *causal story*, plus the tell in
:func:`suspicious_zero_cells`: a marker that appears in exactly 0.00% of one
class is almost never natural. Real phenomena are leaky; assembly pipelines
are absolute.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

__all__ = [
    "DEFAULT_MARKERS",
    "cohens_d",
    "feature_viability",
    "learning_curve_f1",
    "marker_audit",
    "suspicious_zero_cells",
]

# Surface markers that should never, on their own, decide a class. Each is a
# regex applied to the raw text; `regex=False` entries are literal.
DEFAULT_MARKERS: Mapping[str, tuple[str, bool]] = {
    "has a raw digit": (r"\d", True),
    "has any uppercase": (r"[A-Z]", True),
    "has 'escapenumber'": ("escapenumber", False),
    "has http/www": (r"http|www\.", True),
    "has '!'": ("!", False),
    "has a currency symbol": (r"[£$€¥₹]", True),
}


def cohens_d(values: pd.Series, target: pd.Series) -> float:
    """Standardised mean difference between the two classes."""
    values = pd.Series(values).astype(float)
    target = pd.Series(target).reset_index(drop=True)
    values = values.reset_index(drop=True)
    a, b = values[target == 1], values[target == 0]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    difference = float(a.mean() - b.mean())
    if pooled == 0:
        # Zero within-class spread. Two very different cases share this branch:
        # a genuinely constant column (means equal too) has no effect, while
        # equal-but-different constants per class separate the classes
        # perfectly. Collapsing both to 0.0 would report the second as "no
        # signal", which is the opposite of the truth.
        if difference == 0:
            return 0.0
        return float(np.inf) if difference > 0 else float(-np.inf)
    return difference / pooled


def marker_audit(
    texts: pd.Series,
    y: pd.Series,
    markers: Mapping[str, tuple[str, bool]] = DEFAULT_MARKERS,
) -> pd.DataFrame:
    """Per-marker prevalence in each class, plus the AUC of the marker alone.

    Returns a frame with one row per marker: share in the negative class,
    share in the positive class, AUC, and whether either cell is an exact
    zero.

    **The AUC column does not classify a corpus by itself.** Read it together
    with `exact_zero` and with a causal account of why the marker would differ
    between classes. See the module docstring.
    """
    texts = pd.Series(texts).fillna("").astype(str).reset_index(drop=True)
    y = pd.Series(y).astype(int).reset_index(drop=True)
    if len(texts) != len(y):
        raise ValueError(f"texts has {len(texts)} rows but y has {len(y)}.")
    if y.nunique() < 2:
        raise ValueError("y must contain both classes.")

    rows = []
    for name, (pattern, is_regex) in markers.items():
        present = texts.str.contains(pattern, regex=is_regex, case=False, na=False)
        neg = float(present[y == 0].mean())
        pos = float(present[y == 1].mean())
        rows.append(
            {
                "marker": name,
                "negative_%": neg,
                "positive_%": pos,
                "auc": float(roc_auc_score(y, present.astype(int))),
                "exact_zero": bool(neg == 0.0) ^ bool(pos == 0.0),
            }
        )
    frame = pd.DataFrame(rows)
    frame["auc_distance"] = (frame["auc"] - 0.5).abs()
    return frame.sort_values("auc_distance", ascending=False, ignore_index=True)


def suspicious_zero_cells(audit: pd.DataFrame, min_auc_distance: float = 0.2) -> pd.DataFrame:
    """Markers that are absent from exactly one class *and* separate strongly.

    This pairing is the contamination tell. A marker can legitimately be rare
    in one class, but a hard 0.00% alongside substantial prevalence in the
    other usually means the two classes travelled through different
    preprocessing, not that the world is that tidy.
    """
    return audit[
        audit["exact_zero"] & (audit["auc_distance"] >= min_auc_distance)
    ].reset_index(drop=True)


def feature_viability(
    texts: pd.Series,
    feature_functions: Mapping[str, Callable[[object], float]],
    *,
    dead_threshold: float = 0.01,
    weak_threshold: float = 0.05,
    sample: int | None = 6000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Which features have any variance on this corpus?

    A feature whose value is zero on effectively every row contributes
    nothing, however many rows there are. Reports the share of non-zero values
    per feature and grades it dead / weak / ok.
    """
    texts = pd.Series(texts).dropna().astype(str)
    if sample is not None and len(texts) > sample:
        texts = texts.sample(sample, random_state=random_state)

    rows = []
    for name, func in feature_functions.items():
        values = texts.apply(func).astype(float)
        nonzero = float((values != 0).mean())
        status = (
            "dead" if nonzero < dead_threshold
            else "weak" if nonzero < weak_threshold
            else "ok"
        )
        rows.append(
            {
                "feature": name,
                "mean": float(values.mean()),
                "nonzero_%": nonzero,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def learning_curve_f1(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    fractions: Sequence[float] = (0.05, 0.10, 0.25, 0.50, 0.75, 1.00),
    seeds: int = 5,
    estimator_factory: Callable[[], object] | None = None,
    pos_label: int = 1,
) -> pd.DataFrame:
    """Test-set F1 as a function of how much training data the model is given.

    Each fraction is a stratified subsample of the *training* partition; the
    test set never changes. Repeated over `seeds` draws so a point is a mean
    with a spread rather than one lucky subsample.

    A curve that has gone flat is the answer to "would more data help?" -- and
    therefore also to "would synthetic data help?", since oversampling cannot
    add information the corpus does not contain.
    """
    if estimator_factory is None:
        def estimator_factory():  # noqa: E306
            return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    rows = []
    for fraction in fractions:
        scores = []
        n_rows = len(X_train)
        for seed in range(seeds):
            if fraction >= 1.0:
                X_sub, y_sub = X_train, y_train
            else:
                X_sub, _, y_sub, _ = train_test_split(
                    X_train, y_train,
                    train_size=fraction, random_state=seed, stratify=y_train,
                )
            model = estimator_factory()
            model.fit(X_sub, y_sub)
            scores.append(
                f1_score(
                    y_test, model.predict(X_test),
                    pos_label=pos_label, zero_division=0,
                )
            )
            n_rows = len(X_sub)
            if fraction >= 1.0:
                break            # deterministic; repeating adds nothing
        rows.append(
            {
                "fraction": fraction,
                "rows": n_rows,
                "f1_mean": float(np.mean(scores)),
                "f1_std": float(np.std(scores)),
            }
        )
    frame = pd.DataFrame(rows)
    frame["gain"] = frame["f1_mean"].diff()
    return frame
