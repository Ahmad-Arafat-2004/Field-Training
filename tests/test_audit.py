"""Tests for the corpus audit checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.audit import (
    cohens_d,
    feature_viability,
    learning_curve_f1,
    marker_audit,
    suspicious_zero_cells,
)
from src.features.spam_features import FEATURE_FUNCTIONS


@pytest.fixture
def contaminated() -> tuple[pd.Series, pd.Series]:
    """One class carries a marker the other never does -- an assembly artefact."""
    ham = [f"meeting on escapenumber about budget {i}" for i in range(200)]
    spam = [f"win big prize call now offer {i}" for i in range(200)]
    texts = pd.Series(ham + spam)
    y = pd.Series([0] * 200 + [1] * 200)
    return texts, y


@pytest.fixture
def clean_corpus() -> tuple[pd.Series, pd.Series]:
    """Both classes drawn through the same pipeline; markers overlap."""
    rng = np.random.default_rng(0)
    ham, spam = [], []
    for i in range(200):
        ham.append("hey are we still on for later" + (f" {i}" if i % 6 == 0 else ""))
        spam.append("WIN a prize" + (f" call 0900{i}" if i % 3 else " reply now"))
    return pd.Series(ham + spam), pd.Series([0] * 200 + [1] * 200)


class TestMarkerAudit:
    def test_returns_a_row_per_marker(self, clean_corpus) -> None:
        texts, y = clean_corpus
        audit = marker_audit(texts, y)
        assert len(audit) == 6
        assert set(audit.columns) >= {"marker", "negative_%", "positive_%", "auc"}

    def test_detects_a_one_sided_marker(self, contaminated) -> None:
        texts, y = contaminated
        audit = marker_audit(texts, y)
        row = audit[audit["marker"] == "has 'escapenumber'"].iloc[0]
        assert row["negative_%"] == 1.0
        assert row["positive_%"] == 0.0
        assert row["exact_zero"]

    def test_sorted_by_separating_power(self, contaminated) -> None:
        texts, y = contaminated
        audit = marker_audit(texts, y)
        assert audit.iloc[0]["auc_distance"] == audit["auc_distance"].max()

    def test_marker_absent_from_both_classes_is_not_flagged(self, clean_corpus) -> None:
        """0% on both sides is uninformative, not suspicious."""
        texts, y = clean_corpus
        audit = marker_audit(texts, y)
        row = audit[audit["marker"] == "has 'escapenumber'"].iloc[0]
        assert row["negative_%"] == 0.0 and row["positive_%"] == 0.0
        assert not row["exact_zero"]          # XOR, so both-zero is False
        assert row["auc"] == pytest.approx(0.5)

    def test_mismatched_lengths_rejected(self) -> None:
        with pytest.raises(ValueError, match="rows"):
            marker_audit(pd.Series(["a", "b"]), pd.Series([0]))

    def test_single_class_rejected(self) -> None:
        with pytest.raises(ValueError, match="both classes"):
            marker_audit(pd.Series(["a", "b"]), pd.Series([1, 1]))


class TestSuspiciousZeroCells:
    def test_flags_the_contaminating_marker(self, contaminated) -> None:
        texts, y = contaminated
        flagged = suspicious_zero_cells(marker_audit(texts, y))
        assert "has 'escapenumber'" in set(flagged["marker"])

    def test_clean_corpus_flags_nothing(self, clean_corpus) -> None:
        texts, y = clean_corpus
        assert len(suspicious_zero_cells(marker_audit(texts, y))) == 0

    def test_a_strong_marker_without_a_zero_cell_is_not_flagged(self) -> None:
        """The distinction the module exists to make: strong signal is not
        contamination unless one class is at an exact zero."""
        ham = ["no numbers here at all" for _ in range(100)]
        spam = [f"call 0900{i} now" for i in range(100)]
        # 5 ham rows also carry digits, so no cell is an exact zero.
        ham[:5] = [f"see you at {i} pm" for i in range(5)]
        audit = marker_audit(pd.Series(ham + spam), pd.Series([0] * 100 + [1] * 100))
        digit = audit[audit["marker"] == "has a raw digit"].iloc[0]
        assert digit["auc"] > 0.9          # separates strongly
        assert not digit["exact_zero"]     # but leaks, as real phenomena do
        assert "has a raw digit" not in set(suspicious_zero_cells(audit)["marker"])


class TestFeatureViability:
    def test_grades_dead_weak_and_ok(self) -> None:
        # Lowercased, no URLs -> case and URL features cannot fire.
        texts = pd.Series([f"plain lowercase message number {i}" for i in range(300)])
        via = feature_viability(texts, FEATURE_FUNCTIONS, sample=None)
        status = dict(zip(via["feature"], via["status"]))
        assert status["uppercase_ratio"] == "dead"
        assert status["all_caps_word_count"] == "dead"
        assert status["has_url_like"] == "dead"
        assert status["char_length"] == "ok"

    def test_all_features_alive_on_raw_text(self) -> None:
        texts = pd.Series(
            [
                "WIN £1000 now!! call 09061701461 www.prize.com",
                "hey are we still on for later?",
                "URGENT: claim your $500 at http://x.co 08001234567",
            ]
            * 60
        )
        via = feature_viability(texts, FEATURE_FUNCTIONS, sample=None)
        assert (via["status"] != "dead").all()

    def test_row_per_feature(self) -> None:
        texts = pd.Series(["hello world"] * 50)
        via = feature_viability(texts, FEATURE_FUNCTIONS, sample=None)
        assert len(via) == len(FEATURE_FUNCTIONS)

    def test_sampling_caps_the_work(self) -> None:
        texts = pd.Series(["hello world 123"] * 5000)
        via = feature_viability(texts, FEATURE_FUNCTIONS, sample=100)
        assert len(via) == len(FEATURE_FUNCTIONS)


class TestLearningCurve:
    @pytest.fixture
    def split(self):
        # Deliberately hard: 8 informative-but-noisy features and a nonlinear
        # boundary, so the curve has not saturated at 5% of the data. An easy
        # 1-D problem plateaus immediately and would make the curve test
        # vacuous.
        rng = np.random.default_rng(0)
        n = 1600
        X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"f{i}" for i in range(8)])
        signal = (X["f0"] * X["f1"] + X["f2"] ** 2 - X["f3"]).to_numpy()
        y = pd.Series((signal + rng.normal(0, 1.5, n) > 0).astype(int))
        return X.iloc[:1200], y.iloc[:1200], X.iloc[1200:], y.iloc[1200:]

    def test_returns_a_row_per_fraction(self, split) -> None:
        X_tr, y_tr, X_te, y_te = split
        curve = learning_curve_f1(
            X_tr, y_tr, X_te, y_te, fractions=(0.25, 0.5, 1.0), seeds=2
        )
        assert list(curve["fraction"]) == [0.25, 0.5, 1.0]
        assert (curve["f1_mean"] > 0).all()

    def test_row_counts_scale_with_the_fraction(self, split) -> None:
        X_tr, y_tr, X_te, y_te = split
        curve = learning_curve_f1(
            X_tr, y_tr, X_te, y_te, fractions=(0.25, 1.0), seeds=1
        )
        assert curve.iloc[0]["rows"] < curve.iloc[1]["rows"]
        assert curve.iloc[1]["rows"] == len(X_tr)

    def test_full_fraction_has_no_spread(self, split) -> None:
        """100% is deterministic -- there is nothing to resample."""
        X_tr, y_tr, X_te, y_te = split
        curve = learning_curve_f1(X_tr, y_tr, X_te, y_te, fractions=(1.0,), seeds=5)
        assert curve.iloc[0]["f1_std"] == 0.0

    def test_more_data_helps_on_a_learnable_problem(self, split) -> None:
        X_tr, y_tr, X_te, y_te = split
        curve = learning_curve_f1(
            X_tr, y_tr, X_te, y_te, fractions=(0.05, 1.0), seeds=3
        )
        assert curve.iloc[-1]["f1_mean"] > curve.iloc[0]["f1_mean"]


class TestCohensD:
    def test_zero_when_classes_match(self) -> None:
        values = pd.Series([1.0, 2.0, 1.0, 2.0])
        y = pd.Series([0, 0, 1, 1])
        assert cohens_d(values, y) == pytest.approx(0.0)

    def test_sign_follows_the_positive_class(self) -> None:
        values = pd.Series([0.0, 1.0, 5.0, 6.0])
        y = pd.Series([0, 0, 1, 1])
        assert cohens_d(values, y) > 0
        assert cohens_d(values, 1 - y) < 0

    def test_constant_column_is_zero_not_nan(self) -> None:
        values = pd.Series([3.0] * 10)
        y = pd.Series([0] * 5 + [1] * 5)
        assert cohens_d(values, y) == 0.0

    def test_perfect_separation_is_not_reported_as_no_effect(self) -> None:
        """Zero within-class spread but different class means separates the
        classes perfectly; returning 0.0 would say the opposite."""
        values = pd.Series([0.0, 0.0, 5.0, 5.0])
        y = pd.Series([0, 0, 1, 1])
        assert cohens_d(values, y) == np.inf
        assert cohens_d(values, 1 - y) == -np.inf
