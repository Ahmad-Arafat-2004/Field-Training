"""Tests for the preprocessing template.

The four cases the brief calls for are covered by the four classes below:
non-UTF-8 input, duplicate rows, unseen categories at transform time, and the
scaler's fitted mean matching the training partition rather than the full
dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.template import (
    AlreadyFittedError,
    NotFittedError,
    PreprocessingTemplate,
    TrainingPartition,
    describe_unseen_categories,
    split_dataset,
)
from src.utils.loader import read_csv_safe


# --------------------------------------------------------------------------
# 1. Non-UTF-8 input
# --------------------------------------------------------------------------
class TestNonUtf8Input:
    def test_latin1_csv_flows_into_the_template_intact(
        self, latin1_csv: Path
    ) -> None:
        """A Latin-1 corpus must survive loading and reach the template unmangled."""
        result = read_csv_safe(latin1_csv, verbose=False)
        assert result.encoding == "latin-1"
        assert result.used_fallback

        # The characters that broke UTF-8 are present and correct -- not U+FFFD.
        joined = " ".join(result.frame["message"])
        assert "£1000" in joined
        assert "Café" in joined
        assert "�" not in joined

        frame = result.frame.assign(
            n_chars=result.frame["message"].str.len().astype(float)
        )
        split = split_dataset(
            frame[["n_chars", "label"]], test_size=0.5, random_state=0
        )
        out = PreprocessingTemplate().fit_transform(split.train)
        assert not out.isna().any().any()

    def test_non_utf8_bytes_do_not_become_replacement_chars(
        self, latin1_csv: Path
    ) -> None:
        """Guards against a lenient reader that 'succeeds' by corrupting data."""
        with pytest.raises(UnicodeDecodeError):
            pd.read_csv(latin1_csv, encoding="utf-8")

        lenient = pd.read_csv(latin1_csv, encoding="utf-8", encoding_errors="replace")
        assert "�" in " ".join(lenient["message"])  # the failure mode

        safe = read_csv_safe(latin1_csv, verbose=False)
        assert "�" not in " ".join(safe.frame["message"])  # ours

    def test_utf8_is_preferred_when_it_works(self, utf8_csv: Path) -> None:
        result = read_csv_safe(utf8_csv, verbose=False)
        assert result.encoding == "utf-8"
        assert not result.used_fallback

    def test_latin1_categorical_values_encode_normally(
        self, latin1_csv: Path
    ) -> None:
        """Non-ASCII category levels must one-hot like any other level."""
        frame = read_csv_safe(latin1_csv, verbose=False).frame
        frame["city"] = ["Zürich", "Málaga", "Kraków"]
        frame["n"] = [1.0, 2.0, 3.0]
        split = split_dataset(
            frame[["city", "n"]], test_size=0.5, random_state=0
        )
        prep = PreprocessingTemplate()
        out = prep.fit_transform(split.train)
        assert any("ü" in c or "á" in c or "ó" in c for c in out.columns)


# --------------------------------------------------------------------------
# 2. Duplicate rows
# --------------------------------------------------------------------------
class TestDuplicateRows:
    def test_duplicates_are_dropped_before_the_split(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        doubled = pd.concat([mixed_frame, mixed_frame], ignore_index=True)
        split = split_dataset(doubled, test_size=0.2, random_state=42)

        assert split.n_rows_before_dedupe == 600
        assert split.n_duplicates_dropped == 300
        assert len(split.train.X) + len(split.test_X) == 300

    def test_no_row_appears_in_both_partitions(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        """The reason de-duplication has to precede the split.

        With duplicates left in, the same row can land on both sides and the
        held-out score stops measuring generalisation.
        """
        doubled = pd.concat([mixed_frame, mixed_frame], ignore_index=True)
        split = split_dataset(doubled, test_size=0.2, random_state=42)

        def as_rows(frame: pd.DataFrame) -> set[tuple]:
            return set(map(tuple, frame.astype(str).to_numpy()))

        assert not (as_rows(split.train.X) & as_rows(split.test_X))

    def test_leaving_duplicates_in_does_put_rows_on_both_sides(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        """Characterises the bug, so the guard above is known to be load-bearing."""
        doubled = pd.concat([mixed_frame, mixed_frame], ignore_index=True)
        split = split_dataset(
            doubled, test_size=0.2, random_state=42, drop_duplicates=False
        )

        def as_rows(frame: pd.DataFrame) -> set[tuple]:
            return set(map(tuple, frame.astype(str).to_numpy()))

        assert as_rows(split.train.X) & as_rows(split.test_X)
        assert split.n_duplicates_dropped == 0

    def test_target_stays_aligned_after_dedupe(
        self, mixed_frame: pd.DataFrame, binary_target: pd.Series
    ) -> None:
        """A dedupe that drops X rows without dropping the matching y rows
        silently shifts every label by one. Check the pairing directly."""
        frame = mixed_frame.assign(label=binary_target.to_numpy())
        doubled = pd.concat([frame, frame.iloc[:50]], ignore_index=True)

        split = split_dataset(
            doubled.drop(columns="label"),
            doubled["label"],
            test_size=0.25,
            random_state=42,
        )
        assert split.n_duplicates_dropped == 50
        assert len(split.train.X) == len(split.train.y)

        lookup = {
            tuple(row): label
            for row, label in zip(
                frame.drop(columns="label").astype(str).to_numpy().tolist(),
                frame["label"].tolist(),
            )
        }
        for row, label in zip(
            split.train.X.astype(str).to_numpy().tolist(), split.train.y.tolist()
        ):
            assert lookup[tuple(row)] == label

    def test_duplicate_subset_restricts_the_check(self) -> None:
        """Dedupe on the message text alone, as the spam notebook does."""
        frame = pd.DataFrame(
            {
                "message": ["a", "a", "b", "c"],
                "noise": [1.0, 2.0, 3.0, 4.0],  # makes full rows distinct
            }
        )
        full = split_dataset(frame, test_size=0.5, random_state=0)
        assert full.n_duplicates_dropped == 0

        by_text = split_dataset(
            frame, test_size=0.5, random_state=0, duplicate_subset="message"
        )
        assert by_text.n_duplicates_dropped == 1


# --------------------------------------------------------------------------
# 3. Unseen categories at transform time
# --------------------------------------------------------------------------
class TestUnseenCategories:
    @pytest.fixture
    def fitted(self, mixed_frame: pd.DataFrame):
        split = split_dataset(mixed_frame, test_size=0.2, random_state=42)
        prep = PreprocessingTemplate()
        prep.fit(split.train)
        return prep, split

    def test_unseen_category_does_not_raise(self, fitted) -> None:
        prep, split = fitted
        frame = split.test_X.copy()
        frame.loc[frame.index[0], "colour"] = "ultraviolet"
        out = prep.transform(frame)  # must not raise
        assert len(out) == len(frame)

    def test_unseen_category_encodes_to_all_zeros(self, fitted) -> None:
        """All-zeros is the honest encoding: fit-time saw nothing about it."""
        prep, split = fitted
        frame = split.test_X.copy()
        frame.loc[frame.index[0], "colour"] = "ultraviolet"
        out = prep.transform(frame)

        colour_cols = [c for c in out.columns if c.startswith("colour_")]
        assert out.iloc[0][colour_cols].sum() == 0.0
        # Other rows are unaffected.
        assert out.iloc[1][colour_cols].sum() == 1.0

    def test_column_count_is_stable_under_unseen_categories(self, fitted) -> None:
        """A widened test matrix would break every fitted estimator downstream."""
        prep, split = fitted
        frame = split.test_X.copy()
        frame["colour"] = "brand-new-colour"
        frame["size"] = "XXL"
        out = prep.transform(frame)
        assert list(out.columns) == prep.get_feature_names_out()

    def test_all_categorical_values_unseen(self, fitted) -> None:
        prep, split = fitted
        frame = split.test_X.copy()
        frame["colour"] = "nope"
        frame["size"] = "nope"
        out = prep.transform(frame)
        cat_cols = [
            c
            for c in out.columns
            if c.startswith("colour_") or c.startswith("size_")
        ]
        assert (out[cat_cols].to_numpy() == 0.0).all()

    def test_describe_unseen_categories_reports_them(self, fitted) -> None:
        prep, split = fitted
        frame = split.test_X.copy()
        frame.loc[frame.index[0], "colour"] = "ultraviolet"
        frame.loc[frame.index[1], "size"] = "XXL"
        assert describe_unseen_categories(prep, frame) == {
            "colour": ["ultraviolet"],
            "size": ["XXL"],
        }

    def test_missing_value_in_unseen_position_is_imputed(self, fitted) -> None:
        prep, split = fitted
        frame = split.test_X.copy()
        frame.loc[frame.index[0], "colour"] = np.nan
        out = prep.transform(frame)
        colour_cols = [c for c in out.columns if c.startswith("colour_")]
        # Imputed to the training mode, so it encodes as a real category.
        assert out.iloc[0][colour_cols].sum() == 1.0


# --------------------------------------------------------------------------
# 4. Scaler statistics come from the training partition only
# --------------------------------------------------------------------------
class TestNoLeakageIntoScaler:
    def test_scaler_mean_equals_train_mean_not_full_dataset_mean(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        """The headline assertion: fitted mean is the training-partition mean."""
        split = split_dataset(mixed_frame, test_size=0.2, random_state=42)
        prep = PreprocessingTemplate()
        prep.fit(split.train)

        numeric = prep.numeric_columns_
        train_imputed = prep.numeric_imputer_.transform(
            split.train.X[numeric].astype(float)
        )
        full_imputed = prep.numeric_imputer_.transform(
            mixed_frame[numeric].astype(float)
        )

        np.testing.assert_allclose(
            prep.scaler_.mean_, train_imputed.mean(axis=0), rtol=1e-12
        )
        # And is genuinely different from the full-dataset mean, so the
        # assertion above could actually fail if fit() ever saw everything.
        assert not np.allclose(
            prep.scaler_.mean_, full_imputed.mean(axis=0), rtol=1e-6
        )

    def test_scaler_scale_equals_train_std_not_full_dataset_std(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        split = split_dataset(mixed_frame, test_size=0.2, random_state=42)
        prep = PreprocessingTemplate().fit(split.train)

        numeric = prep.numeric_columns_
        train_imputed = prep.numeric_imputer_.transform(
            split.train.X[numeric].astype(float)
        )
        np.testing.assert_allclose(
            prep.scaler_.scale_, train_imputed.std(axis=0), rtol=1e-12
        )

    def test_imputer_median_comes_from_train_only(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        split = split_dataset(mixed_frame, test_size=0.2, random_state=42)
        prep = PreprocessingTemplate().fit(split.train)
        expected = split.train.X[prep.numeric_columns_].median().to_numpy()
        np.testing.assert_allclose(
            prep.numeric_imputer_.statistics_, expected, rtol=1e-12
        )

    def test_encoder_categories_come_from_train_only(self) -> None:
        frame = pd.DataFrame(
            {"n": np.arange(100, dtype=float), "cat": ["a"] * 99 + ["rare"]}
        )
        # random_state chosen so the single 'rare' row lands in the test half.
        split = split_dataset(frame, test_size=0.2, random_state=2)
        assert "rare" in set(split.test_X["cat"])
        assert "rare" not in set(split.train.X["cat"])

        prep = PreprocessingTemplate().fit(split.train)
        assert "rare" not in set(prep.encoder_.categories_[0].tolist())
        assert "cat_rare" not in prep.get_feature_names_out()

    def test_transformed_train_is_standardised_but_test_is_not_recentred(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        """Test columns must not come out with mean 0 -- that would mean the
        scaler was re-fitted on them."""
        split = split_dataset(mixed_frame, test_size=0.2, random_state=42)
        prep = PreprocessingTemplate()
        X_train, X_test, _, _ = prep.fit_on_split(split)

        numeric = prep.numeric_columns_
        np.testing.assert_allclose(
            X_train[numeric].mean().to_numpy(), np.zeros(len(numeric)), atol=1e-12
        )
        assert not np.allclose(
            X_test[numeric].mean().to_numpy(), np.zeros(len(numeric)), atol=1e-3
        )


# --------------------------------------------------------------------------
# Structural guards
# --------------------------------------------------------------------------
class TestStructuralGuards:
    def test_fit_rejects_a_bare_dataframe(self, mixed_frame: pd.DataFrame) -> None:
        with pytest.raises(TypeError, match="TrainingPartition"):
            PreprocessingTemplate().fit(mixed_frame)

    def test_fit_rejects_the_test_frame(self, mixed_frame: pd.DataFrame) -> None:
        split = split_dataset(mixed_frame, test_size=0.2, random_state=42)
        with pytest.raises(TypeError):
            PreprocessingTemplate().fit(split.test_X)

    def test_training_partition_cannot_be_built_directly(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        with pytest.raises(TypeError, match="cannot be constructed directly"):
            TrainingPartition(X=mixed_frame)

    def test_refit_is_refused(self, mixed_frame: pd.DataFrame) -> None:
        split = split_dataset(mixed_frame, test_size=0.2, random_state=42)
        prep = PreprocessingTemplate().fit(split.train)
        with pytest.raises(AlreadyFittedError):
            prep.fit(split.train)

    def test_clone_unfitted_gives_a_usable_fresh_instance(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        split = split_dataset(mixed_frame, test_size=0.2, random_state=42)
        prep = PreprocessingTemplate(numeric_strategy="mean").fit(split.train)
        fresh = prep.clone_unfitted()
        assert not fresh.is_fitted
        assert fresh.numeric_strategy == "mean"
        fresh.fit(split.train)
        np.testing.assert_allclose(fresh.scaler_.mean_, prep.scaler_.mean_)

    def test_transform_before_fit_raises(self, mixed_frame: pd.DataFrame) -> None:
        with pytest.raises(NotFittedError):
            PreprocessingTemplate().transform(mixed_frame)

    def test_transform_rejects_a_partition(self, mixed_frame: pd.DataFrame) -> None:
        split = split_dataset(mixed_frame, test_size=0.2, random_state=42)
        prep = PreprocessingTemplate().fit(split.train)
        with pytest.raises(TypeError, match="partition.X"):
            prep.transform(split.train)

    def test_missing_column_at_transform_raises_clearly(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        split = split_dataset(mixed_frame, test_size=0.2, random_state=42)
        prep = PreprocessingTemplate().fit(split.train)
        with pytest.raises(KeyError, match="colour"):
            prep.transform(split.test_X.drop(columns="colour"))

    def test_extra_column_at_transform_is_dropped_with_a_warning(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        split = split_dataset(mixed_frame, test_size=0.2, random_state=42)
        prep = PreprocessingTemplate().fit(split.train)
        frame = split.test_X.assign(leftover=1.0)
        with pytest.warns(UserWarning, match="not seen at fit time"):
            out = prep.transform(frame)
        assert list(out.columns) == prep.get_feature_names_out()

    def test_column_order_at_transform_does_not_matter(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        split = split_dataset(mixed_frame, test_size=0.2, random_state=42)
        prep = PreprocessingTemplate().fit(split.train)
        straight = prep.transform(split.test_X)
        shuffled = prep.transform(split.test_X[split.test_X.columns[::-1]])
        pd.testing.assert_frame_equal(straight, shuffled)


class TestSplitBehaviour:
    def test_stratify_preserves_class_balance(
        self, mixed_frame: pd.DataFrame, binary_target: pd.Series
    ) -> None:
        split = split_dataset(
            mixed_frame,
            binary_target,
            test_size=0.2,
            random_state=42,
            stratify=True,
        )
        train_rate = split.train.y.mean()
        test_rate = split.test_y.mean()
        assert abs(train_rate - test_rate) < 0.02

    def test_split_is_reproducible(self, mixed_frame: pd.DataFrame) -> None:
        a = split_dataset(mixed_frame, test_size=0.2, random_state=42)
        b = split_dataset(mixed_frame, test_size=0.2, random_state=42)
        pd.testing.assert_frame_equal(a.train.X, b.train.X)

    def test_different_seeds_give_different_splits(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        a = split_dataset(mixed_frame, test_size=0.2, random_state=42)
        b = split_dataset(mixed_frame, test_size=0.2, random_state=7)
        assert not a.train.X.index.equals(b.train.X.index)

    def test_mismatched_lengths_are_rejected(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="rows"):
            split_dataset(mixed_frame, pd.Series([0, 1, 0]))

    def test_stratify_true_without_y_is_rejected(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="requires y"):
            split_dataset(mixed_frame, stratify=True)

    def test_fitting_on_an_empty_partition_is_rejected(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        split = split_dataset(frame, test_size=0.2, random_state=0)
        empty = TrainingPartition.__new__(TrainingPartition)
        object.__setattr__(empty, "X", split.train.X.iloc[0:0])
        object.__setattr__(empty, "y", None)
        object.__setattr__(empty, "_token", None)
        with pytest.raises(ValueError, match="empty training partition"):
            PreprocessingTemplate().fit(empty)

    def test_transforming_an_empty_frame_returns_the_fitted_schema(self) -> None:
        """An empty batch is a legitimate input, not an error."""
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        split = split_dataset(frame, test_size=0.2, random_state=0)
        prep = PreprocessingTemplate().fit(split.train)
        out = prep.transform(split.test_X.iloc[0:0])
        assert out.shape == (0, 1)
        assert list(out.columns) == prep.get_feature_names_out()


class TestMessyValues:
    def test_numeric_column_arriving_as_object_is_coerced(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        """One stray 'n/a' should cost one value, not the whole transform."""
        split = split_dataset(mixed_frame, test_size=0.2, random_state=42)
        prep = PreprocessingTemplate().fit(split.train)

        frame = split.test_X.copy()
        frame["length"] = frame["length"].astype(object)
        frame.loc[frame.index[0], "length"] = "n/a"

        out = prep.transform(frame)
        assert not out.isna().any().any()
        # The unparseable cell became the training median, then was scaled.
        expected = (
            prep.numeric_imputer_.statistics_[prep.numeric_columns_.index("length")]
            - prep.scaler_.mean_[prep.numeric_columns_.index("length")]
        ) / prep.scaler_.scale_[prep.numeric_columns_.index("length")]
        assert out.iloc[0]["length"] == pytest.approx(expected)

    def test_all_numeric_frame_needs_no_encoder(self) -> None:
        frame = pd.DataFrame(
            {"a": np.arange(50, dtype=float), "b": np.arange(50, dtype=float) * 2}
        )
        split = split_dataset(frame, test_size=0.2, random_state=0)
        prep = PreprocessingTemplate().fit(split.train)
        assert prep.encoder_ is None
        assert prep.get_feature_names_out() == ["a", "b"]

    def test_all_categorical_frame_needs_no_scaler(self) -> None:
        frame = pd.DataFrame({"c": ["a", "b", "c"] * 20})
        split = split_dataset(
            frame, test_size=0.2, random_state=0, drop_duplicates=False
        )
        prep = PreprocessingTemplate().fit(split.train)
        assert prep.scaler_ is None
        assert len(prep.get_feature_names_out()) == 3

    def test_max_categories_caps_the_one_hot_width(self, rng) -> None:
        frame = pd.DataFrame(
            {
                "n": rng.normal(size=500),
                "high_card": [f"v{i % 120}" for i in range(500)],
            }
        )
        split = split_dataset(frame, test_size=0.2, random_state=0)
        prep = PreprocessingTemplate(max_categories=10).fit(split.train)
        one_hot = [c for c in prep.get_feature_names_out() if c != "n"]
        assert len(one_hot) <= 10

    def test_scale_encoded_scaler_is_also_fitted_on_train_only(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        split = split_dataset(mixed_frame, test_size=0.2, random_state=42)
        prep = PreprocessingTemplate(scale_encoded=True)
        X_train, X_test, _, _ = prep.fit_on_split(split)
        assert prep.encoded_scaler_ is not None

        cat_cols = [c for c in X_train.columns if "_" in c and c.split("_")[0] in
                    prep.categorical_columns_]
        np.testing.assert_allclose(
            X_train[cat_cols].mean().to_numpy(),
            np.zeros(len(cat_cols)),
            atol=1e-12,
        )
        # Test half is transformed, not re-centred.
        assert not np.allclose(
            X_test[cat_cols].mean().to_numpy(), np.zeros(len(cat_cols)), atol=1e-6
        )
