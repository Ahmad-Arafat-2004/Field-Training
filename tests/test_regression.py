"""Tests for the used-car cleaning and the regression comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.car_features import (
    clean_car_frame,
    parse_measurement,
    parse_owner_rank,
    parse_torque_nm,
)
from src.models.regression_comparison import (
    build_estimators,
    overfitting_demo,
    prepare_split,
    run_comparison,
)
from src.preprocessing.template import PreprocessingTemplate, split_dataset


@pytest.fixture
def raw_cars(rng: np.random.Generator) -> pd.DataFrame:
    """A synthetic frame with the same schema and messiness as the real one."""
    n = 400
    engine = rng.integers(800, 3000, n).astype(float)
    power = engine / 12 + rng.normal(0, 5, n)
    age = rng.integers(1, 20, n)
    price = 40000 + power * 9000 - age * 25000 + rng.normal(0, 40000, n)
    frame = pd.DataFrame(
        {
            "name": [
                f"{make} Model{i % 7}"
                for i, make in enumerate(
                    rng.choice(["Maruti", "Honda", "Bmw", "Tata"], n)
                )
            ],
            "year": 2021 - age,
            "selling_price": price.clip(30000).round(),
            "km_driven": rng.integers(1000, 200000, n),
            "fuel": rng.choice(["Petrol", "Diesel", "CNG"], n),
            "seller_type": rng.choice(["Individual", "Dealer"], n),
            "transmission": rng.choice(["Manual", "Automatic"], n),
            "owner": rng.choice(["First Owner", "Second Owner", "Third Owner"], n),
            "mileage": [f"{v:.2f} kmpl" for v in rng.uniform(10, 25, n)],
            "engine": [f"{int(v)} CC" for v in engine],
            "max_power": [f"{v:.2f} bhp" for v in power],
            "torque": [f"{int(v)}Nm@ 2000rpm" for v in power * 2],
            "seats": rng.choice([5.0, 7.0], n),
        }
    )
    frame.loc[3:9, "mileage"] = np.nan
    return frame


class TestMeasurementParsing:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("23.4 kmpl", 23.4),
            ("1248 CC", 1248.0),
            ("74 bhp", 74.0),
            ("103.52 bhp", 103.52),
            (42, 42.0),
            ("", np.nan),
            (None, np.nan),
            (np.nan, np.nan),
            ("bhp", np.nan),
        ],
    )
    def test_parse_measurement(self, value: object, expected: float) -> None:
        out = parse_measurement(value)
        if np.isnan(expected):
            assert np.isnan(out)
        else:
            assert out == pytest.approx(expected)

    def test_nm_is_taken_as_is(self) -> None:
        assert parse_torque_nm("190Nm@ 2000rpm") == pytest.approx(190.0)
        assert parse_torque_nm("250Nm at 1500-2500rpm") == pytest.approx(250.0)

    def test_kgm_is_converted_to_nm(self) -> None:
        """Mixing units in one column would put 12.7 next to 190 meaning the same."""
        assert parse_torque_nm("12.7@ 2,700(kgm@ rpm)") == pytest.approx(
            12.7 * 9.80665
        )

    def test_bare_small_number_is_treated_as_kgm(self) -> None:
        assert parse_torque_nm("11.5@ 4,500(rpm)") == pytest.approx(11.5 * 9.80665)

    def test_bare_large_number_is_treated_as_nm(self) -> None:
        assert parse_torque_nm("380@ 2000") == pytest.approx(380.0)

    def test_unparseable_torque_is_nan(self) -> None:
        assert np.isnan(parse_torque_nm(""))
        assert np.isnan(parse_torque_nm(None))

    def test_owner_rank_is_ordinal(self) -> None:
        assert parse_owner_rank("First Owner") == 1.0
        assert parse_owner_rank("Third Owner") == 3.0
        assert parse_owner_rank("Test Drive Car") == 0.0
        assert np.isnan(parse_owner_rank("Something Else"))

    def test_owner_rank_ordering_is_preserved(self) -> None:
        ranks = [
            parse_owner_rank(label)
            for label in ["First Owner", "Second Owner", "Fourth & Above Owner"]
        ]
        assert ranks == sorted(ranks)


class TestCleanCarFrame:
    def test_unit_columns_become_numeric(self, raw_cars: pd.DataFrame) -> None:
        cleaned = clean_car_frame(raw_cars)
        for column in ("mileage", "engine", "max_power", "torque"):
            assert pd.api.types.is_numeric_dtype(cleaned[column]), column

    def test_missing_values_survive_as_nan(self, raw_cars: pd.DataFrame) -> None:
        """~220 real rows are genuinely missing these; that is the imputer's job."""
        cleaned = clean_car_frame(raw_cars)
        assert cleaned["mileage"].isna().sum() == 7

    def test_age_replaces_year(self, raw_cars: pd.DataFrame) -> None:
        cleaned = clean_car_frame(raw_cars, reference_year=2021)
        assert "year" not in cleaned.columns
        assert (cleaned["age"] >= 0).all()

    def test_name_is_reduced_to_make(self, raw_cars: pd.DataFrame) -> None:
        cleaned = clean_car_frame(raw_cars)
        assert "name" not in cleaned.columns
        assert set(cleaned["make"]) <= {"Maruti", "Honda", "Bmw", "Tata"}

    def test_owner_becomes_a_rank(self, raw_cars: pd.DataFrame) -> None:
        cleaned = clean_car_frame(raw_cars)
        assert "owner" not in cleaned.columns
        assert pd.api.types.is_numeric_dtype(cleaned["owner_rank"])

    def test_input_frame_is_not_mutated(self, raw_cars: pd.DataFrame) -> None:
        before = raw_cars.copy()
        clean_car_frame(raw_cars)
        pd.testing.assert_frame_equal(raw_cars, before)


class TestDropFirstFixesRankDeficiency:
    """The dummy-variable trap, which is what broke multiple linear regression."""

    @staticmethod
    def _matrix(drop_first: bool) -> pd.DataFrame:
        # The two categorical columns must be drawn independently. Repeating
        # fixed patterns of period 4 and 2 would make 'b' a function of 'a',
        # adding a collinearity that has nothing to do with the dummy trap.
        draw = np.random.default_rng(0)
        frame = pd.DataFrame(
            {
                "n": np.arange(200, dtype=float),
                "a": draw.choice(["x", "y", "z", "w"], 200),
                "b": draw.choice(["p", "q"], 200),
            }
        )
        split = split_dataset(frame, test_size=0.2, random_state=0,
                              drop_duplicates=False)
        return PreprocessingTemplate(drop_first=drop_first).fit_transform(split.train)

    def test_full_dummies_are_rank_deficient(self) -> None:
        matrix = self._matrix(drop_first=False)
        # 1 numeric + 4 + 2 dummies = 7 columns, but each dummy set sums to 1,
        # so two directions are redundant once an intercept is added.
        assert matrix.shape[1] == 7
        augmented = np.hstack([matrix.to_numpy(), np.ones((len(matrix), 1))])
        assert np.linalg.matrix_rank(augmented) < augmented.shape[1]

    def test_drop_first_restores_full_rank(self) -> None:
        matrix = self._matrix(drop_first=True)
        assert matrix.shape[1] == 5  # 1 + 3 + 1
        augmented = np.hstack([matrix.to_numpy(), np.ones((len(matrix), 1))])
        assert np.linalg.matrix_rank(augmented) == augmented.shape[1]

    def test_drop_first_survives_a_clone(self) -> None:
        template = PreprocessingTemplate(drop_first=True)
        assert template.clone_unfitted().drop_first is True

    def test_unseen_still_does_not_raise_with_drop_first(self) -> None:
        frame = pd.DataFrame(
            {"n": np.arange(100, dtype=float), "a": ["x", "y"] * 50}
        )
        split = split_dataset(frame, test_size=0.2, random_state=0,
                              drop_duplicates=False)
        template = PreprocessingTemplate(drop_first=True).fit(split.train)
        test = split.test_X.copy()
        test.loc[test.index[0], "a"] = "brand_new"
        out = template.transform(test)  # must not raise
        assert list(out.columns) == template.get_feature_names_out()


class TestRunComparison:
    def test_all_six_families_are_present(self) -> None:
        names = list(build_estimators(["a", "b"]))
        assert len(names) == 6
        assert any("Simple linear" in n for n in names)
        assert any("Multiple linear" in n for n in names)
        assert any("Polynomial" in n for n in names)
        assert any("SVR" in n for n in names)
        assert any("Decision tree" in n for n in names)
        assert any("Random forest" in n for n in names)

    def test_random_forest_uses_200_trees(self) -> None:
        forest = build_estimators(["a"])["Random forest (200 trees)"]
        assert forest.n_estimators == 200

    def test_svr_scales_the_target(self) -> None:
        """An RBF kernel works on distances; an unscaled target breaks epsilon."""
        svr = build_estimators(["a"])["SVR (RBF kernel)"]
        assert hasattr(svr, "transformer")
        assert svr.regressor.kernel == "rbf"

    def test_comparison_runs_and_ranks_sensibly(
        self, raw_cars: pd.DataFrame
    ) -> None:
        table, fitted, split = run_comparison(
            clean_car_frame(raw_cars), verbose=False
        )
        assert len(table) == 6
        assert len(fitted) == 6
        # No model should be catastrophically broken: a negative R2 here would
        # mean worse than predicting the mean, which is what the rank-deficient
        # design matrix used to produce.
        assert (table["Test R2"] > 0).all(), table.to_string()

    def test_every_model_sees_the_same_split(self, raw_cars: pd.DataFrame) -> None:
        frame = clean_car_frame(raw_cars)
        a_split, _, a_train, a_test = prepare_split(frame)
        b_split, _, b_train, b_test = prepare_split(frame)
        pd.testing.assert_frame_equal(a_train, b_train)
        pd.testing.assert_frame_equal(a_test, b_test)
        pd.testing.assert_series_equal(a_split.train.y, b_split.train.y)

    def test_missing_target_column_raises_clearly(
        self, raw_cars: pd.DataFrame
    ) -> None:
        with pytest.raises(KeyError, match="not in the data"):
            prepare_split(clean_car_frame(raw_cars), target="no_such_column")

    def test_decision_tree_memorises_the_training_set(
        self, raw_cars: pd.DataFrame
    ) -> None:
        """An unpruned tree hits train R2 = 1.0; that gap is the point of the table."""
        table, _, _ = run_comparison(clean_car_frame(raw_cars), verbose=False)
        tree = table.loc[table["Model"] == "Decision tree"].iloc[0]
        assert tree["Train R2"] > 0.99
        assert tree["R2 gap"] > 0


class TestOverfittingDemo:
    def test_degree_four_gap_exceeds_degree_three_gap(
        self, raw_cars: pd.DataFrame
    ) -> None:
        """The demo only demonstrates anything if the gap actually opens up."""
        demo = overfitting_demo(clean_car_frame(raw_cars), degrees=(3, 4))
        gaps = dict(zip(demo["Degree"], demo["Gap (train - test)"]))
        assert gaps[4] > gaps[3]

    def test_train_r2_rises_with_degree(self, raw_cars: pd.DataFrame) -> None:
        """More terms always fit the training data better -- that is the trap."""
        demo = overfitting_demo(clean_car_frame(raw_cars), degrees=(1, 2, 3, 4))
        train = demo["Train R2"].tolist()
        assert train == sorted(train)

    def test_term_count_grows_combinatorially(
        self, raw_cars: pd.DataFrame
    ) -> None:
        demo = overfitting_demo(
            clean_car_frame(raw_cars), degrees=(1, 4), n_features=6
        )
        terms = dict(zip(demo["Degree"], demo["Terms"]))
        assert terms[1] == 6
        assert terms[4] == 209  # C(10,4) - 1
