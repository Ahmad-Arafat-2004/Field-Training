"""Compare six regression families through one shared preprocessing template.

    python -m src.models.regression_comparison
    python -m src.models.regression_comparison --data path/to/other.csv --target price

The design constraint is that the estimator is the *only* thing that varies.
Every model receives the identical train/test split and the identical fitted
preprocessing, so a difference in the results table is a difference between
model families and not between two people's idea of how to scale a column.

The one place that would break the rule is SVR, which needs a scaled target as
well as scaled features -- an RBF kernel works on distances, and a target in
the hundreds of thousands makes epsilon and C meaningless. Rather than scaling
y outside the loop (which would change the data every other model sees) the
target scaling is attached to the SVR estimator itself with
``TransformedTargetRegressor``. That keeps y in native units for everyone
else, keeps the scaler fitted on training folds only, and returns predictions
in rupees so RMSE stays comparable down the column.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # pragma: no cover - script invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.features.car_features import CAR_TARGET, clean_car_frame
from src.preprocessing.template import (
    DataSplit,
    PreprocessingTemplate,
    split_dataset,
)
from src.utils.loader import read_csv_safe

__all__ = [
    "ModelResult",
    "build_estimators",
    "evaluate",
    "load_cars",
    "overfitting_demo",
    "prepare_split",
    "run_comparison",
]

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data" / "car_details_v3.csv"

RANDOM_STATE = 42

# The single feature for the simple-linear model. max_power is the strongest
# individual correlate of price in this dataset, so simple-vs-multiple linear
# measures what the *other* columns add rather than penalising simple linear
# for a badly chosen feature.
SIMPLE_FEATURE = "max_power"


@dataclass
class ModelResult:
    """Held-out scores for one estimator, plus its training scores."""

    name: str
    family: str
    r2: float
    rmse: float
    train_r2: float
    train_rmse: float
    n_features: int

    @property
    def r2_gap(self) -> float:
        """Train R2 minus test R2 -- the overfitting tell."""
        return self.train_r2 - self.r2


def load_cars(path: str | Path = DEFAULT_DATA) -> pd.DataFrame:
    """Load and clean the used-car listings."""
    frame = read_csv_safe(path, verbose=False).frame
    return clean_car_frame(frame)


def prepare_split(
    frame: pd.DataFrame,
    target: str = CAR_TARGET,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    max_categories: int = 20,
) -> tuple[DataSplit, PreprocessingTemplate, pd.DataFrame, pd.DataFrame]:
    """De-duplicate, split, and fit the shared preprocessing on the train half.

    Returns ``(split, fitted_template, X_train, X_test)``. Everything
    downstream reuses these, which is what makes the comparison fair.
    """
    if target not in frame.columns:
        raise KeyError(
            f"Target column {target!r} not in the data. "
            f"Columns are: {list(frame.columns)}"
        )

    rows = frame.dropna(subset=[target])
    y = rows[target]
    feature_columns = [c for c in rows.columns if c != target]

    # The whole row, target included, goes into the duplicate check, and
    # keep_columns holds the target out of the model matrix. Checking features
    # alone would treat two listings with the same specification but different
    # asking prices as duplicates and discard one of them -- but those are not
    # duplicates, they are two real observations that happen to disagree, and
    # that disagreement is exactly the irreducible noise the model should be
    # scored against. Only a row identical in features *and* price is a
    # genuine duplicate.
    split = split_dataset(
        rows,
        y,
        test_size=test_size,
        random_state=random_state,
        drop_duplicates=True,
        keep_columns=feature_columns,
    )
    # drop_first because four of the six families here are linear. A full dummy
    # set costs one rank per categorical column: without it this design matrix
    # comes out rank 34 of 37 with a condition number of 1.6e16, and ordinary
    # least squares answers with coefficients of order 1e17 that cancel on the
    # training rows and produce a test R2 of -2.9e18.
    template = PreprocessingTemplate(
        max_categories=max_categories, drop_first=True
    )
    X_train, X_test, _, _ = template.fit_on_split(split)
    return split, template, X_train, X_test


def _polynomial_pipeline(
    degree: int, numeric_columns: Sequence[str]
) -> Pipeline:
    """Polynomial expansion over the numeric columns only.

    Expanding the one-hot columns too would be worse than useless: for a 0/1
    indicator x, x**2 == x**3 == x, so every power of every dummy is an exact
    duplicate of the dummy itself and the expansion manufactures thousands of
    perfectly collinear terms. Restricting the expansion is the standard
    treatment, and it keeps the polynomial model a *model* choice rather than a
    preprocessing difference -- the shared template still produced the columns.
    """
    return make_pipeline(
        ColumnTransformer(
            [
                (
                    "poly",
                    PolynomialFeatures(degree=degree, include_bias=False),
                    list(numeric_columns),
                )
            ],
            # Dummies pass through unexpanded; they still enter the linear fit.
            remainder="passthrough",
        ),
        LinearRegression(),
    )


def build_estimators(
    numeric_columns: Sequence[str], random_state: int = RANDOM_STATE
) -> dict[str, Any]:
    """The six families, in increasing order of flexibility."""
    return {
        "Simple linear (1 feature)": LinearRegression(),
        "Multiple linear": LinearRegression(),
        # include_bias=False because LinearRegression already fits an
        # intercept; keeping both would make the design matrix rank-deficient.
        "Polynomial (degree 3)": _polynomial_pipeline(3, numeric_columns),
        "SVR (RBF kernel)": TransformedTargetRegressor(
            # C=100 and epsilon=0.1 apply to the *scaled* target, so epsilon is
            # 0.1 standard deviations of price rather than 0.1 rupees. With
            # default C=1 the RBF fit underfits badly on this data.
            regressor=SVR(kernel="rbf", C=100.0, epsilon=0.1, gamma="scale"),
            transformer=StandardScaler(),
        ),
        "Decision tree": DecisionTreeRegressor(random_state=random_state),
        "Random forest (200 trees)": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def _score(model: Any, X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    predictions = model.predict(X)
    return (
        float(r2_score(y, predictions)),
        float(np.sqrt(mean_squared_error(y, predictions))),
    )


def evaluate(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    family: str = "",
) -> ModelResult:
    """Fit on train, score on both halves."""
    model.fit(X_train, y_train)
    test_r2, test_rmse = _score(model, X_test, y_test)
    train_r2, train_rmse = _score(model, X_train, y_train)
    return ModelResult(
        name=name,
        family=family or name,
        r2=test_r2,
        rmse=test_rmse,
        train_r2=train_r2,
        train_rmse=train_rmse,
        n_features=X_train.shape[1],
    )


def run_comparison(
    frame: pd.DataFrame,
    target: str = CAR_TARGET,
    *,
    random_state: int = RANDOM_STATE,
    max_categories: int = 20,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], DataSplit]:
    """Fit all six families and return the comparison table.

    Returns ``(results_table, fitted_models, split)``.
    """
    split, template, X_train, X_test = prepare_split(
        frame, target, random_state=random_state, max_categories=max_categories
    )
    y_train, y_test = split.train.y, split.test_y

    estimators = build_estimators(template.numeric_columns_, random_state)
    results: list[ModelResult] = []
    fitted: dict[str, Any] = {}

    for name, model in estimators.items():
        # Simple linear is the one model that sees a different design matrix:
        # by definition it gets a single feature. Everything else gets the
        # identical fitted preprocessing.
        if name.startswith("Simple linear"):
            if SIMPLE_FEATURE not in X_train.columns:
                raise KeyError(
                    f"{SIMPLE_FEATURE!r} is not a column after preprocessing; "
                    f"available: {list(X_train.columns)[:10]}"
                )
            train_X = X_train[[SIMPLE_FEATURE]]
            test_X = X_test[[SIMPLE_FEATURE]]
        else:
            train_X, test_X = X_train, X_test

        if verbose:
            print(f"  fitting {name} ...", flush=True)
        results.append(
            evaluate(name, model, train_X, test_X, y_train, y_test)
        )
        fitted[name] = model

    table = pd.DataFrame(
        [
            {
                "Model": r.name,
                "Features": r.n_features if not r.name.startswith("Simple") else 1,
                "Test R2": r.r2,
                "Test RMSE": r.rmse,
                "Train R2": r.train_r2,
                "R2 gap": r.r2_gap,
            }
            for r in results
        ]
    ).sort_values("Test R2", ascending=False, ignore_index=True)

    return table, fitted, split


def overfitting_demo(
    frame: pd.DataFrame,
    target: str = CAR_TARGET,
    *,
    degrees: tuple[int, ...] = (1, 2, 3, 4),
    random_state: int = RANDOM_STATE,
    n_features: int = 6,
    max_categories: int = 20,
) -> pd.DataFrame:
    """Train R2 vs test R2 across polynomial degrees, to show the gap open up.

    Restricted to the `n_features` numeric columns with the strongest linear
    relationship to the target. Degree 4 over the full ~40-column design matrix
    expands to well over a hundred thousand terms, which is a memory problem
    rather than a teaching example; the pathology is identical and legible at
    six.
    """
    split, template, X_train, X_test = prepare_split(
        frame, target, random_state=random_state, max_categories=max_categories
    )
    y_train, y_test = split.train.y, split.test_y

    # Numeric columns only. Raising a 0/1 dummy to a power returns the dummy,
    # so including one-hots would pad the expansion with exact duplicates and
    # confound the degree sweep with a rank problem.
    numeric = [c for c in template.numeric_columns_ if c in X_train.columns]
    strongest = (
        X_train[numeric]
        .corrwith(y_train)
        .abs()
        .sort_values(ascending=False)
        .head(n_features)
    )
    columns = list(strongest.index)

    rows = []
    for degree in degrees:
        model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            LinearRegression(),
        )
        model.fit(X_train[columns], y_train)
        train_r2, train_rmse = _score(model, X_train[columns], y_train)
        test_r2, test_rmse = _score(model, X_test[columns], y_test)
        n_terms = model.named_steps["polynomialfeatures"].n_output_features_
        rows.append(
            {
                "Degree": degree,
                "Terms": n_terms,
                "Train R2": train_r2,
                "Test R2": test_r2,
                "Gap (train - test)": train_r2 - test_r2,
                "Train RMSE": train_rmse,
                "Test RMSE": test_rmse,
            }
        )
    return pd.DataFrame(rows)


def format_table(table: pd.DataFrame) -> str:
    """Render the comparison table for a terminal."""
    display = table.copy()
    for column in ("Test R2", "Train R2", "R2 gap"):
        if column in display:
            display[column] = display[column].map(lambda v: f"{v:.4f}")
    for column in ("Test RMSE", "Train RMSE"):
        if column in display:
            display[column] = display[column].map(lambda v: f"{v:,.0f}")
    return display.to_string(index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare six regression families through one shared preprocessing "
            "template on a tabular dataset."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help=(
            "CSV to model. Defaults to the used-car listings in data/. "
            "Run scripts/download_data.py if it is missing."
        ),
    )
    parser.add_argument(
        "--target",
        default=CAR_TARGET,
        help=f"Target column (default: {CAR_TARGET}).",
    )
    parser.add_argument(
        "--random-state", type=int, default=RANDOM_STATE, help="Split seed."
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=20,
        help="One-hot width cap per categorical column (default: 20).",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help=(
            "Skip the used-car specific column cleaning. Use this when "
            "--data points at a different dataset."
        ),
    )
    args = parser.parse_args(argv)

    try:
        raw = read_csv_safe(args.data, verbose=True).frame
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    frame = raw if args.no_clean else clean_car_frame(raw)

    try:
        table, _fitted, split = run_comparison(
            frame,
            args.target,
            random_state=args.random_state,
            max_categories=args.max_categories,
        )
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print()
    print(split.describe())
    print()
    print("=" * 88)
    print("REGRESSION FAMILY COMPARISON  (held-out test set)")
    print("=" * 88)
    print(format_table(table))

    print()
    print("=" * 88)
    print("OVERFITTING DEMO  (polynomial degree sweep)")
    print("=" * 88)
    demo = overfitting_demo(
        frame,
        args.target,
        random_state=args.random_state,
        max_categories=args.max_categories,
    )
    display = demo.copy()
    for column in ("Train R2", "Test R2", "Gap (train - test)"):
        display[column] = display[column].map(lambda v: f"{v:.4f}")
    for column in ("Train RMSE", "Test RMSE"):
        display[column] = display[column].map(lambda v: f"{v:,.0f}")
    print(display.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
