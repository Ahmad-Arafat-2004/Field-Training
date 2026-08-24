"""Reusable preprocessing template with structurally-enforced split hygiene.

The usual advice for avoiding train/test leakage is "remember to fit on the
training set only". That is a discipline problem, and discipline fails: the
one-line ``scaler.fit_transform(X)`` before the split looks identical to the
correct version and produces a better-looking score, so nothing about the run
signals the mistake.

This module removes the opportunity instead of restating the rule:

* :func:`split_dataset` is the only thing that produces a
  :class:`TrainingPartition`, and it de-duplicates *before* it splits.
* :meth:`PreprocessingTemplate.fit` accepts a :class:`TrainingPartition` and
  nothing else. Handing it a bare ``DataFrame`` -- the whole dataset, the test
  frame, anything -- raises ``TypeError`` before a single statistic is
  computed.
* :class:`TrainingPartition` cannot be constructed directly; its ``__init__``
  demands a module-private token that only :func:`split_dataset` holds.
* A fitted template refuses to re-fit, so a second ``fit`` call on the test
  partition cannot silently overwrite the training statistics.

The result is that the leaky version of the code does not typecheck at
runtime, rather than running fine and quietly reporting a better number.

Typical use::

    split = split_dataset(X, y, test_size=0.2, random_state=42, stratify=y)
    prep = PreprocessingTemplate()
    X_train = prep.fit_transform(split.train)     # training partition only
    X_test = prep.transform(split.test_X)         # reuses fitted statistics
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

__all__ = [
    "AlreadyFittedError",
    "DataSplit",
    "NotFittedError",
    "PreprocessingTemplate",
    "TrainingPartition",
    "describe_unseen_categories",
    "split_dataset",
]


class NotFittedError(RuntimeError):
    """Raised when transform() is called before fit()."""


class AlreadyFittedError(RuntimeError):
    """Raised when fit() is called on an already-fitted template."""


# Capability token. Holding a reference to this object is what authorises the
# creation of a TrainingPartition, and only split_dataset() has one. It is
# module-private and never exported.
_SPLIT_TOKEN = object()


@dataclass(frozen=True)
class TrainingPartition:
    """The training half of a split, and the only thing ``fit`` will accept.

    Not constructible directly -- use :func:`split_dataset`. The guard is what
    makes "fit on training data only" a property of the type rather than a
    convention the caller has to remember.
    """

    X: pd.DataFrame
    y: pd.Series | None = None
    _token: Any = None

    def __post_init__(self) -> None:
        if self._token is not _SPLIT_TOKEN:
            raise TypeError(
                "TrainingPartition() cannot be constructed directly -- that "
                "would reintroduce the very mistake this type exists to "
                "prevent. Use split_dataset(X, y, ...) and pass its .train."
            )
        if not isinstance(self.X, pd.DataFrame):
            raise TypeError(f"X must be a DataFrame, got {type(self.X).__name__}.")
        if self.y is not None and len(self.y) != len(self.X):
            raise ValueError(
                f"X has {len(self.X)} rows but y has {len(self.y)}."
            )

    def __len__(self) -> int:
        return len(self.X)


@dataclass(frozen=True)
class DataSplit:
    """Result of :func:`split_dataset`: a guarded train half and a plain test half.

    The test half is deliberately *not* a :class:`TrainingPartition`, so it
    cannot be passed to ``fit`` even by accident.
    """

    train: TrainingPartition
    test_X: pd.DataFrame
    test_y: pd.Series | None = None
    n_duplicates_dropped: int = 0
    n_rows_before_dedupe: int = 0

    @property
    def train_X(self) -> pd.DataFrame:
        return self.train.X

    @property
    def train_y(self) -> pd.Series | None:
        return self.train.y

    def describe(self) -> str:
        lines = [
            f"rows in            : {self.n_rows_before_dedupe:,}",
            f"exact duplicates   : {self.n_duplicates_dropped:,} dropped "
            f"before the split",
            f"train / test       : {len(self.train.X):,} / {len(self.test_X):,}",
        ]
        if self.train.y is not None and self.test_y is not None:
            tr = self.train.y.value_counts(normalize=True).sort_index()
            te = self.test_y.value_counts(normalize=True).sort_index()
            if len(tr) <= 10:
                lines.append("class balance      :")
                for label in tr.index:
                    lines.append(
                        f"    {label!s:<12} train {tr[label]:6.2%}   "
                        f"test {te.get(label, 0.0):6.2%}"
                    )
        return "\n".join(lines)


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray | None = None,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool | pd.Series | np.ndarray | None = None,
    drop_duplicates: bool = True,
    duplicate_subset: str | Sequence[str] | None = None,
    verbose: bool = False,
) -> DataSplit:
    """De-duplicate, then split, and hand back a guarded training partition.

    Ordering is the point: duplicates are dropped *before* the split, never
    after. A message that appears twice in the corpus and survives into the
    split can land one copy in train and the other in test, at which point the
    model is scored on rows it memorised verbatim and the held-out number stops
    measuring generalisation.

    Parameters
    ----------
    X, y:
        Features and (optional) target. `y` is aligned to `X` by position.
    test_size:
        Held-out fraction. Default 0.2.
    random_state:
        Seed, so the split is reproducible across runs. Default 42.
    stratify:
        ``True`` to stratify on `y`, or an explicit array to stratify on.
        ``None``/``False`` for a plain random split. Stratifying matters
        whenever the target is imbalanced -- an unstratified split of a 13%
        positive class can hand the test set a materially different prior than
        the model trained against.
    drop_duplicates:
        Drop exact duplicate rows before splitting. Default True.
    duplicate_subset:
        Restrict the duplicate check to these columns (e.g. just the raw
        message text). Default None = all columns.

    Returns
    -------
    DataSplit
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a DataFrame, got {type(X).__name__}.")

    n_before = len(X)
    frame = X.reset_index(drop=True)
    target: pd.Series | None = None
    if y is not None:
        target = pd.Series(np.asarray(y)).reset_index(drop=True)
        target.name = getattr(y, "name", None) or "target"
        if len(target) != n_before:
            raise ValueError(
                f"X has {n_before} rows but y has {len(target)}."
            )

    n_dropped = 0
    if drop_duplicates:
        keep_mask = ~frame.duplicated(subset=duplicate_subset, keep="first")
        n_dropped = int((~keep_mask).sum())
        if n_dropped:
            frame = frame.loc[keep_mask].reset_index(drop=True)
            if target is not None:
                target = target.loc[keep_mask].reset_index(drop=True)

    strat_on: Any = None
    if stratify is True:
        if target is None:
            raise ValueError("stratify=True requires y.")
        strat_on = target
    elif stratify is not None and stratify is not False:
        strat_on = np.asarray(stratify)
        if len(strat_on) == n_before and n_dropped:
            raise ValueError(
                "An explicit `stratify` array must be aligned to the "
                "de-duplicated frame. Pass stratify=True to stratify on y, "
                "which is realigned for you."
            )

    if target is None:
        train_X, test_X = train_test_split(
            frame,
            test_size=test_size,
            random_state=random_state,
            stratify=strat_on,
        )
        train_y = test_y = None
    else:
        train_X, test_X, train_y, test_y = train_test_split(
            frame,
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=strat_on,
        )

    split = DataSplit(
        train=TrainingPartition(X=train_X, y=train_y, _token=_SPLIT_TOKEN),
        test_X=test_X,
        test_y=test_y,
        n_duplicates_dropped=n_dropped,
        n_rows_before_dedupe=n_before,
    )
    if verbose:
        print(split.describe())
    return split


@dataclass
class PreprocessingTemplate:
    """scikit-learn-style preprocessing: impute -> encode -> scale.

    Fitted state is kept in named attributes so it can be inspected and reused
    (``scaler_.mean_``, ``encoder_.categories_``, ...), which is also what the
    leakage test asserts against.

    Parameters
    ----------
    numeric_strategy:
        Imputation strategy for numeric columns. Median rather than mean by
        default: the numeric columns here (message length, km driven, price)
        are right-skewed, and the mean of a skewed column is pulled toward the
        tail, so imputing with it shifts the centre of the column.
    categorical_strategy:
        Imputation strategy for categorical columns. ``most_frequent`` keeps
        the imputed value inside the observed category set.
    scale:
        Standardise numeric columns. Needed by the distance- and
        magnitude-sensitive estimators (SVR, linear models with regularisation)
        and harmless for the tree-based ones.
    scale_encoded:
        Also standardise the one-hot columns. Off by default: it destroys the
        0/1 reading of an indicator and rescales rare categories into large
        values without making them more informative.
    max_categories:
        Cap on one-hot columns per categorical feature; the rarest levels are
        folded into a single ``infrequent`` column. Guards against a
        high-cardinality free-text column (the used-car ``name`` field has
        ~2,000 distinct values) exploding the design matrix.
    unknown_policy:
        What one-hot does with a category unseen during fit. ``ignore``
        encodes it as all-zeros, which is the honest representation: at
        training time the model was told nothing about that level.
    """

    numeric_strategy: str = "median"
    categorical_strategy: str = "most_frequent"
    scale: bool = True
    scale_encoded: bool = False
    max_categories: int | None = None
    min_frequency: int | float | None = None
    unknown_policy: str = "ignore"
    verbose: bool = False

    # --- fitted state (trailing underscore, sklearn convention) -------------
    numeric_columns_: list[str] = field(default_factory=list, init=False)
    categorical_columns_: list[str] = field(default_factory=list, init=False)
    feature_names_out_: list[str] = field(default_factory=list, init=False)
    numeric_imputer_: SimpleImputer | None = field(default=None, init=False)
    categorical_imputer_: SimpleImputer | None = field(default=None, init=False)
    encoder_: OneHotEncoder | None = field(default=None, init=False)
    scaler_: StandardScaler | None = field(default=None, init=False)
    encoded_scaler_: StandardScaler | None = field(default=None, init=False)
    n_rows_seen_at_fit_: int = field(default=0, init=False)
    _fitted: bool = field(default=False, init=False)

    # ---------------------------------------------------------------- fit ---
    def fit(self, partition: TrainingPartition) -> "PreprocessingTemplate":
        """Fit on the training partition. Refuses anything that is not one.

        The type check is the whole safety mechanism, so it runs first and
        raises rather than coercing: silently accepting a DataFrame here would
        re-open the exact hole the module exists to close.
        """
        if not isinstance(partition, TrainingPartition):
            raise TypeError(
                "fit() takes a TrainingPartition, got "
                f"{type(partition).__name__}. Preprocessing statistics must "
                "come from the training rows alone; fitting on a full frame "
                "or on the test frame leaks held-out information into the "
                "model. Build a split with split_dataset(...) and pass "
                "split.train."
            )
        if self._fitted:
            raise AlreadyFittedError(
                "This template is already fitted. Re-fitting would overwrite "
                "the training statistics -- most often with the test "
                "partition's, which is leakage. Use .clone_unfitted() for a "
                "fresh instance."
            )

        X = partition.X
        if X.empty:
            raise ValueError("Cannot fit on an empty training partition.")

        self.numeric_columns_ = list(
            X.select_dtypes(include=[np.number, "bool"]).columns
        )
        self.categorical_columns_ = [
            c for c in X.columns if c not in self.numeric_columns_
        ]
        self.n_rows_seen_at_fit_ = len(X)

        names: list[str] = []

        if self.numeric_columns_:
            numeric = self._as_numeric(X[self.numeric_columns_])
            # keep_empty_features so an all-NaN column is imputed to a constant
            # rather than silently dropped. A dropped column would change the
            # matrix width between fit and transform and desynchronise the
            # feature names from the data.
            self.numeric_imputer_ = SimpleImputer(
                strategy=self.numeric_strategy, keep_empty_features=True
            )
            imputed = self.numeric_imputer_.fit_transform(numeric)
            if self.scale:
                self.scaler_ = StandardScaler()
                self.scaler_.fit(imputed)
            names.extend(self.numeric_columns_)

        if self.categorical_columns_:
            categorical = X[self.categorical_columns_].astype(object)
            self.categorical_imputer_ = SimpleImputer(
                strategy=self.categorical_strategy,
                missing_values=np.nan,
                keep_empty_features=True,
            )
            imputed_cat = self.categorical_imputer_.fit_transform(categorical)
            self.encoder_ = OneHotEncoder(
                handle_unknown=self.unknown_policy,
                sparse_output=False,
                max_categories=self.max_categories,
                min_frequency=self.min_frequency,
                dtype=np.float64,
            )
            self.encoder_.fit(imputed_cat)
            if self.scale_encoded:
                # Fitted here, on the training rows, for the same reason the
                # numeric scaler is: a scaler fitted inside transform() would
                # derive its mean from whatever frame it was handed, which for
                # the test half is textbook leakage.
                self.encoded_scaler_ = StandardScaler().fit(
                    self.encoder_.transform(imputed_cat)
                )
            names.extend(
                self.encoder_.get_feature_names_out(self.categorical_columns_)
            )

        self.feature_names_out_ = names
        self._fitted = True
        if self.verbose:
            print(self.summary())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted statistics to any frame (typically the test half)."""
        self._check_fitted()
        if isinstance(X, TrainingPartition):
            raise TypeError(
                "transform() takes a DataFrame -- pass partition.X, not the "
                "partition itself."
            )
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a DataFrame, got {type(X).__name__}.")

        X = self._align_columns(X)
        if X.empty:
            # sklearn's imputers reject a zero-row array, but "nothing to
            # transform" is a legitimate input (an empty filtered batch), and
            # the caller still needs the fitted schema back.
            return pd.DataFrame(
                np.empty((0, len(self.feature_names_out_)), dtype=np.float64),
                columns=self.feature_names_out_,
                index=X.index,
            )

        blocks: list[np.ndarray] = []

        if self.numeric_columns_:
            numeric = self._as_numeric(X[self.numeric_columns_])
            assert self.numeric_imputer_ is not None
            block = self.numeric_imputer_.transform(numeric)
            if self.scaler_ is not None:
                block = self.scaler_.transform(block)
            blocks.append(block)

        if self.categorical_columns_:
            categorical = X[self.categorical_columns_].astype(object)
            assert self.categorical_imputer_ is not None
            assert self.encoder_ is not None
            imputed = self.categorical_imputer_.transform(categorical)
            encoded = self.encoder_.transform(imputed)
            if self.encoded_scaler_ is not None:
                encoded = self.encoded_scaler_.transform(encoded)
            blocks.append(encoded)

        matrix = (
            np.hstack(blocks)
            if blocks
            else np.empty((len(X), 0), dtype=np.float64)
        )
        return pd.DataFrame(
            matrix, columns=self.feature_names_out_, index=X.index
        )

    def fit_transform(self, partition: TrainingPartition) -> pd.DataFrame:
        """Fit on the training partition and transform it in one step.

        Note the signature: unlike sklearn's, this cannot be handed a full
        dataset, because ``TrainingPartition`` is the only accepted input.
        """
        return self.fit(partition).transform(partition.X)

    # ------------------------------------------------------------ helpers ---
    def fit_on_split(
        self, split: DataSplit
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None, pd.Series | None]:
        """Convenience: fit on `split.train`, transform both halves.

        Returns ``(X_train, X_test, y_train, y_test)``. This is the shape the
        modelling code wants and it is impossible to get the order wrong.
        """
        X_train = self.fit_transform(split.train)
        X_test = self.transform(split.test_X)
        return X_train, X_test, split.train.y, split.test_y

    def clone_unfitted(self) -> "PreprocessingTemplate":
        """A fresh instance with the same configuration and no fitted state."""
        return PreprocessingTemplate(
            numeric_strategy=self.numeric_strategy,
            categorical_strategy=self.categorical_strategy,
            scale=self.scale,
            scale_encoded=self.scale_encoded,
            max_categories=self.max_categories,
            min_frequency=self.min_frequency,
            unknown_policy=self.unknown_policy,
            verbose=self.verbose,
        )

    def get_feature_names_out(self) -> list[str]:
        self._check_fitted()
        return list(self.feature_names_out_)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def summary(self) -> str:
        """Readable dump of what was fitted and from how many rows."""
        if not self._fitted:
            return "PreprocessingTemplate(unfitted)"
        lines = [
            "PreprocessingTemplate fitted on "
            f"{self.n_rows_seen_at_fit_:,} training rows",
            f"  numeric ({len(self.numeric_columns_)}): "
            f"{', '.join(self.numeric_columns_) or '-'}",
            f"  categorical ({len(self.categorical_columns_)}): "
            f"{', '.join(self.categorical_columns_) or '-'}",
            f"  output columns: {len(self.feature_names_out_)}",
        ]
        if self.scaler_ is not None:
            lines.append(
                "  scaler means (train only): "
                + ", ".join(
                    f"{c}={m:.4g}"
                    for c, m in list(
                        zip(self.numeric_columns_, self.scaler_.mean_)
                    )[:6]
                )
                + (" ..." if len(self.numeric_columns_) > 6 else "")
            )
        if self.encoder_ is not None:
            total = sum(len(c) for c in self.encoder_.categories_)
            lines.append(f"  categories learned: {total}")
        return "\n".join(lines)

    # ----------------------------------------------------------- internal ---
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError(
                "This PreprocessingTemplate is not fitted yet. Call "
                "fit(split.train) or fit_transform(split.train) first."
            )

    @staticmethod
    def _as_numeric(frame: pd.DataFrame) -> pd.DataFrame:
        """Coerce to float, turning unparseable entries into NaN for the imputer.

        A column that was numeric in train can arrive as object in test (a
        stray ``"n/a"`` in one row is enough). Coercing keeps that row alive as
        a missing value instead of failing the whole transform.
        """
        out = frame.apply(pd.to_numeric, errors="coerce")
        return out.astype(np.float64)

    def _align_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reorder to the fitted schema; error on missing, drop extras."""
        expected = self.numeric_columns_ + self.categorical_columns_
        missing = [c for c in expected if c not in X.columns]
        if missing:
            raise KeyError(
                f"Frame is missing columns seen at fit time: {missing}. "
                f"Expected {expected}."
            )
        extra = [c for c in X.columns if c not in expected]
        if extra:
            warnings.warn(
                f"Dropping {len(extra)} column(s) not seen at fit time: "
                f"{extra[:8]}{' ...' if len(extra) > 8 else ''}",
                stacklevel=3,
            )
        return X[expected]


def describe_unseen_categories(
    template: PreprocessingTemplate, X: pd.DataFrame
) -> dict[str, list[Any]]:
    """List, per categorical column, the values in `X` that fit() never saw.

    Encoding them as all-zeros is the right default, but it is silent, so this
    exists to make the silence inspectable.
    """
    template._check_fitted()
    if template.encoder_ is None:
        return {}
    out: dict[str, list[Any]] = {}
    for col, known in zip(
        template.categorical_columns_, template.encoder_.categories_
    ):
        if col not in X.columns:
            continue
        seen = set(known.tolist())
        unseen = sorted(
            {v for v in X[col].dropna().unique().tolist() if v not in seen},
            key=str,
        )
        if unseen:
            out[col] = unseen
    return out
