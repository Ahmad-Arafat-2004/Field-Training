"""Regenerate the figures embedded in README.md.

    python scripts/export_figures.py

The notebooks store their charts as base64 inside the .ipynb, which markdown
cannot reference, so the README needs the same figures as committed files.
Rather than exporting stale copies out of the notebooks, this re-runs the
pipelines and re-plots -- so `assets/` can never silently drift from the
results the notebooks actually produce.

Figures are written to assets/ and are committed; they are small and the
README is unreadable without them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # no display needed, and none available in CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from src.features.car_features import CAR_TARGET, clean_car_frame
from src.features.spam_features import FEATURE_ORDER, build_feature_frame
from src.models.regression_comparison import (
    SIMPLE_FEATURE,
    build_estimators,
    evaluate,
    load_cars,
    overfitting_demo,
    prepare_split,
)
from src.preprocessing.template import PreprocessingTemplate, split_dataset
from src.utils.loader import read_csv_safe
from src.utils.plotstyle import (
    CLASS_COLORS,
    INK_SECONDARY,
    SERIES,
    annotate_bars,
    apply_style,
    despine,
)

ASSETS = ROOT / "assets"
RANDOM_STATE = 42
DPI = 130


def save(fig, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------- project 1 ---
def spam_pipeline():
    """Rebuild the notebook-02 pipeline and return everything the plots need."""
    sms = read_csv_safe(
        ROOT / "data" / "SMSSpamCollection.tsv",
        sep="\t", header=None, names=["label", "message"], quoting=3, verbose=False,
    ).frame

    features = build_feature_frame(sms["message"])
    frame = pd.concat([sms[["label", "message"]], features], axis=1)
    y = pd.Series(LabelEncoder().fit_transform(frame["label"]), name="is_spam")

    split = split_dataset(
        frame, y,
        test_size=0.2, random_state=RANDOM_STATE, stratify=True,
        drop_duplicates=True, duplicate_subset=["label", "message"],
        keep_columns=list(FEATURE_ORDER),
    )
    prep = PreprocessingTemplate().fit(split.train)
    X_train = prep.transform(split.train.X)
    X_test = prep.transform(split.test_X)

    tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    forest = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    )
    tree.fit(X_train, split.train.y)
    forest.fit(X_train, split.train.y)
    return sms, split, X_test, tree, forest


def fig_spam_eda(sms: pd.DataFrame) -> None:
    counts = sms["label"].value_counts()
    lengths = sms.assign(char_length=sms["message"].str.len())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))

    order = ["ham", "spam"]
    axes[0].bar(order, [counts[k] for k in order],
                color=[CLASS_COLORS[k] for k in order], width=0.5)
    annotate_bars(axes[0], fmt="{:,.0f}")
    axes[0].set_title("Class balance — 86.6% is one class")
    axes[0].set_ylabel("messages")
    axes[0].grid(axis="x", visible=False)
    despine(axes[0])

    for label in order:
        sns.kdeplot(
            lengths.loc[lengths.label == label, "char_length"].clip(upper=350),
            ax=axes[1], label=label, color=CLASS_COLORS[label],
            fill=True, alpha=0.20, linewidth=2, clip=(0, 350),
        )
    axes[1].set_title("Message length by class (density)")
    axes[1].set_xlabel("characters")
    axes[1].set_ylabel("density")
    axes[1].legend(title=None)
    despine(axes[1])

    fig.tight_layout()
    save(fig, "spam_eda.png")


def fig_spam_confusion(split, X_test, tree, forest) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
    for ax, (name, model) in zip(
        axes, [("Decision tree", tree), ("Random forest (200 trees)", forest)]
    ):
        cm = confusion_matrix(split.test_y, model.predict(X_test))
        sns.heatmap(
            cm, annot=True, fmt=",d", cbar=False, ax=ax,
            cmap=sns.light_palette(SERIES[0], as_cmap=True),
            xticklabels=["ham", "spam"], yticklabels=["ham", "spam"],
            annot_kws={"fontsize": 12}, linewidths=2, linecolor="white",
        )
        _tn, fp, fn, _tp = cm.ravel()
        ax.set_title(f"{name}\n{fn} spam missed · {fp} ham misfiled")
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.grid(False)
    fig.tight_layout()
    save(fig, "spam_confusion_matrices.png")


def fig_spam_importance(forest, columns) -> None:
    importances = pd.Series(forest.feature_importances_, index=columns).sort_values()
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.barh(importances.index, importances.to_numpy(), color=SERIES[0], height=0.62)
    annotate_bars(ax, fmt="{:.3f}", horizontal=True)
    ax.set_title("Random forest feature importance (mean decrease in impurity)")
    ax.set_xlabel("importance")
    ax.grid(axis="y", visible=False)
    despine(ax)
    fig.tight_layout()
    save(fig, "spam_feature_importance.png")


def fig_spam_pr_curve(split, X_test, forest) -> None:
    probabilities = forest.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(
        split.test_y, probabilities, pos_label=1
    )
    ap = average_precision_score(split.test_y, probabilities, pos_label=1)

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.plot(recall, precision, color=SERIES[0], linewidth=2)

    default = int(np.argmin(np.abs(thresholds - 0.5)))
    ax.scatter(recall[default], precision[default], s=75, color=SERIES[1],
               zorder=3, edgecolor="white", linewidth=1.5)
    ax.annotate(
        f"  threshold 0.5\n  precision {precision[default]:.3f}\n"
        f"  recall {recall[default]:.3f}",
        (recall[default], precision[default]), fontsize=9,
        color=INK_SECONDARY, va="top",
    )
    baseline = float(split.test_y.mean())
    ax.axhline(baseline, color="#c3c2b7", linestyle="--", linewidth=1)
    ax.annotate(f"random classifier ({baseline:.3f})", (0.02, baseline + 0.02),
                fontsize=8.5, color="#898781")

    ax.set_title(f"Precision–recall, spam class · average precision {ap:.3f}")
    ax.set_xlabel("recall (share of spam caught)")
    ax.set_ylabel("precision (share of flags that are spam)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    despine(ax)
    fig.tight_layout()
    save(fig, "spam_precision_recall.png")


# ---------------------------------------------------------------- project 2 ---
def fig_regression_comparison(table: pd.DataFrame) -> None:
    ordered = table.sort_values("Test R2")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))

    axes[0].barh(ordered["Model"], ordered["Test R2"], color=SERIES[0], height=0.62)
    annotate_bars(axes[0], fmt="{:.3f}", horizontal=True)
    axes[0].set_title("Held-out R² — higher is better")
    axes[0].set_xlabel("R²")
    axes[0].set_xlim(0, 1.08)
    axes[0].grid(axis="y", visible=False)
    despine(axes[0])

    axes[1].barh(ordered["Model"], ordered["Test RMSE"] / 1000,
                 color=SERIES[1], height=0.62)
    annotate_bars(axes[1], fmt="{:,.0f}k", horizontal=True)
    axes[1].set_title("Held-out RMSE — lower is better")
    axes[1].set_xlabel("RMSE (thousand ₹)")
    axes[1].set_yticklabels([])
    axes[1].tick_params(left=False)
    axes[1].grid(axis="y", visible=False)
    despine(axes[1])

    fig.tight_layout()
    save(fig, "regression_comparison.png")


def fig_overfitting(demo: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(demo["Degree"], demo["Train R2"], marker="o", color=SERIES[0],
            label="train R²")
    ax.plot(demo["Degree"], demo["Test R2"], marker="o", color=SERIES[1],
            label="test R² (held out)")
    ax.fill_between(demo["Degree"], demo["Test R2"], demo["Train R2"],
                    color=SERIES[1], alpha=0.10)

    for _, row in demo.iterrows():
        ax.annotate(f"{row['Train R2']:.3f}", (row["Degree"], row["Train R2"]),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8.5, color=INK_SECONDARY)
        ax.annotate(f"{row['Test R2']:.3f}", (row["Degree"], row["Test R2"]),
                    textcoords="offset points", xytext=(0, -16), ha="center",
                    fontsize=8.5, color=INK_SECONDARY)

    peak = demo.loc[demo["Test R2"].idxmax()]
    ax.axvline(peak["Degree"], color="#c3c2b7", linestyle="--", linewidth=1)
    ax.annotate(f"test R² peaks at degree {int(peak['Degree'])}",
                (peak["Degree"] - 0.04, 0.985), fontsize=9,
                color=INK_SECONDARY, ha="right", va="top")

    ax.set_title("Polynomial degree sweep — the gap is the overfitting")
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("R²")
    ax.set_xticks(demo["Degree"])
    ax.set_ylim(0.32, 1.02)
    ax.legend(loc="lower left")
    despine(ax)
    fig.tight_layout()
    save(fig, "regression_overfitting.png")


def fig_audit(sms: pd.DataFrame, split, X_test, forest) -> None:
    """Notebook 04: learning curve and resampling comparison.

    Both come from the SMS corpus alone, so this runs without the optional
    ~734 MB of audit corpora.
    """
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.metrics import f1_score

    from src.analysis.audit import learning_curve_f1

    prep = PreprocessingTemplate().fit(split.train)
    X_train, y_train = prep.transform(split.train.X), split.train.y
    y_test = split.test_y

    curve = learning_curve_f1(X_train, y_train, X_test, y_test, seeds=5)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))

    axes[0].plot(curve["rows"], curve["f1_mean"], marker="o", color=SERIES[0])
    axes[0].fill_between(
        curve["rows"],
        curve["f1_mean"] - curve["f1_std"],
        curve["f1_mean"] + curve["f1_std"],
        color=SERIES[0], alpha=0.15,
    )
    for _, r in curve.iterrows():
        axes[0].annotate(
            f"{r['f1_mean']:.3f}", (r["rows"], r["f1_mean"]),
            textcoords="offset points", xytext=(0, 9), ha="center",
            fontsize=8.5, color=INK_SECONDARY,
        )
    axes[0].set_title("Learning curve — flat, so more rows add little")
    axes[0].set_xlabel("training rows")
    axes[0].set_ylabel("test F1 (spam)")
    axes[0].set_ylim(0.87, 0.97)
    despine(axes[0])

    def score(X, y):
        m = RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ).fit(X, y)
        return f1_score(y_test, m.predict(X_test), pos_label=1, zero_division=0)

    base = score(X_train, y_train)
    rows = [("none (shipped)", base)]
    balanced = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
        class_weight="balanced",
    ).fit(X_train, y_train)
    rows.append((
        "class_weight=balanced",
        f1_score(y_test, balanced.predict(X_test), pos_label=1, zero_division=0),
    ))
    for label, sampler in [
        ("RandomOverSampler", RandomOverSampler(random_state=RANDOM_STATE)),
        ("SMOTE (k=5)", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("RandomUnderSampler", RandomUnderSampler(random_state=RANDOM_STATE)),
    ]:
        Xr, yr = sampler.fit_resample(X_train, y_train)
        rows.append((label, score(Xr, pd.Series(yr))))

    table = pd.DataFrame(rows, columns=["strategy", "f1"]).sort_values("f1")
    axes[1].barh(table["strategy"], table["f1"], color=SERIES[0], height=0.6)
    axes[1].axvline(base, color=SERIES[1], linestyle="--", linewidth=1.5)
    annotate_bars(axes[1], fmt="{:.3f}", horizontal=True)
    # Precise wording: class_weight does beat the shipped model, by +0.005.
    # Every strategy that synthesises or discards rows loses.
    axes[1].set_title("Resampling — every synthetic-row strategy loses")
    axes[1].set_xlabel("test F1 (spam)")
    axes[1].set_xlim(0.85, 1.0)
    axes[1].grid(axis="y", visible=False)
    despine(axes[1])

    fig.tight_layout()
    save(fig, "audit_learning_curve_and_resampling.png")


def main() -> int:
    apply_style()

    sms_path = ROOT / "data" / "SMSSpamCollection.tsv"
    cars_path = ROOT / "data" / "car_details_v3.csv"
    for path in (sms_path, cars_path):
        if not path.is_file():
            print(f"error: {path} not found. Run scripts/download_data.py first.",
                  file=sys.stderr)
            return 1

    print("Project 1 — SMS spam")
    sms, split, X_test, tree, forest = spam_pipeline()
    fig_spam_eda(sms)
    fig_spam_confusion(split, X_test, tree, forest)
    fig_spam_importance(forest, list(FEATURE_ORDER))
    fig_spam_pr_curve(split, X_test, forest)

    print("Notebook 04 — dataset audit")
    fig_audit(sms, split, X_test, forest)

    print("Project 2 — regression comparison")
    cars = load_cars(cars_path)
    car_split, template, X_train_c, X_test_c = prepare_split(
        cars, CAR_TARGET, random_state=RANDOM_STATE, max_categories=20
    )
    rows = []
    for name, model in build_estimators(
        template.numeric_columns_, random_state=RANDOM_STATE
    ).items():
        if name.startswith("Simple linear"):
            tr, te = X_train_c[[SIMPLE_FEATURE]], X_test_c[[SIMPLE_FEATURE]]
        else:
            tr, te = X_train_c, X_test_c
        result = evaluate(name, model, tr, te, car_split.train.y, car_split.test_y)
        rows.append({"Model": result.name, "Test R2": result.r2,
                     "Test RMSE": result.rmse})
    fig_regression_comparison(pd.DataFrame(rows))
    fig_overfitting(
        overfitting_demo(cars, CAR_TARGET, degrees=(1, 2, 3, 4),
                         random_state=RANDOM_STATE, n_features=6,
                         max_categories=20)
    )

    print(f"\nDone. {len(list(ASSETS.glob('*.png')))} figures in {ASSETS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
