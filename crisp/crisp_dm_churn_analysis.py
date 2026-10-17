
"""
Bank Churn AI
CRISP-DM Machine Learning Analysis

This script performs an end-to-end CRISP-DM style analysis of the
customer churn dataset.

Phases covered:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modelling
5. Evaluation
6. Results / artefacts

The script:
* profiles the dataset
* checks missing values and duplicates
* analyses categorical variables
* detects numerical outliers using the IQR rule
* analyses correlations
* creates exploratory charts
* compares Logistic Regression, Random Forest, SVM,
  Neural Network and Gaussian Naive Bayes
* uses stratified 10-fold cross-validation
* evaluates the selected model on an untouched test set
* creates ROC curves and confusion matrices
* creates model comparison tables and charts
* saves the best fitted model for possible use by the application

Run:
    python crisp_dm_churn_analysis.py

Optional:
    python crisp_dm_churn_analysis.py --data "path/to/customer_churn.csv"
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "churn"
ID_COLUMN = "customer_id"
CV_FOLDS = 10
TEST_SIZE = 0.20

# The report supplied with the project uses AUC, Recall, F1 and MCC
# as the main evaluation metrics. We preserve that logic here.
PRIMARY_METRICS = ["roc_auc", "recall", "f1", "mcc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CRISP-DM analysis for the Bank Churn dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "customer_churn.csv",
        help="Path to the churn CSV file.",
    )
    parser.add_argument(
        "--output",
        default="crisp_dm_output",
        help="Directory where reports, charts and the model will be saved.",
    )
    return parser.parse_args()


def setup_output(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "root": output_dir,
        "eda": output_dir / "01_data_understanding",
        "prep": output_dir / "02_data_preparation",
        "models": output_dir / "03_modeling",
        "eval": output_dir / "04_evaluation",
        "model": output_dir / "05_model_artifact",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def save_json(data: dict, path: Path) -> None:
    def convert(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    serializable = json.loads(json.dumps(data, default=convert))
    path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path.resolve()}"
        )

    df = pd.read_csv(data_path)

    if TARGET not in df.columns:
        raise ValueError(
            f"The dataset must contain the target column '{TARGET}'."
        )

    return df


# ---------------------------------------------------------------------
# CRISP-DM 1. BUSINESS UNDERSTANDING
# ---------------------------------------------------------------------

def business_understanding() -> None:
    print("\n" + "=" * 80)
    print("CRISP-DM 1. BUSINESS UNDERSTANDING")
    print("=" * 80)

    print(
        """
Objective:
Predict whether a bank customer is likely to churn.

Business problem:
Identify customers with a higher probability of leaving the bank so that
retention actions can be considered.

Machine Learning problem:
Supervised binary classification.

Target:
churn = 0 -> customer does not churn
churn = 1 -> customer churns

Model evaluation:
AUC, Recall, F1, MCC and Accuracy, complemented by ROC curves and
confusion matrices.

Model selection:
The models are compared using stratified 10-fold cross-validation.
MCC, F1, Recall and AUC are considered together rather than relying
on Accuracy alone.
"""
    )


# ---------------------------------------------------------------------
# CRISP-DM 2. DATA UNDERSTANDING
# ---------------------------------------------------------------------

def data_understanding(df: pd.DataFrame, paths: dict[str, Path]) -> dict:
    print("\n" + "=" * 80)
    print("CRISP-DM 2. DATA UNDERSTANDING")
    print("=" * 80)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Duplicates: {df.duplicated().sum():,}")

    # Basic profile
    profile = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "non_null": [df[c].notna().sum() for c in df.columns],
            "missing": [df[c].isna().sum() for c in df.columns],
            "missing_pct": [
                df[c].isna().mean() * 100 for c in df.columns
            ],
            "unique": [df[c].nunique(dropna=False) for c in df.columns],
        }
    )

    profile.to_csv(
        paths["eda"] / "data_profile.csv",
        index=False,
    )

    # Missing values
    missing = (
        df.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
    )
    missing["missing_pct"] = (
        missing["missing_count"] / len(df) * 100
    )
    missing.to_csv(
        paths["eda"] / "missing_values.csv"
    )

    if missing["missing_count"].sum() > 0:
        missing_nonzero = missing[missing["missing_count"] > 0]
        plt.figure(figsize=(10, 5))
        missing_nonzero["missing_pct"].sort_values().plot(kind="barh")
        plt.title("Missing Values by Variable")
        plt.xlabel("Missing values (%)")
        plt.tight_layout()
        plt.savefig(
            paths["eda"] / "missing_values.png",
            dpi=160,
        )
        plt.close()
    else:
        plt.figure(figsize=(8, 4))
        plt.text(
            0.5,
            0.5,
            "No missing values detected",
            ha="center",
            va="center",
            fontsize=18,
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            paths["eda"] / "missing_values.png",
            dpi=160,
        )
        plt.close()

    # Target distribution
    target_counts = df[TARGET].value_counts().sort_index()
    target_pct = (
        df[TARGET].value_counts(normalize=True)
        .sort_index()
        .mul(100)
    )

    target_table = pd.DataFrame(
        {
            "count": target_counts,
            "percentage": target_pct.round(2),
        }
    )
    target_table.to_csv(
        paths["eda"] / "target_distribution.csv"
    )

    plt.figure(figsize=(7, 5))
    labels = ["No churn (0)", "Churn (1)"]
    values = [
        target_counts.get(0, 0),
        target_counts.get(1, 0),
    ]
    bars = plt.bar(labels, values)
    plt.title("Target Distribution")
    plt.ylabel("Number of customers")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(
        paths["eda"] / "target_distribution.png",
        dpi=160,
    )
    plt.close()

    # Numeric descriptive statistics
    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    numeric_stats = (
        df[numeric_cols]
        .describe()
        .T
        .reset_index()
        .rename(columns={"index": "variable"})
    )

    numeric_stats.to_csv(
        paths["eda"] / "numeric_descriptive_statistics.csv",
        index=False,
    )

    # Numeric distributions
    for column in numeric_cols:
        if column == TARGET:
            continue

        plt.figure(figsize=(8, 5))
        plt.hist(
            df[column].dropna(),
            bins=30,
            edgecolor="black",
        )
        plt.title(f"Distribution: {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(
            paths["eda"] / f"distribution_{column}.png",
            dpi=160,
        )
        plt.close()

    # Categorical distributions
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    for column in categorical_cols:
        if column == ID_COLUMN:
            continue

        counts = (
            df[column]
            .value_counts(dropna=False)
            .sort_values(ascending=False)
        )

        counts.to_csv(
            paths["eda"] / f"category_counts_{column}.csv"
        )

        plt.figure(figsize=(8, 5))
        counts.plot(kind="bar")
        plt.title(f"Distribution: {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(
            paths["eda"] / f"distribution_{column}.png",
            dpi=160,
        )
        plt.close()

    # Correlation matrix
    correlation = df[numeric_cols].corr(numeric_only=True)
    correlation.to_csv(
        paths["eda"] / "correlation_matrix.csv"
    )

    plt.figure(figsize=(11, 9))
    matrix = plt.imshow(
        correlation,
        cmap="coolwarm",
        aspect="auto",
        vmin=-1,
        vmax=1,
    )
    plt.colorbar(matrix, label="Correlation")
    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=70,
        ha="right",
    )
    plt.yticks(
        range(len(correlation.index)),
        correlation.index,
    )
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(
        paths["eda"] / "correlation_matrix.png",
        dpi=160,
    )
    plt.close()

    # Strong correlations
    corr_pairs = []
    for i, col1 in enumerate(correlation.columns):
        for j, col2 in enumerate(correlation.columns):
            if j <= i:
                continue

            value = correlation.loc[col1, col2]

            if abs(value) >= 0.80:
                corr_pairs.append(
                    {
                        "variable_1": col1,
                        "variable_2": col2,
                        "correlation": value,
                    }
                )

    strong_corr = pd.DataFrame(corr_pairs)
    strong_corr.to_csv(
        paths["eda"] / "strong_correlations.csv",
        index=False,
    )

    # Dataset summary
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "target_positive_count": int((df[TARGET] == 1).sum()),
        "target_negative_count": int((df[TARGET] == 0).sum()),
        "target_positive_pct": float(
            (df[TARGET] == 1).mean() * 100
        ),
        "target_negative_pct": float(
            (df[TARGET] == 0).mean() * 100
        ),
        "numeric_variables": [
            c for c in numeric_cols if c != TARGET
        ],
        "categorical_variables": [
            c for c in categorical_cols if c != ID_COLUMN
        ],
    }

    save_json(
        summary,
        paths["eda"] / "data_summary.json",
    )

    print("\nTarget distribution:")
    print(target_table)

    print("\nMissing values:")
    print(missing[missing["missing_count"] > 0])

    if strong_corr.empty:
        print("\nNo correlations >= |0.80| were detected.")
    else:
        print("\nStrong correlations:")
        print(strong_corr)

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "profile": profile,
        "correlation": correlation,
    }


# ---------------------------------------------------------------------
# CRISP-DM 3. DATA PREPARATION
# ---------------------------------------------------------------------

def outlier_analysis(
    df: pd.DataFrame,
    numeric_cols: list[str],
    paths: dict[str, Path],
) -> pd.DataFrame:
    records = []

    for column in numeric_cols:
        if column == TARGET:
            continue

        series = df[column].dropna()

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (series < lower) | (series > upper)

        records.append(
            {
                "variable": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": int(mask.sum()),
                "outlier_pct": float(mask.mean() * 100),
            }
        )

        # Boxplot
        plt.figure(figsize=(8, 4))
        plt.boxplot(series, vert=False)
        plt.title(f"Outlier Analysis: {column}")
        plt.xlabel(column)
        plt.tight_layout()
        plt.savefig(
            paths["prep"] / f"boxplot_{column}.png",
            dpi=160,
        )
        plt.close()

    result = pd.DataFrame(records)

    result.to_csv(
        paths["prep"] / "outlier_analysis.csv",
        index=False,
    )

    return result


def data_preparation(
    df: pd.DataFrame,
    metadata: dict,
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], pd.DataFrame]:

    print("\n" + "=" * 80)
    print("CRISP-DM 3. DATA PREPARATION")
    print("=" * 80)

    data = df.copy()

    # Duplicate analysis
    duplicate_count = int(data.duplicated().sum())

    if duplicate_count > 0:
        print(
            f"Removing {duplicate_count} duplicated rows."
        )
        data = data.drop_duplicates().reset_index(drop=True)
    else:
        print("No duplicate rows found.")

    # ID is not a predictive feature.
    if ID_COLUMN in data.columns:
        print(
            f"Excluding '{ID_COLUMN}' from the predictive features "
            "because it is an identifier."
        )

    X = data.drop(
        columns=[TARGET, ID_COLUMN],
        errors="ignore",
    )
    y = data[TARGET].astype(int)

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Outlier analysis
    outliers = outlier_analysis(
        data,
        metadata["numeric_cols"],
        paths,
    )

    # Prepared data is saved for transparency.
    prepared = X.copy()
    prepared[TARGET] = y

    prepared.to_csv(
        paths["prep"] / "prepared_dataset.csv",
        index=False,
    )

    # Text report
    report_lines = [
        "DATA PREPARATION DECISIONS",
        "",
        f"Original rows: {len(df)}",
        f"Original columns: {len(df.columns)}",
        f"Duplicates found: {duplicate_count}",
        f"Rows after duplicate handling: {len(data)}",
        f"Missing cells before modelling: {int(data.isna().sum().sum())}",
        "",
        "Identifier excluded from modelling:",
        f"  {ID_COLUMN}",
        "",
        "Target:",
        f"  {TARGET}",
        "",
        "Numeric features:",
        *[f"  {c}" for c in numeric_features],
        "",
        "Categorical features:",
        *[f"  {c}" for c in categorical_features],
        "",
        "Outliers:",
        "  Outliers were identified using the IQR rule.",
        "  No observations are automatically removed because an extreme",
        "  customer value may represent a legitimate banking behaviour.",
        "  The model pipeline handles preprocessing without changing the",
        "  original dataset.",
    ]

    (paths["prep"] / "preparation_decisions.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\nOutlier summary:")
    print(
        outliers[
            [
                "variable",
                "outlier_count",
                "outlier_pct",
            ]
        ].sort_values(
            "outlier_pct",
            ascending=False,
        )
    )

    return (
        X,
        y,
        numeric_features,
        categorical_features,
        outliers,
    )


# ---------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------

def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def create_models(
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Pipeline]:

    def pipeline(model):
        return Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        numeric_features,
                        categorical_features,
                    ),
                ),
                ("model", model),
            ]
        )

    return {
        "Logistic Regression": pipeline(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "Random Forest": pipeline(
            RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "SVM": pipeline(
            SVC(
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "Neural Network": pipeline(
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=1000,
                early_stopping=True,
                random_state=RANDOM_STATE,
            )
        ),
        "Naive Bayes": pipeline(
            GaussianNB()
        ),
    }


# ---------------------------------------------------------------------
# CRISP-DM 4. MODELLING
# ---------------------------------------------------------------------

def modelling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: dict[str, Pipeline],
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:

    print("\n" + "=" * 80)
    print("CRISP-DM 4. MODELLING")
    print("=" * 80)

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "auc": "roc_auc",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "accuracy": "accuracy",
        "mcc": "matthews_corrcoef",
    }

    rows = []

    fitted_models = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        result = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        row = {
            "model": name,
        }

        for metric in scoring:
            row[f"{metric}_mean"] = result[
                f"test_{metric}"
            ].mean()
            row[f"{metric}_std"] = result[
                f"test_{metric}"
            ].std()

        rows.append(row)

        fitted = clone(model)
        fitted.fit(X_train, y_train)
        fitted_models[name] = fitted

    results = pd.DataFrame(rows)

    # Primary selection criterion: MCC.
    # Tie breakers: F1, Recall and AUC.
    results = results.sort_values(
        by=[
            "mcc_mean",
            "f1_mean",
            "recall_mean",
            "auc_mean",
        ],
        ascending=False,
    ).reset_index(drop=True)

    results["rank"] = np.arange(1, len(results) + 1)

    results.to_csv(
        paths["models"] / "cross_validation_results.csv",
        index=False,
    )

    print("\nCross-validation results:")
    print(
        results[
            [
                "rank",
                "model",
                "auc_mean",
                "recall_mean",
                "precision_mean",
                "f1_mean",
                "mcc_mean",
                "accuracy_mean",
            ]
        ].round(4).to_string(index=False)
    )

    # Comparison chart
    metric_labels = {
        "auc_mean": "AUC",
        "recall_mean": "Recall",
        "f1_mean": "F1",
        "mcc_mean": "MCC",
        "accuracy_mean": "Accuracy",
    }

    chart_data = results.set_index("model")[
        list(metric_labels.keys())
    ].rename(columns=metric_labels)

    ax = chart_data.plot(
        kind="bar",
        figsize=(12, 6),
    )

    ax.set_title(
        "Model Comparison, Stratified 10-Fold Cross-Validation"
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(
        paths["models"] / "model_comparison.png",
        dpi=160,
    )
    plt.close()

    return results, fitted_models


# ---------------------------------------------------------------------
# CRISP-DM 5. EVALUATION
# ---------------------------------------------------------------------

def evaluate_on_test_set(
    results: pd.DataFrame,
    fitted_models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    paths: dict[str, Path],
) -> tuple[str, Pipeline, pd.DataFrame]:

    print("\n" + "=" * 80)
    print("CRISP-DM 5. EVALUATION")
    print("=" * 80)

    best_model_name = results.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    print(
        f"\nSelected model based on cross-validation: "
        f"{best_model_name}"
    )

    evaluation_rows = []

    for name, model in fitted_models.items():
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        evaluation_rows.append(
            {
                "model": name,
                "auc": roc_auc_score(
                    y_test,
                    probabilities,
                ),
                "recall": recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "precision": precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "mcc": matthews_corrcoef(
                    y_test,
                    predictions,
                ),
                "accuracy": accuracy_score(
                    y_test,
                    predictions,
                ),
            }
        )

    test_results = pd.DataFrame(evaluation_rows).sort_values(
        by=[
            "mcc",
            "f1",
            "recall",
            "auc",
        ],
        ascending=False,
    )

    test_results.to_csv(
        paths["eval"] / "test_set_results.csv",
        index=False,
    )

    print("\nUntouched test-set results:")
    print(
        test_results.round(4).to_string(index=False)
    )

    # ROC curves for all models
    plt.figure(figsize=(10, 7))

    for name, model in fitted_models.items():
        probabilities = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(
            y_test,
            probabilities,
        )

        RocCurveDisplay.from_predictions(
            y_test,
            probabilities,
            name=f"{name} (AUC={auc:.3f})",
        )

    plt.title("ROC Curves, Test Set")
    plt.tight_layout()
    plt.savefig(
        paths["eval"] / "roc_curves.png",
        dpi=160,
    )
    plt.close()

    # Confusion matrices
    for name, model in fitted_models.items():
        predictions = model.predict(X_test)

        fig, ax = plt.subplots(figsize=(5, 5))

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            display_labels=[
                "No churn",
                "Churn",
            ],
            cmap="Blues",
            ax=ax,
        )

        ax.set_title(f"Confusion Matrix, {name}")
        plt.tight_layout()

        safe_name = (
            name.lower()
            .replace(" ", "_")
        )

        plt.savefig(
            paths["eval"] / f"confusion_matrix_{safe_name}.png",
            dpi=160,
        )
        plt.close()

    # Best model detailed evaluation
    best_probabilities = best_model.predict_proba(
        X_test
    )[:, 1]

    best_predictions = (
        best_probabilities >= 0.5
    ).astype(int)

    best_metrics = {
        "model": best_model_name,
        "test_auc": roc_auc_score(
            y_test,
            best_probabilities,
        ),
        "test_recall": recall_score(
            y_test,
            best_predictions,
            zero_division=0,
        ),
        "test_precision": precision_score(
            y_test,
            best_predictions,
            zero_division=0,
        ),
        "test_f1": f1_score(
            y_test,
            best_predictions,
            zero_division=0,
        ),
        "test_mcc": matthews_corrcoef(
            y_test,
            best_predictions,
        ),
        "test_accuracy": accuracy_score(
            y_test,
            best_predictions,
        ),
    }

    save_json(
        best_metrics,
        paths["eval"] / "best_model_metrics.json",
    )

    # Threshold analysis for the best model
    threshold_rows = []

    for threshold in np.arange(
        0.10,
        0.91,
        0.05,
    ):
        predictions = (
            best_probabilities >= threshold
        ).astype(int)

        threshold_rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "mcc": matthews_corrcoef(
                    y_test,
                    predictions,
                ),
            }
        )

    threshold_df = pd.DataFrame(threshold_rows)

    threshold_df.to_csv(
        paths["eval"] / "threshold_analysis.csv",
        index=False,
    )

    plt.figure(figsize=(10, 6))

    for metric in [
        "precision",
        "recall",
        "f1",
        "mcc",
    ]:
        plt.plot(
            threshold_df["threshold"],
            threshold_df[metric],
            marker="o",
            label=metric.upper(),
        )

    plt.axvline(
        0.50,
        linestyle="--",
        label="Default threshold = 0.50",
    )

    plt.title(
        f"Threshold Analysis, {best_model_name}"
    )
    plt.xlabel("Classification threshold")
    plt.ylabel("Score")
    plt.ylim(-1, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        paths["eval"] / "threshold_analysis.png",
        dpi=160,
    )
    plt.close()

    return best_model_name, best_model, test_results


# ---------------------------------------------------------------------
# MODEL INTERPRETABILITY
# ---------------------------------------------------------------------

def model_interpretability(
    best_model_name: str,
    best_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    paths: dict[str, Path],
) -> None:

    print("\n" + "=" * 80)
    print("MODEL INTERPRETABILITY")
    print("=" * 80)

    # Permutation importance works across different model types
    # and gives importance at the original feature level.
    print(
        f"Calculating permutation importance for {best_model_name}..."
    )

    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values(
        "importance_mean",
        ascending=False,
    )

    importance_df.to_csv(
        paths["eval"] / "permutation_feature_importance.csv",
        index=False,
    )

    top = importance_df.head(10).sort_values(
        "importance_mean"
    )

    plt.figure(figsize=(9, 6))
    plt.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
    )
    plt.title(
        f"Top 10 Feature Importance, {best_model_name}"
    )
    plt.xlabel("Mean decrease in ROC AUC")
    plt.tight_layout()
    plt.savefig(
        paths["eval"] / "feature_importance.png",
        dpi=160,
    )
    plt.close()


# ---------------------------------------------------------------------
# CRISP-DM 6. RESULTS
# ---------------------------------------------------------------------

def create_final_report(
    df: pd.DataFrame,
    results: pd.DataFrame,
    test_results: pd.DataFrame,
    outliers: pd.DataFrame,
    best_model_name: str,
    paths: dict[str, Path],
) -> None:

    print("\n" + "=" * 80)
    print("CRISP-DM 6. RESULTS")
    print("=" * 80)

    best_cv = results.iloc[0]
    best_test = test_results[
        test_results["model"] == best_model_name
    ].iloc[0]

    report = f"""
BANK CHURN AI, CRISP-DM ANALYSIS
================================

DATASET
-------
Rows: {len(df):,}
Columns: {len(df.columns)}
Target: {TARGET}
Target class 0: {(df[TARGET] == 0).sum():,}
Target class 1: {(df[TARGET] == 1).sum():,}

DATA QUALITY
------------
Missing values: {int(df.isna().sum().sum()):,}
Duplicate rows: {int(df.duplicated().sum()):,}

DATA PREPARATION
----------------
The customer_id identifier was excluded from predictive modelling.

Missing value handling is implemented inside the model pipeline.
This prevents data leakage because imputation is fitted only on the
training data of each cross-validation fold.

Categorical variables are one-hot encoded.

Numerical variables are imputed with the median when necessary and
standardised for models that benefit from scaling.

Outliers were analysed using the IQR rule. No observations were
automatically removed because extreme customer values may represent
legitimate banking behaviour.

MODEL COMPARISON
----------------
Models evaluated:
* Logistic Regression
* Random Forest
* SVM
* Neural Network
* Naive Bayes

Validation:
Stratified {CV_FOLDS}-fold cross-validation on the training set.

PRIMARY MODEL SELECTION
-----------------------
The primary ranking criterion is MCC, followed by F1, Recall and AUC.

Selected model:
{best_model_name}

Cross-validation results for selected model:
AUC      = {best_cv['auc_mean']:.4f}
Recall   = {best_cv['recall_mean']:.4f}
F1       = {best_cv['f1_mean']:.4f}
MCC      = {best_cv['mcc_mean']:.4f}
Accuracy = {best_cv['accuracy_mean']:.4f}

FINAL TEST SET
--------------
The final test set was not used during cross-validation or model
selection.

Selected model:
{best_model_name}

Test AUC      = {best_test['auc']:.4f}
Test Recall   = {best_test['recall']:.4f}
Test Precision= {best_test['precision']:.4f}
Test F1       = {best_test['f1']:.4f}
Test MCC      = {best_test['mcc']:.4f}
Test Accuracy = {best_test['accuracy']:.4f}

INTERPRETATION
--------------
The selected model should not be judged using Accuracy alone.
For customer churn, Recall is important because missing a customer
who is likely to churn can reduce the opportunity for a retention
action. MCC and F1 provide additional information about the balance
of the classification results.

The ROC curve, confusion matrix and threshold analysis should be
reviewed before selecting an operational probability threshold.

OUTPUTS
-------
The script generates:

01_data_understanding/
    Data profile
    Missing values analysis
    Target distribution
    Numeric distributions
    Categorical distributions
    Correlation matrix
    Strong correlation report

02_data_preparation/
    Outlier analysis
    Boxplots
    Prepared dataset
    Preparation decisions

03_modeling/
    10-fold cross-validation results
    Model comparison chart

04_evaluation/
    Test set metrics
    ROC curves
    Confusion matrices
    Threshold analysis
    Feature importance

05_model_artifact/
    Final fitted model
"""

    (paths["root"] / "CRISP_DM_FINAL_REPORT.txt").write_text(
        report.strip(),
        encoding="utf-8",
    )


def save_model_artifact(
    best_model_name: str,
    best_model: Pipeline,
    paths: dict[str, Path],
) -> None:

    artifact = {
        "model_name": best_model_name,
        "model": best_model,
        "target": TARGET,
        "threshold": 0.50,
        "random_state": RANDOM_STATE,
    }

    joblib.dump(
        artifact,
        paths["model"] / "best_churn_model.joblib",
    )

    metadata = {
        "model_name": best_model_name,
        "target": TARGET,
        "threshold": 0.50,
        "random_state": RANDOM_STATE,
        "note": (
            "The saved object contains the complete preprocessing "
            "pipeline and fitted model."
        ),
    }

    save_json(
        metadata,
        paths["model"] / "model_metadata.json",
    )


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output)

    paths = setup_output(output_dir)

    print("\n" + "=" * 80)
    print("BANK CHURN AI, CRISP-DM ANALYSIS")
    print("=" * 80)

    print(f"\nDataset: {data_path.resolve()}")
    print(f"Output:  {output_dir.resolve()}")

    business_understanding()

    df = load_data(data_path)

    metadata = data_understanding(
        df,
        paths,
    )

    (
        X,
        y,
        numeric_features,
        categorical_features,
        outliers,
    ) = data_preparation(
        df,
        metadata,
        paths,
    )

    # Holdout test set.
    # Stratification preserves the churn ratio in both partitions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("\nTrain/test split:")
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows:     {len(X_test):,}")

    models = create_models(
        numeric_features,
        categorical_features,
    )

    results, fitted_models = modelling(
        X_train,
        y_train,
        models,
        paths,
    )

    (
        best_model_name,
        best_model,
        test_results,
    ) = evaluate_on_test_set(
        results,
        fitted_models,
        X_test,
        y_test,
        paths,
    )

    model_interpretability(
        best_model_name,
        best_model,
        X_test,
        y_test,
        paths,
    )

    save_model_artifact(
        best_model_name,
        best_model,
        paths,
    )

    create_final_report(
        df,
        results,
        test_results,
        outliers,
        best_model_name,
        paths,
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(
        f"\nBest model: {best_model_name}"
    )
    print(
        f"Results saved to: {output_dir.resolve()}"
    )
    print(
        "\nThe fitted model is available at:"
    )
    print(
        (paths["model"] / "best_churn_model.joblib").resolve()
    )


if __name__ == "__main__":
    main()
