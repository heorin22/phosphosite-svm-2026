from __future__ import annotations

from pathlib import Path
import os
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = Path("phosphosite_training_dataset.csv")
OUTPUT_DIR = Path("svm_outputs")

EXPERIMENT_NAME = "phosphosite-onehot-linear-svm"

N_SPLITS = 5
RANDOM_STATE = 42

# Try these C values.
C_VALUES = [0.01, 0.1, 1.0, 10.0]

# Train these two biological models separately.
MODEL_GROUPS = {
    "ST": ["S", "T"],
    "Y": ["Y"],
}

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYU"

AA_TO_INDEX = {
    amino_acid: index
    for index, amino_acid in enumerate(AMINO_ACIDS)
}


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(filepath: Path) -> pd.DataFrame:
    required_columns = {
        "protein_id",
        "sequence_window",
        "site_residue",
        "label",
    }

    if not filepath.exists():
        raise FileNotFoundError(
            f"Could not find input file:\n{filepath.resolve()}"
        )

    df = pd.read_csv(filepath)

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Input file is missing columns: {sorted(missing_columns)}"
        )

    df = df.copy()

    df["protein_id"] = (
        df["protein_id"]
        .astype(str)
        .str.strip()
    )

    df["sequence_window"] = (
        df["sequence_window"]
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(" ", "", regex=False)
    )

    df["site_residue"] = (
        df["site_residue"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["label"] = pd.to_numeric(
        df["label"],
        errors="raise",
    ).astype(np.int8)

    if not df["label"].isin([0, 1]).all():
        raise ValueError("The label column must contain only 0 and 1.")

    if not df["site_residue"].isin(["S", "T", "Y"]).all():
        raise ValueError(
            "The site_residue column contains residues other than S, T, Y."
        )

    window_lengths = df["sequence_window"].str.len()

    if not (window_lengths == 21).all():
        raise ValueError(
            "Not all sequence windows are 21 amino acids long."
        )

    central_residue = df["sequence_window"].str[10]

    mismatch_mask = central_residue != df["site_residue"]

    if mismatch_mask.any():
        raise ValueError(
            f"Found {int(mismatch_mask.sum())} rows where the central "
            "sequence residue does not match site_residue."
        )

    valid_amino_acids = set(AMINO_ACIDS)

    invalid_sequence_mask = df["sequence_window"].apply(
        lambda sequence: not set(sequence).issubset(valid_amino_acids)
    )

    if invalid_sequence_mask.any():
        invalid_rows = df.loc[
            invalid_sequence_mask,
            ["protein_id", "sequence_window"],
        ].head(10)

        raise ValueError(
            "Some windows contain non-standard amino acids.\n"
            f"First examples:\n{invalid_rows}"
        )

    return df


# ============================================================
# SPARSE ONE-HOT ENCODING
# ============================================================

def one_hot_encode_sparse(
    sequences: pd.Series,
) -> sparse.csr_matrix:
    """
    Encode 21-aa sequence windows into a sparse binary matrix.

    Each sequence has:
        21 positions × 21 residue symbols = 441 features

    Sparse storage is important because each row has only 21 ones
    out of 420 possible features.
    """

    sequence_list = sequences.tolist()

    number_of_sequences = len(sequence_list)
    sequence_length = 21
    number_of_amino_acids = len(AMINO_ACIDS)
    number_of_features = sequence_length * number_of_amino_acids

    # Each row has exactly one non-zero value per sequence position.
    nonzero_values = number_of_sequences * sequence_length

    row_indices = np.empty(
        nonzero_values,
        dtype=np.int32,
    )

    column_indices = np.empty(
        nonzero_values,
        dtype=np.int32,
    )

    data = np.ones(
        nonzero_values,
        dtype=np.float32,
    )

    output_index = 0

    for row_number, sequence in enumerate(sequence_list):
        for position, amino_acid in enumerate(sequence):
            feature_index = (
                position * number_of_amino_acids
                + AA_TO_INDEX[amino_acid]
            )

            row_indices[output_index] = row_number
            column_indices[output_index] = feature_index
            output_index += 1

    encoded_matrix = sparse.csr_matrix(
        (
            data,
            (row_indices, column_indices),
        ),
        shape=(
            number_of_sequences,
            number_of_features,
        ),
        dtype=np.float32,
    )

    return encoded_matrix


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    decision_scores: np.ndarray,
) -> dict[str, float]:
    return {
        "pr_auc": float(
            average_precision_score(
                y_true,
                decision_scores,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                decision_scores,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
    }


def calculate_mean_and_std(
    fold_results: pd.DataFrame,
) -> dict[str, float]:
    metric_columns = [
        "pr_auc",
        "roc_auc",
        "mcc",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
    ]

    summary = {}

    for metric in metric_columns:
        summary[f"mean_{metric}"] = float(
            fold_results[metric].mean()
        )

        summary[f"std_{metric}"] = float(
            fold_results[metric].std(ddof=1)
        )

    return summary


# ============================================================
# CROSS-VALIDATION
# ============================================================

def run_cross_validation(
    X: sparse.csr_matrix,
    y: np.ndarray,
    groups: np.ndarray,
    model_group_name: str,
    c_value: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_records = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(
            X,
            y,
            groups=groups,
        ),
        start=1,
    ):
        fold_start_time = time.time()

        X_train = X[train_indices]
        X_validation = X[validation_indices]

        y_train = y[train_indices]
        y_validation = y[validation_indices]

        training_groups = set(groups[train_indices])
        validation_groups = set(groups[validation_indices])

        overlapping_groups = (
            training_groups.intersection(validation_groups)
        )

        if overlapping_groups:
            raise RuntimeError(
                f"Protein leakage found in fold {fold_number}."
            )

        model = LinearSVC(
            C=c_value,
            class_weight="balanced",
            dual="auto",
            max_iter=10_000,
            random_state=RANDOM_STATE,
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(X_validation)

        decision_scores = model.decision_function(
            X_validation
        )

        metrics = calculate_metrics(
            y_true=y_validation,
            predictions=predictions,
            decision_scores=decision_scores,
        )

        matrix = confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1],
        )

        true_negative, false_positive, false_negative, true_positive = (
            matrix.ravel()
        )

        fold_duration_seconds = time.time() - fold_start_time

        fold_record = {
            "model_group": model_group_name,
            "C": c_value,
            "fold": fold_number,
            "train_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "train_proteins": len(training_groups),
            "validation_proteins": len(validation_groups),
            "train_positives": int(np.sum(y_train == 1)),
            "train_negatives": int(np.sum(y_train == 0)),
            "validation_positives": int(
                np.sum(y_validation == 1)
            ),
            "validation_negatives": int(
                np.sum(y_validation == 0)
            ),
            "true_positive": int(true_positive),
            "false_positive": int(false_positive),
            "true_negative": int(true_negative),
            "false_negative": int(false_negative),
            "duration_seconds": float(fold_duration_seconds),
            **metrics,
        }

        fold_records.append(fold_record)

        print(
            f"{model_group_name} | C={c_value} | "
            f"Fold {fold_number}/{N_SPLITS} | "
            f"PR-AUC={metrics['pr_auc']:.4f} | "
            f"MCC={metrics['mcc']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"Time={fold_duration_seconds:.1f}s"
        )

    fold_results = pd.DataFrame(fold_records)

    summary_metrics = calculate_mean_and_std(
        fold_results
    )

    return fold_results, summary_metrics


# ============================================================
# MLFLOW LOGGING
# ============================================================

def log_cv_run_to_mlflow(
    model_group_name: str,
    residue_types: list[str],
    c_value: float,
    dataset: pd.DataFrame,
    fold_results: pd.DataFrame,
    summary_metrics: dict[str, float],
    results_file: Path,
) -> None:
    run_name = (
        f"onehot_LinearSVC_{model_group_name}_C_{c_value}"
    )

    with mlflow.start_run(run_name=run_name):
        positive_count = int(
            np.sum(dataset["label"].to_numpy() == 1)
        )
        negative_count = int(
            np.sum(dataset["label"].to_numpy() == 0)
        )

        mlflow.log_params(
            {
                "encoding": "position_specific_one_hot",
                "window_length": 21,
                "number_of_features": len(AMINO_ACIDS) * 21,
                "classifier": "LinearSVC",
                "model_group": model_group_name,
                "residue_types": ",".join(residue_types),
                "C": c_value,
                "class_weight": "balanced",
                "loss": "squared_hinge",
                "penalty": "l2",
                "cv_method": "StratifiedGroupKFold",
                "cv_folds": N_SPLITS,
                "group_column": "protein_id",
                "random_state": RANDOM_STATE,
                "number_of_samples": len(dataset),
                "positive_samples": positive_count,
                "negative_samples": negative_count,
                "negative_positive_ratio": (
                    negative_count / positive_count
                    if positive_count > 0
                    else float("nan")
                ),
                "number_of_proteins": (
                    dataset["protein_id"].nunique()
                ),
            }
        )

        mlflow.log_metrics(summary_metrics)

        # Log each fold as a metric history.
        for _, row in fold_results.iterrows():
            fold_number = int(row["fold"])

            for metric_name in [
                "pr_auc",
                "roc_auc",
                "mcc",
                "precision",
                "recall",
                "f1",
                "balanced_accuracy",
            ]:
                mlflow.log_metric(
                    key=f"fold_{metric_name}",
                    value=float(row[metric_name]),
                    step=fold_number,
                )

        mlflow.log_artifact(str(results_file))


# ============================================================
# FINAL MODEL
# ============================================================

def train_and_save_final_model(
    X: sparse.csr_matrix,
    y: np.ndarray,
    model_group_name: str,
    best_c: float,
) -> Path:
    final_model = LinearSVC(
        C=best_c,
        class_weight="balanced",
        dual="auto",
        max_iter=10_000,
        random_state=RANDOM_STATE,
    )

    print(
        f"\nTraining final {model_group_name} model "
        f"on all data with C={best_c}..."
    )

    final_model.fit(X, y)

    model_file = OUTPUT_DIR / (
        f"final_onehot_linear_svm_{model_group_name}.joblib"
    )

    joblib.dump(
        {
            "model": final_model,
            "amino_acids": AMINO_ACIDS,
            "window_length": 21,
            "encoding": "position_specific_one_hot",
            "model_group": model_group_name,
            "C": best_c,
        },
        model_file,
    )

    with mlflow.start_run(
        run_name=f"FINAL_onehot_LinearSVC_{model_group_name}"
    ):
        mlflow.log_params(
            {
                "encoding": "position_specific_one_hot",
                "window_length": 21,
                "number_of_features": len(AMINO_ACIDS) * 21,
                "classifier": "LinearSVC",
                "model_group": model_group_name,
                "C": best_c,
                "class_weight": "balanced",
                "training_samples": len(y),
                "positive_samples": int(np.sum(y == 1)),
                "negative_samples": int(np.sum(y == 0)),
            }
        )

        mlflow.sklearn.log_model(
            sk_model=final_model,
            name="model",
        )

        mlflow.log_artifact(str(model_file))

    return model_file


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print("Loading modelling dataset...")

    full_dataset = load_dataset(
        INPUT_FILE
    )

    print(f"Total rows loaded: {len(full_dataset):,}")
    print(
        f"Unique proteins: "
        f"{full_dataset['protein_id'].nunique():,}"
    )

    all_summary_records = []

    for model_group_name, residue_types in MODEL_GROUPS.items():
        print("\n" + "=" * 70)
        print(
            f"Preparing {model_group_name} model "
            f"for residues {residue_types}"
        )
        print("=" * 70)

        group_dataset = full_dataset[
            full_dataset["site_residue"].isin(residue_types)
        ].copy()

        group_dataset = group_dataset.reset_index(
            drop=True
        )

        y = group_dataset["label"].to_numpy(
            dtype=np.int8
        )

        groups = group_dataset["protein_id"].to_numpy()

        positive_count = int(np.sum(y == 1))
        negative_count = int(np.sum(y == 0))

        print(f"Samples: {len(group_dataset):,}")
        print(f"Positives: {positive_count:,}")
        print(f"Negatives: {negative_count:,}")
        print(
            "Negative:positive ratio: "
            f"{negative_count / positive_count:.2f}:1"
        )

        print("Creating sparse one-hot encoding...")

        X = one_hot_encode_sparse(
            group_dataset["sequence_window"]
        )

        print(f"Encoded matrix shape: {X.shape}")
        print(
            f"Stored non-zero values: {X.nnz:,}"
        )

        group_summary_records = []

        for c_value in C_VALUES:
            print("\n" + "-" * 70)
            print(
                f"Running {N_SPLITS}-fold CV for "
                f"{model_group_name}, C={c_value}"
            )
            print("-" * 70)

            fold_results, summary_metrics = (
                run_cross_validation(
                    X=X,
                    y=y,
                    groups=groups,
                    model_group_name=model_group_name,
                    c_value=c_value,
                )
            )

            results_file = OUTPUT_DIR / (
                f"cv_results_{model_group_name}_C_{c_value}.csv"
            )

            fold_results.to_csv(
                results_file,
                index=False,
            )

            log_cv_run_to_mlflow(
                model_group_name=model_group_name,
                residue_types=residue_types,
                c_value=c_value,
                dataset=group_dataset,
                fold_results=fold_results,
                summary_metrics=summary_metrics,
                results_file=results_file,
            )

            summary_record = {
                "model_group": model_group_name,
                "residue_types": ",".join(residue_types),
                "C": c_value,
                "samples": len(group_dataset),
                "positives": positive_count,
                "negatives": negative_count,
                **summary_metrics,
            }

            group_summary_records.append(
                summary_record
            )

            all_summary_records.append(
                summary_record
            )

            print(
                f"\nMean CV PR-AUC: "
                f"{summary_metrics['mean_pr_auc']:.4f} "
                f"± {summary_metrics['std_pr_auc']:.4f}"
            )

            print(
                f"Mean CV MCC: "
                f"{summary_metrics['mean_mcc']:.4f} "
                f"± {summary_metrics['std_mcc']:.4f}"
            )

        group_summary_df = pd.DataFrame(
            group_summary_records
        )

        # Select C using mean PR-AUC.
        best_row = group_summary_df.loc[
            group_summary_df["mean_pr_auc"].idxmax()
        ]

        best_c = float(best_row["C"])

        print("\n" + "=" * 70)
        print(
            f"Best C for {model_group_name}: {best_c}"
        )
        print(
            f"Best mean CV PR-AUC: "
            f"{best_row['mean_pr_auc']:.4f}"
        )
        print("=" * 70)

        model_file = train_and_save_final_model(
            X=X,
            y=y,
            model_group_name=model_group_name,
            best_c=best_c,
        )

        print(f"Final model saved to: {model_file}")

        # Release the potentially large matrix before the next group.
        del X

    all_summary_df = pd.DataFrame(
        all_summary_records
    )

    summary_file = OUTPUT_DIR / "all_cv_summary.csv"

    all_summary_df.to_csv(
        summary_file,
        index=False,
    )

    print("\n" + "=" * 70)
    print("ALL TRAINING FINISHED")
    print("=" * 70)
    print(f"Summary saved to: {summary_file}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()