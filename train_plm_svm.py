from __future__ import annotations

from pathlib import Path
import os
import json
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ============================================================
# SETTINGS
# ============================================================

EMBEDDING_DIR = Path("plm_embeddings")

EMBEDDINGS_FILE = EMBEDDING_DIR / (
    "esm2_embeddings_centre_plus_mean.npy"
)

METADATA_FILE = EMBEDDING_DIR / "plm_metadata.csv"
CONFIG_FILE = EMBEDDING_DIR / "embedding_config.json"

OUTPUT_DIR = Path("plm_svm_outputs")

EXPERIMENT_NAME = "phosphosite-plm-linear-svm"

N_SPLITS = 5
RANDOM_STATE = 42

# Start with two values to check runtime.
# Add 0.01 and 10.0 later if necessary.
C_VALUES = [0.1, 1.0]

MODEL_GROUPS = {
    "ST": ["S", "T"],
    "Y": ["Y"],
}


def load_data():
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Missing embeddings file:\n"
            f"{EMBEDDINGS_FILE.resolve()}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing metadata file:\n"
            f"{METADATA_FILE.resolve()}"
        )

    embeddings = np.load(
        EMBEDDINGS_FILE,
        mmap_mode="r",
    )

    metadata = pd.read_csv(
        METADATA_FILE
    )

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Embedding rows ({len(embeddings):,}) do not match "
            f"metadata rows ({len(metadata):,})."
        )

    metadata["protein_id"] = (
        metadata["protein_id"]
        .astype(str)
        .str.strip()
    )

    metadata["site_residue"] = (
        metadata["site_residue"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    metadata["label"] = pd.to_numeric(
        metadata["label"],
        errors="raise",
    ).astype(np.int8)

    return embeddings, metadata


def calculate_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    return {
        "pr_auc": float(
            average_precision_score(
                y_true,
                scores,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                scores,
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


def summarise_fold_metrics(
    fold_results: pd.DataFrame,
) -> dict[str, float]:
    metrics = [
        "pr_auc",
        "roc_auc",
        "mcc",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
    ]

    summary = {}

    for metric in metrics:
        summary[f"mean_{metric}"] = float(
            fold_results[metric].mean()
        )

        summary[f"std_{metric}"] = float(
            fold_results[metric].std(ddof=1)
        )

    return summary


def run_cv(
    X,
    y: np.ndarray,
    groups: np.ndarray,
    model_group: str,
    c_value: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    records = []

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(
            np.zeros(len(y)),
            y,
            groups=groups,
        ),
        start=1,
    ):
        start_time = time.time()

        train_proteins = set(groups[train_indices])
        validation_proteins = set(
            groups[validation_indices]
        )

        if train_proteins & validation_proteins:
            raise RuntimeError(
                f"Protein leakage detected in fold {fold}."
            )

        # Convert only this fold's data from float16 to float32.
        X_train = np.asarray(
            X[train_indices],
            dtype=np.float32,
        )

        X_validation = np.asarray(
            X[validation_indices],
            dtype=np.float32,
        )

        y_train = y[train_indices]
        y_validation = y[validation_indices]

        pipeline = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "svm",
                    LinearSVC(
                        C=c_value,
                        class_weight="balanced",
                        dual="auto",
                        max_iter=10_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        predictions = pipeline.predict(
            X_validation
        )

        scores = pipeline.decision_function(
            X_validation
        )

        metrics = calculate_metrics(
            y_true=y_validation,
            predictions=predictions,
            scores=scores,
        )

        matrix = confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1],
        )

        tn, fp, fn, tp = matrix.ravel()

        duration = time.time() - start_time

        record = {
            "model_group": model_group,
            "C": c_value,
            "fold": fold,
            "train_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "train_proteins": len(train_proteins),
            "validation_proteins": len(validation_proteins),
            "train_positives": int(
                np.sum(y_train == 1)
            ),
            "train_negatives": int(
                np.sum(y_train == 0)
            ),
            "validation_positives": int(
                np.sum(y_validation == 1)
            ),
            "validation_negatives": int(
                np.sum(y_validation == 0)
            ),
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
            "duration_seconds": float(duration),
            **metrics,
        }

        records.append(record)

        print(
            f"{model_group} | C={c_value} | "
            f"fold {fold}/{N_SPLITS} | "
            f"PR-AUC={metrics['pr_auc']:.4f} | "
            f"MCC={metrics['mcc']:.4f} | "
            f"time={duration:.1f}s"
        )

        del X_train
        del X_validation
        del pipeline

    results = pd.DataFrame(records)
    summary = summarise_fold_metrics(results)

    return results, summary


def log_run(
    model_group: str,
    residues: list[str],
    c_value: float,
    group_metadata: pd.DataFrame,
    fold_results: pd.DataFrame,
    summary: dict[str, float],
    result_file: Path,
    embedding_config: dict,
) -> None:
    run_name = (
        f"ESM2_{embedding_config['embedding_method']}_"
        f"LinearSVC_{model_group}_C_{c_value}"
    )

    with mlflow.start_run(
        run_name=run_name
    ):
        positives = int(
            (group_metadata["label"] == 1).sum()
        )

        negatives = int(
            (group_metadata["label"] == 0).sum()
        )

        mlflow.log_params(
            {
                "feature_type": "protein_language_model",
                "plm_model": embedding_config["model_name"],
                "embedding_method": (
                    embedding_config["embedding_method"]
                ),
                "embedding_dimension": (
                    embedding_config["output_dimension"]
                ),
                "window_length": (
                    embedding_config["window_length"]
                ),
                "classifier": "LinearSVC",
                "model_group": model_group,
                "residues": ",".join(residues),
                "C": c_value,
                "class_weight": "balanced",
                "scaler": "StandardScaler",
                "cv_method": "StratifiedGroupKFold",
                "cv_folds": N_SPLITS,
                "group_column": "protein_id",
                "random_state": RANDOM_STATE,
                "samples": len(group_metadata),
                "positives": positives,
                "negatives": negatives,
                "negative_positive_ratio": (
                    negatives / positives
                ),
                "unique_proteins": (
                    group_metadata["protein_id"].nunique()
                ),
            }
        )

        mlflow.log_metrics(summary)

        for _, row in fold_results.iterrows():
            fold = int(row["fold"])

            for metric in [
                "pr_auc",
                "roc_auc",
                "mcc",
                "precision",
                "recall",
                "f1",
                "balanced_accuracy",
            ]:
                mlflow.log_metric(
                    key=f"fold_{metric}",
                    value=float(row[metric]),
                    step=fold,
                )

        mlflow.log_artifact(
            str(result_file)
        )


def train_final_model(
    X,
    y: np.ndarray,
    model_group: str,
    best_c: float,
    group_indices: np.ndarray,
) -> Path:
    print(
        f"Training final {model_group} PLM model "
        f"with C={best_c}..."
    )

    X_all = np.asarray(
        X[group_indices],
        dtype=np.float32,
    )

    pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "svm",
                LinearSVC(
                    C=best_c,
                    class_weight="balanced",
                    dual="auto",
                    max_iter=10_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    pipeline.fit(
        X_all,
        y,
    )

    output_file = OUTPUT_DIR / (
        f"final_plm_linear_svm_{model_group}.joblib"
    )

    joblib.dump(
        pipeline,
        output_file,
    )

    with mlflow.start_run(
        run_name=f"FINAL_PLM_LinearSVC_{model_group}"
    ):
        mlflow.log_params(
            {
                "classifier": "LinearSVC",
                "model_group": model_group,
                "C": best_c,
                "class_weight": "balanced",
                "scaler": "StandardScaler",
                "training_samples": len(y),
            }
        )

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
        )

        mlflow.log_artifact(
            str(output_file)
        )

    return output_file


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

    embeddings, metadata = load_data()

    with CONFIG_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        embedding_config = json.load(file)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Metadata rows: {len(metadata):,}")

    all_summaries = []

    for model_group, residues in MODEL_GROUPS.items():
        print("\n" + "=" * 70)
        print(
            f"Model group: {model_group}, residues={residues}"
        )
        print("=" * 70)

        group_mask = (
            metadata["site_residue"]
            .isin(residues)
            .to_numpy()
        )

        group_indices = np.flatnonzero(
            group_mask
        )

        group_metadata = (
            metadata.iloc[group_indices]
            .reset_index(drop=True)
        )

        y = group_metadata["label"].to_numpy(
            dtype=np.int8
        )

        groups = (
            group_metadata["protein_id"]
            .to_numpy()
        )

        # This is still disk-backed; it is not loaded fully yet.
        X_group = embeddings[group_indices]

        positives = int(np.sum(y == 1))
        negatives = int(np.sum(y == 0))

        print(f"Samples: {len(y):,}")
        print(f"Positives: {positives:,}")
        print(f"Negatives: {negatives:,}")
        print(
            f"Negative:positive = "
            f"{negatives / positives:.2f}:1"
        )

        group_summaries = []

        for c_value in C_VALUES:
            fold_results, summary = run_cv(
                X=X_group,
                y=y,
                groups=groups,
                model_group=model_group,
                c_value=c_value,
            )

            result_file = OUTPUT_DIR / (
                f"cv_results_{model_group}_C_{c_value}.csv"
            )

            fold_results.to_csv(
                result_file,
                index=False,
            )

            log_run(
                model_group=model_group,
                residues=residues,
                c_value=c_value,
                group_metadata=group_metadata,
                fold_results=fold_results,
                summary=summary,
                result_file=result_file,
                embedding_config=embedding_config,
            )

            summary_record = {
                "model_group": model_group,
                "C": c_value,
                "samples": len(y),
                "positives": positives,
                "negatives": negatives,
                **summary,
            }

            group_summaries.append(
                summary_record
            )

            all_summaries.append(
                summary_record
            )

            print(
                f"Mean PR-AUC: "
                f"{summary['mean_pr_auc']:.4f} "
                f"± {summary['std_pr_auc']:.4f}"
            )

        group_summary_df = pd.DataFrame(
            group_summaries
        )

        best_row = group_summary_df.loc[
            group_summary_df["mean_pr_auc"].idxmax()
        ]

        best_c = float(best_row["C"])

        print(
            f"\nBest C for {model_group}: {best_c}"
        )

        final_model_file = train_final_model(
            X=embeddings,
            y=y,
            model_group=model_group,
            best_c=best_c,
            group_indices=group_indices,
        )

        print(
            f"Saved final model: {final_model_file}"
        )

        del X_group

    summary_df = pd.DataFrame(
        all_summaries
    )

    summary_file = OUTPUT_DIR / "all_plm_cv_summary.csv"

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    print("\nTraining completed.")
    print(f"Summary: {summary_file}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()