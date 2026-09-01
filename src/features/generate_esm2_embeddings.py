from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = Path("phosphosite_training_dataset.csv")
OUTPUT_DIR = Path("plm_embeddings")

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# Reduce this if you receive an out-of-memory error.
GPU_BATCH_SIZE = 512
CPU_BATCH_SIZE = 64

WINDOW_LENGTH = 21
CENTRAL_INDEX = 10

# centre + mean gives 320 + 320 = 640 features.
EMBEDDING_METHOD = "centre_plus_mean"

# Save float16 to reduce disk usage.
OUTPUT_DTYPE = np.float16


def clean_sequences(series: pd.Series) -> pd.Series:
    sequences = (
        series.astype(str)
        .str.upper()
        .str.strip()
        .str.replace(" ", "", regex=False)
    )

    # ESM supports common protein sequence symbols, but rare U residues
    # may be handled inconsistently across tokenizers/model versions.
    # Map selenocysteine U to X as an unknown residue.
    sequences = sequences.str.replace("U", "X", regex=False)

    return sequences


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "protein_id",
        "sequence_window",
        "site_residue",
        "label",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df.copy()

    df["protein_id"] = (
        df["protein_id"]
        .astype(str)
        .str.strip()
    )

    df["sequence_window"] = clean_sequences(
        df["sequence_window"]
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

    lengths = df["sequence_window"].str.len()

    if not (lengths == WINDOW_LENGTH).all():
        raise ValueError(
            "Not every sequence window is 21 residues long."
        )

    # U was converted to X, so compare after the same conversion.
    expected_centre = (
        df["site_residue"]
        .replace({"U": "X"})
    )

    actual_centre = df["sequence_window"].str[CENTRAL_INDEX]

    mismatch = actual_centre != expected_centre

    if mismatch.any():
        examples = df.loc[
            mismatch,
            [
                "protein_id",
                "sequence_window",
                "site_residue",
            ],
        ].head(10)

        raise ValueError(
            f"Found {int(mismatch.sum())} central-residue mismatches.\n"
            f"{examples}"
        )

    return df


def get_batch_size(device: torch.device) -> int:
    if device.type == "cuda":
        return GPU_BATCH_SIZE

    return CPU_BATCH_SIZE


def get_embedding_dimension(model) -> int:
    hidden_size = int(model.config.hidden_size)

    if EMBEDDING_METHOD == "centre_only":
        return hidden_size

    if EMBEDDING_METHOD == "mean_only":
        return hidden_size

    if EMBEDDING_METHOD == "centre_plus_mean":
        return hidden_size * 2

    raise ValueError(
        f"Unknown embedding method: {EMBEDDING_METHOD}"
    )


def pool_batch_embeddings(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    hidden_states shape:
        batch_size × token_count × hidden_size

    ESM adds special tokens around the sequence, so residue tokens occupy
    positions 1 through 21 for a 21-residue input.
    """

    residue_embeddings = hidden_states[
        :,
        1 : WINDOW_LENGTH + 1,
        :,
    ]

    centre_embedding = residue_embeddings[
        :,
        CENTRAL_INDEX,
        :,
    ]

    mean_embedding = residue_embeddings.mean(
        dim=1
    )

    if EMBEDDING_METHOD == "centre_only":
        return centre_embedding

    if EMBEDDING_METHOD == "mean_only":
        return mean_embedding

    if EMBEDDING_METHOD == "centre_plus_mean":
        return torch.cat(
            [
                centre_embedding,
                mean_embedding,
            ],
            dim=1,
        )

    raise ValueError(
        f"Unknown embedding method: {EMBEDDING_METHOD}"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)
    df = validate_dataset(df)

    number_of_samples = len(df)

    print(f"Rows: {number_of_samples:,}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )
    else:
        print(
            "Warning: CPU embedding generation may take a long time."
        )

    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    )

    model = model.to(device)
    model.eval()

    hidden_size = int(model.config.hidden_size)
    output_dimension = get_embedding_dimension(model)

    print(f"PLM hidden size: {hidden_size}")
    print(f"Output embedding dimension: {output_dimension}")
    print(f"Embedding method: {EMBEDDING_METHOD}")

    batch_size = get_batch_size(device)

    print(f"Batch size: {batch_size}")

    embeddings_file = OUTPUT_DIR / (
        f"esm2_embeddings_{EMBEDDING_METHOD}.npy"
    )

    # Open a disk-backed NumPy array so all embeddings do not need to
    # remain in RAM simultaneously.
    embedding_array = np.lib.format.open_memmap(
        embeddings_file,
        mode="w+",
        dtype=OUTPUT_DTYPE,
        shape=(
            number_of_samples,
            output_dimension,
        ),
    )

    sequences = df["sequence_window"].tolist()

    start_time = time.time()

    for batch_start in range(
        0,
        number_of_samples,
        batch_size,
    ):
        batch_end = min(
            batch_start + batch_size,
            number_of_samples,
        )

        batch_sequences = sequences[
            batch_start:batch_end
        ]

        encoded = tokenizer(
            batch_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=WINDOW_LENGTH + 2,
            add_special_tokens=True,
        )

        encoded = {
            name: tensor.to(device)
            for name, tensor in encoded.items()
        }

        with torch.inference_mode():
            # Mixed precision makes GPU inference faster and reduces
            # GPU memory use. CPU uses normal float32 inference.
            if device.type == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    output = model(**encoded)
            else:
                output = model(**encoded)

            pooled = pool_batch_embeddings(
                hidden_states=output.last_hidden_state,
                attention_mask=encoded["attention_mask"],
            )

        embedding_array[
            batch_start:batch_end
        ] = (
            pooled
            .detach()
            .cpu()
            .numpy()
            .astype(OUTPUT_DTYPE)
        )

        embedding_array.flush()

        completed = batch_end
        elapsed = time.time() - start_time
        rate = completed / elapsed

        remaining = number_of_samples - completed
        estimated_remaining_seconds = (
            remaining / rate
            if rate > 0
            else float("nan")
        )

        print(
            f"Embedded {completed:,}/{number_of_samples:,} "
            f"({100 * completed / number_of_samples:.2f}%) | "
            f"{rate:.1f} windows/s | "
            f"ETA {estimated_remaining_seconds / 60:.1f} min"
        )

    metadata_file = OUTPUT_DIR / "plm_metadata.csv"

    metadata_columns = [
        "protein_id",
        "position",
        "site_residue",
        "label",
        "sequence_window",
    ]

    available_metadata_columns = [
        column
        for column in metadata_columns
        if column in df.columns
    ]

    df[available_metadata_columns].to_csv(
        metadata_file,
        index=False,
    )

    configuration = {
        "model_name": MODEL_NAME,
        "embedding_method": EMBEDDING_METHOD,
        "model_hidden_size": hidden_size,
        "output_dimension": output_dimension,
        "window_length": WINDOW_LENGTH,
        "central_index": CENTRAL_INDEX,
        "number_of_samples": number_of_samples,
        "stored_dtype": str(np.dtype(OUTPUT_DTYPE)),
        "selenocysteine_handling": "U mapped to X",
    }

    config_file = OUTPUT_DIR / "embedding_config.json"

    with config_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            indent=2,
        )

    print("\nEmbedding generation finished.")
    print(f"Embeddings: {embeddings_file}")
    print(f"Metadata:   {metadata_file}")
    print(f"Config:     {config_file}")


if __name__ == "__main__":
    main()
