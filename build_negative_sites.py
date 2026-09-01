from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ============================================================
# 1. Command-line arguments
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate a putative negative phosphosite dataset from "
            "unannotated S/T/Y residues in proteins represented in "
            "a positive phosphosite dataset."
        )
    )

    parser.add_argument(
        "--positive-csv",
        required=True,
        help="Filtered positive phosphosite CSV file.",
    )

    parser.add_argument(
        "--fasta",
        required=True,
        help="Human proteome FASTA or FASTA.GZ file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file for the negative dataset.",
    )

    parser.add_argument(
        "--missing-output",
        default="missing_accessions.csv",
        help=(
            "Output CSV for accessions not found in the FASTA. "
            "Default: missing_accessions.csv"
        ),
    )

    parser.add_argument(
        "--invalid-output",
        default="invalid_positive_sites.csv",
        help=(
            "Output CSV for positive sites that fail FASTA validation. "
            "Default: invalid_positive_sites.csv"
        ),
    )

    parser.add_argument(
        "--accession-col",
        default="input_uniprot",
        help=(
            "Name of the column containing UniProt accessions. "
            "Default: input_uniprot"
        ),
    )

    parser.add_argument(
        "--position-col",
        default="input_position",
        help=(
            "Name of the column containing one-based residue positions. "
            "Default: input_position"
        ),
    )

    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help=(
            "Number of residues to include on each side of the central site. "
            "Default: 10"
        ),
    )

    parser.add_argument(
        "--pad-terminals",
        action="store_true",
        help=(
            "Pad incomplete N- or C-terminal windows with X. "
            "By default, incomplete windows are excluded."
        ),
    )

    return parser.parse_args()


# ============================================================
# 2. Open plain-text or gzip-compressed FASTA
# ============================================================

def open_fasta_file(
    fasta_path: str | Path,
):
    """
    Open either a plain-text FASTA file or a gzip-compressed FASTA file.

    Compression is detected from the file contents, so the function works
    even if a gzip-compressed file does not end with '.gz'.
    """

    fasta_path = Path(fasta_path)

    if not fasta_path.exists():
        raise FileNotFoundError(
            f"FASTA file not found: {fasta_path}"
        )

    with fasta_path.open("rb") as test_file:
        magic_bytes = test_file.read(2)

    is_gzip = magic_bytes == b"\x1f\x8b"

    if is_gzip:
        print("Detected gzip-compressed FASTA file.")

        return gzip.open(
            fasta_path,
            mode="rt",
            encoding="utf-8",
        )

    print("Detected plain-text FASTA file.")

    return fasta_path.open(
        mode="r",
        encoding="utf-8",
    )


# ============================================================
# 3. Extract accession from a UniProt FASTA header
# ============================================================

def extract_accession_from_header(
    header: str,
) -> str:
    """
    Extract a UniProt accession from a FASTA header.

    Examples
    --------
    sp|P04637|P53_HUMAN
        -> P04637

    tr|A0A123|A0A123_HUMAN
        -> A0A123

    P04637
        -> P04637
    """

    first_token = header.split()[0]

    if "|" in first_token:
        parts = first_token.split("|")

        if len(parts) >= 2:
            return parts[1]

    return first_token


# ============================================================
# 4. Read FASTA
# ============================================================

def read_fasta(
    fasta_path: str | Path,
) -> dict[str, str]:
    """
    Read a FASTA file and return:

        accession -> amino-acid sequence
    """

    fasta_path = Path(fasta_path)

    sequences: dict[str, str] = {}

    current_accession: str | None = None
    sequence_parts: list[str] = []

    with open_fasta_file(fasta_path) as handle:

        for raw_line in tqdm(
            handle,
            desc="Reading FASTA",
            unit="line",
            dynamic_ncols=True,
        ):
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):

                if current_accession is not None:
                    sequences[current_accession] = "".join(
                        sequence_parts
                    )

                current_accession = extract_accession_from_header(
                    line[1:]
                )

                sequence_parts = []

            else:
                sequence_parts.append(
                    line.upper()
                )

    if current_accession is not None:
        sequences[current_accession] = "".join(
            sequence_parts
        )

    if not sequences:
        raise ValueError(
            "No FASTA records were found in the supplied file."
        )

    return sequences


# ============================================================
# 5. Extract a local sequence window
# ============================================================

def extract_window(
    sequence: str,
    position: int,
    flank_size: int,
    require_full_window: bool,
) -> str | None:
    """
    Extract a sequence window centred on a one-based residue position.

    For flank_size = 10:

        10 residues before
        + central residue
        + 10 residues after

    Total length = 21.
    """

    if not sequence:
        raise ValueError("Cannot extract a window from an empty sequence.")

    if position < 1:
        raise ValueError(
            f"Position must be at least 1, received {position}."
        )

    if position > len(sequence):
        raise ValueError(
            f"Position {position} is outside sequence length "
            f"{len(sequence)}."
        )

    centre_index = position - 1

    start = centre_index - flank_size
    end = centre_index + flank_size + 1

    expected_length = 2 * flank_size + 1

    if require_full_window:

        if start < 0 or end > len(sequence):
            return None

        window = sequence[start:end]

    else:

        left_padding = max(
            0,
            -start,
        )

        right_padding = max(
            0,
            end - len(sequence),
        )

        valid_start = max(
            0,
            start,
        )

        valid_end = min(
            len(sequence),
            end,
        )

        window = (
            "X" * left_padding
            + sequence[valid_start:valid_end]
            + "X" * right_padding
        )

    if len(window) != expected_length:
        raise ValueError(
            f"Window length mismatch. "
            f"Expected {expected_length}, obtained {len(window)}."
        )

    return window


# ============================================================
# 6. Load and clean the positive dataset
# ============================================================

def prepare_positive_dataframe(
    positive_csv: str | Path,
    accession_column: str,
    position_column: str,
) -> pd.DataFrame:
    """
    Load the positive phosphosite dataset and clean accession
    and position values.
    """

    positive_csv = Path(positive_csv)

    if not positive_csv.exists():
        raise FileNotFoundError(
            f"Positive CSV not found: {positive_csv}"
        )

    positive_df = pd.read_csv(
        positive_csv
    )

    positive_df.columns = (
        positive_df.columns
        .astype(str)
        .str.strip()
    )

    required_columns = {
        accession_column,
        position_column,
    }

    missing_columns = (
        required_columns
        - set(positive_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}\n\n"
            f"Available columns:\n"
            f"{positive_df.columns.tolist()}"
        )

    positive_df = positive_df.copy()

    positive_df[accession_column] = (
        positive_df[accession_column]
        .astype("string")
        .str.strip()
    )

    positive_df[position_column] = pd.to_numeric(
        positive_df[position_column],
        errors="coerce",
    )

    invalid_accessions = (
        positive_df[accession_column].isna()
        | positive_df[accession_column].eq("")
    )

    invalid_positions = (
        positive_df[position_column].isna()
    )

    invalid_input_mask = (
        invalid_accessions
        | invalid_positions
    )

    if invalid_input_mask.any():
        print(
            f"Warning: {invalid_input_mask.sum():,} rows have "
            "missing accessions or invalid positions and will be removed."
        )

    positive_df = positive_df.loc[
        ~invalid_input_mask
    ].copy()

    positive_df[position_column] = (
        positive_df[position_column]
        .astype(int)
    )

    positive_df = positive_df[
        positive_df[position_column] >= 1
    ].copy()

    # Remove duplicate accession-position pairs
    positive_df = (
        positive_df
        .drop_duplicates(
            subset=[
                accession_column,
                position_column,
            ]
        )
        .reset_index(drop=True)
    )

    return positive_df


# ============================================================
# 7. Validate positive sites against the FASTA
# ============================================================

def validate_positive_sites(
    positive_df: pd.DataFrame,
    sequences: dict[str, str],
    accession_column: str,
    position_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Confirm that each positive site:

    - has an accession present in the FASTA
    - has a valid position
    - is centred on S, T, or Y

    Returns
    -------
    valid_positive_df
    invalid_positive_df
    """

    valid_rows: list[dict] = []
    invalid_rows: list[dict] = []

    records = positive_df.to_dict(
        orient="records"
    )

    for row in tqdm(
        records,
        desc="Validating positive sites",
        unit="site",
        dynamic_ncols=True,
    ):
        accession = row[accession_column]
        position = row[position_column]

        sequence = sequences.get(
            accession
        )

        if sequence is None:
            invalid_row = row.copy()
            invalid_row["reason"] = (
                "accession_not_found_in_fasta"
            )

            invalid_rows.append(
                invalid_row
            )

            continue

        if position < 1 or position > len(sequence):
            invalid_row = row.copy()
            invalid_row["reason"] = (
                "position_out_of_range"
            )

            invalid_rows.append(
                invalid_row
            )

            continue

        residue = sequence[
            position - 1
        ]

        if residue not in {"S", "T", "Y"}:
            invalid_row = row.copy()
            invalid_row["reason"] = (
                f"centre_is_{residue}_not_STY"
            )

            invalid_rows.append(
                invalid_row
            )

            continue

        valid_row = row.copy()
        valid_row["residue"] = residue

        valid_rows.append(
            valid_row
        )

    valid_positive_df = pd.DataFrame(
        valid_rows
    )

    invalid_positive_df = pd.DataFrame(
        invalid_rows
    )

    return (
        valid_positive_df,
        invalid_positive_df,
    )


# ============================================================
# 8. Generate the negative dataset
# ============================================================

def build_negative_dataset(
    positive_df: pd.DataFrame,
    sequences: dict[str, str],
    accession_column: str,
    position_column: str,
    flank_size: int,
    require_full_window: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each unique protein in the positive dataset:

    1. Find all S, T, and Y residues.
    2. Exclude positions present in the positive dataset.
    3. Extract the local sequence window.
    4. Save the remaining sites as putative negatives.
    """

    positive_sites = set(
        zip(
            positive_df[accession_column],
            positive_df[position_column],
        )
    )

    unique_accessions = sorted(
        positive_df[accession_column]
        .dropna()
        .unique()
    )

    negative_rows: list[dict] = []
    missing_accession_rows: list[dict] = []

    print()
    print("=" * 70)
    print("Generating negative phosphosite candidates")
    print("=" * 70)

    print(
        f"Unique proteins to process: "
        f"{len(unique_accessions):,}"
    )

    progress_bar = tqdm(
        unique_accessions,
        total=len(unique_accessions),
        desc="Generating negatives",
        unit="protein",
        dynamic_ncols=True,
    )

    for accession in progress_bar:

        sequence = sequences.get(
            accession
        )

        if sequence is None:

            missing_accession_rows.append(
                {
                    accession_column: accession,
                    "reason": "accession_not_found_in_fasta",
                }
            )

            progress_bar.set_postfix(
                negatives=f"{len(negative_rows):,}",
                missing=len(missing_accession_rows),
            )

            continue

        sequence_length = len(
            sequence
        )

        for zero_based_index, residue in enumerate(
            sequence
        ):
            if residue not in {"S", "T", "Y"}:
                continue

            position = zero_based_index + 1

            # Exclude known positive phosphosite positions
            if (
                accession,
                position,
            ) in positive_sites:
                continue

            window = extract_window(
                sequence=sequence,
                position=position,
                flank_size=flank_size,
                require_full_window=require_full_window,
            )

            # Skip terminal sites if complete windows are required
            if window is None:
                continue

            centre_residue = window[
                flank_size
            ]

            if centre_residue != residue:
                raise ValueError(
                    f"Centre mismatch at "
                    f"{accession}:{position}. "
                    f"Expected {residue}, "
                    f"found {centre_residue}."
                )

            negative_rows.append(
                {
                    accession_column: accession,
                    position_column: position,
                    "residue": residue,
                    "sequence_length": sequence_length,
                    "region_start": position - flank_size,
                    "region_end": position + flank_size,
                    "linear_region": window,
                    "label": 0,
                    "annotation_status": "unannotated_STY",
                }
            )

        progress_bar.set_postfix(
            negatives=f"{len(negative_rows):,}",
            missing=len(missing_accession_rows),
        )

    negative_df = pd.DataFrame(
        negative_rows
    )

    missing_df = pd.DataFrame(
        missing_accession_rows
    )

    if not negative_df.empty:
        negative_df = (
            negative_df
            .drop_duplicates(
                subset=[
                    accession_column,
                    position_column,
                ]
            )
            .reset_index(drop=True)
        )

    return (
        negative_df,
        missing_df,
    )


# ============================================================
# 9. Validate the negative dataset
# ============================================================

def validate_negative_dataset(
    positive_df: pd.DataFrame,
    negative_df: pd.DataFrame,
    accession_column: str,
    position_column: str,
    flank_size: int,
) -> None:
    """
    Run sanity checks on the generated negative dataset.
    """

    print()
    print("=" * 70)
    print("Validating negative dataset")
    print("=" * 70)

    if negative_df.empty:
        raise ValueError(
            "The generated negative dataset is empty."
        )

    expected_window_length = (
        2 * flank_size + 1
    )

    # Check that all central residues are S, T, or Y
    valid_residues = (
        negative_df["residue"]
        .isin({"S", "T", "Y"})
    )

    if not valid_residues.all():
        raise ValueError(
            "The negative dataset contains "
            "non-S/T/Y centre residues."
        )

    # Check window lengths
    valid_lengths = (
        negative_df["linear_region"]
        .astype("string")
        .str.len()
        == expected_window_length
    )

    if not valid_lengths.all():
        invalid_count = (
            ~valid_lengths
        ).sum()

        raise ValueError(
            f"{invalid_count:,} sequence windows "
            "have an incorrect length."
        )

    # Check window centre
    valid_centres = (
        negative_df["linear_region"]
        .astype("string")
        .str[flank_size]
        == negative_df["residue"]
    )

    if not valid_centres.all():
        invalid_count = (
            ~valid_centres
        ).sum()

        raise ValueError(
            f"{invalid_count:,} sequence windows "
            "have an incorrect centre residue."
        )

    # Check positive-negative overlap
    positive_pairs = set(
        zip(
            positive_df[accession_column],
            positive_df[position_column],
        )
    )

    negative_pairs = set(
        zip(
            negative_df[accession_column],
            negative_df[position_column],
        )
    )

    overlap = (
        positive_pairs
        & negative_pairs
    )

    if overlap:
        examples = list(
            overlap
        )[:10]

        raise ValueError(
            f"Positive-negative overlap detected.\n"
            f"Examples: {examples}"
        )

    # Check duplicates
    duplicate_count = negative_df.duplicated(
        subset=[
            accession_column,
            position_column,
        ]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"The negative dataset contains "
            f"{duplicate_count:,} duplicated sites."
        )

    print("Validation passed.")
    print("- All centre residues are S/T/Y")
    print(
        f"- All windows have length "
        f"{expected_window_length}"
    )
    print("- All windows have the correct centre residue")
    print("- No positive-negative site overlap")
    print("- No duplicated negative sites")


# ============================================================
# 10. Print summary
# ============================================================

def print_summary(
    positive_df: pd.DataFrame,
    negative_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    invalid_positive_df: pd.DataFrame,
    accession_column: str,
) -> None:
    """
    Print a summary of the generated datasets.
    """

    print()
    print("=" * 70)
    print("Dataset summary")
    print("=" * 70)

    print(
        f"Valid positive sites: "
        f"{len(positive_df):,}"
    )

    print(
        f"Positive proteins: "
        f"{positive_df[accession_column].nunique():,}"
    )

    print(
        f"Generated negative sites: "
        f"{len(negative_df):,}"
    )

    print(
        f"Negative proteins: "
        f"{negative_df[accession_column].nunique():,}"
    )

    print(
        f"Missing accessions: "
        f"{len(missing_df):,}"
    )

    print(
        f"Invalid positive sites: "
        f"{len(invalid_positive_df):,}"
    )

    print()
    print("Negative residue counts:")

    residue_counts = (
        negative_df["residue"]
        .value_counts()
        .reindex(
            ["S", "T", "Y"],
            fill_value=0,
        )
    )

    print(
        residue_counts
    )

    print()
    print("Negative residue proportions:")

    residue_proportions = (
        negative_df["residue"]
        .value_counts(
            normalize=True
        )
        .reindex(
            ["S", "T", "Y"],
            fill_value=0,
        )
        .round(4)
    )

    print(
        residue_proportions
    )


# ============================================================
# 11. Main
# ============================================================

def main() -> None:
    args = parse_arguments()

    if args.window < 0:
        raise ValueError(
            f"--window must be at least 0, received {args.window}."
        )

    require_full_window = not args.pad_terminals

    print("=" * 70)
    print("Run configuration")
    print("=" * 70)

    print(f"Positive CSV: {args.positive_csv}")
    print(f"FASTA file: {args.fasta}")
    print(f"Negative output: {args.output}")
    print(f"Accession column: {args.accession_col}")
    print(f"Position column: {args.position_col}")
    print(f"Window on each side: {args.window}")
    print(f"Require full window: {require_full_window}")

    print()
    print("=" * 70)
    print("Reading positive phosphosite dataset")
    print("=" * 70)

    positive_df = prepare_positive_dataframe(
        positive_csv=args.positive_csv,
        accession_column=args.accession_col,
        position_column=args.position_col,
    )

    print(
        f"Positive rows after deduplication: "
        f"{len(positive_df):,}"
    )

    print(
        f"Unique accessions in positive dataset: "
        f"{positive_df[args.accession_col].nunique():,}"
    )

    print()
    print("=" * 70)
    print("Reading human proteome FASTA")
    print("=" * 70)

    sequences = read_fasta(
        args.fasta
    )

    print(
        f"FASTA proteins loaded: "
        f"{len(sequences):,}"
    )

    print()
    print("=" * 70)
    print("Checking positive sites against FASTA")
    print("=" * 70)

    valid_positive_df, invalid_positive_df = (
        validate_positive_sites(
            positive_df=positive_df,
            sequences=sequences,
            accession_column=args.accession_col,
            position_column=args.position_col,
        )
    )

    print(
        f"Valid positive sites: "
        f"{len(valid_positive_df):,}"
    )

    print(
        f"Invalid positive sites: "
        f"{len(invalid_positive_df):,}"
    )

    if not invalid_positive_df.empty:
        invalid_positive_df.to_csv(
            args.invalid_output,
            index=False,
        )

        print(
            f"Invalid positive sites saved to: "
            f"{args.invalid_output}"
        )

    if valid_positive_df.empty:
        raise ValueError(
            "No valid positive sites remain after FASTA validation."
        )

    negative_df, missing_df = build_negative_dataset(
        positive_df=valid_positive_df,
        sequences=sequences,
        accession_column=args.accession_col,
        position_column=args.position_col,
        flank_size=args.window,
        require_full_window=require_full_window,
    )

    validate_negative_dataset(
        positive_df=valid_positive_df,
        negative_df=negative_df,
        accession_column=args.accession_col,
        position_column=args.position_col,
        flank_size=args.window,
    )

    print()
    print("=" * 70)
    print("Saving output files")
    print("=" * 70)

    negative_df.to_csv(
        args.output,
        index=False,
    )

    print(
        f"Negative dataset saved to: "
        f"{args.output}"
    )

    if not missing_df.empty:
        missing_df.to_csv(
            args.missing_output,
            index=False,
        )

        print(
            f"Missing accessions saved to: "
            f"{args.missing_output}"
        )
    else:
        print(
            "All accessions were found in the FASTA file."
        )

    print_summary(
        positive_df=valid_positive_df,
        negative_df=negative_df,
        missing_df=missing_df,
        invalid_positive_df=invalid_positive_df,
        accession_column=args.accession_col,
    )

    print()
    print("Finished successfully.")


if __name__ == "__main__":
    main()