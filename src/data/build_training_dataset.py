from pathlib import Path

import pandas as pd


POSITIVE_FILE = Path("functional_score_with_regions_valid.csv")
NEGATIVE_FILE = Path("negative_phosphosites_with_regions.csv")

OUTPUT_FILE = Path("phosphosite_training_dataset.csv")
EXCLUDED_FILE = Path("phosphosite_excluded_rows.csv")


def clean_sequence(series: pd.Series) -> pd.Series:
    """Clean amino-acid sequence strings."""
    return (
        series.astype(str)
        .str.upper()
        .str.strip()
        .str.replace(" ", "", regex=False)
    )


def prepare_positive_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the positive dataset into the common format."""

    required_columns = {
        "input_uniprot",
        "input_position",
        "linear_region",
        "window_start",
        "window_end",
        "site_residue",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Positive file is missing columns: {sorted(missing)}"
        )

    positive = pd.DataFrame(
        {
            "protein_id": df["input_uniprot"],
            "position": df["input_position"],
            "site_residue": df["site_residue"],
            "sequence_window": df["linear_region"],
            "window_start": df["window_start"],
            "window_end": df["window_end"],
            "label": 1,
            "source": "positive",
        }
    )

    return positive


def prepare_negative_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the negative dataset into the common format."""

    required_columns = {
        "input_uniprot",
        "input_position",
        "residue",
        "linear_region",
        "region_start",
        "region_end",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Negative file is missing columns: {sorted(missing)}"
        )

    negative = pd.DataFrame(
        {
            "protein_id": df["input_uniprot"],
            "position": df["input_position"],
            "site_residue": df["residue"],
            "sequence_window": df["linear_region"],
            "window_start": df["region_start"],
            "window_end": df["region_end"],
            "label": 0,
            "source": "negative",
        }
    )

    return negative


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate coordinates, window length, and central residue.

    Valid 21-aa windows should have:
        window_length = 21
        site_index = 10
    """

    df = df.copy()

    df["protein_id"] = (
        df["protein_id"]
        .astype(str)
        .str.strip()
    )

    df["position"] = pd.to_numeric(
        df["position"],
        errors="coerce",
    )

    df["window_start"] = pd.to_numeric(
        df["window_start"],
        errors="coerce",
    )

    df["window_end"] = pd.to_numeric(
        df["window_end"],
        errors="coerce",
    )

    df["site_residue"] = (
        df["site_residue"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["sequence_window"] = clean_sequence(
        df["sequence_window"]
    )

    df["window_length"] = (
        df["sequence_window"].str.len()
    )

    # Coordinates are one-based.
    # Python index inside the window is therefore:
    # protein position - window start
    df["site_index"] = (
        df["position"] - df["window_start"]
    )

    df["exclusion_reason"] = ""

    missing_coordinates = (
        df["position"].isna()
        | df["window_start"].isna()
        | df["window_end"].isna()
    )

    df.loc[
        missing_coordinates,
        "exclusion_reason",
    ] = "missing_or_invalid_coordinates"

    invalid_residue = ~df["site_residue"].isin(
        ["S", "T", "Y"]
    )

    df.loc[
        (df["exclusion_reason"] == "")
        & invalid_residue,
        "exclusion_reason",
    ] = "site_residue_not_STY"

    wrong_length = df["window_length"] != 21

    df.loc[
        (df["exclusion_reason"] == "")
        & wrong_length,
        "exclusion_reason",
    ] = "window_length_not_21"

    wrong_site_index = df["site_index"] != 10

    df.loc[
        (df["exclusion_reason"] == "")
        & wrong_site_index,
        "exclusion_reason",
    ] = "site_not_central"

    # Check that the central residue in the sequence matches
    # the recorded S, T, or Y.
    for row_index, row in df.iterrows():
        if row["exclusion_reason"] != "":
            continue

        actual_central_residue = row["sequence_window"][10]
        expected_residue = row["site_residue"]

        if actual_central_residue != expected_residue:
            df.at[
                row_index,
                "exclusion_reason",
            ] = (
                f"central_residue_mismatch_"
                f"expected_{expected_residue}_"
                f"found_{actual_central_residue}"
            )

    return df


def remove_duplicates(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove duplicate protein-position-label rows."""

    duplicate_mask = df.duplicated(
        subset=[
            "protein_id",
            "position",
            "label",
        ],
        keep="first",
    )

    duplicates = df[duplicate_mask].copy()

    if not duplicates.empty:
        duplicates["exclusion_reason"] = "duplicate_row"

    cleaned = df[~duplicate_mask].copy()

    return cleaned, duplicates


def remove_negative_positive_overlap(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove negative rows when the same protein and position
    are present in the positive dataset.
    """

    positive_sites = set(
        zip(
            df.loc[df["label"] == 1, "protein_id"],
            df.loc[df["label"] == 1, "position"],
        )
    )

    overlap_mask = (
        (df["label"] == 0)
        & df.apply(
            lambda row: (
                row["protein_id"],
                row["position"],
            ) in positive_sites,
            axis=1,
        )
    )

    overlapping_negatives = df[overlap_mask].copy()

    if not overlapping_negatives.empty:
        overlapping_negatives["exclusion_reason"] = (
            "negative_site_also_present_as_positive"
        )

    cleaned = df[~overlap_mask].copy()

    return cleaned, overlapping_negatives


def main() -> None:
    print("Reading datasets...")

    positive_raw = pd.read_csv(POSITIVE_FILE)
    negative_raw = pd.read_csv(NEGATIVE_FILE)

    print(f"Positive rows loaded: {len(positive_raw):,}")
    print(f"Negative rows loaded: {len(negative_raw):,}")

    positive = prepare_positive_data(positive_raw)
    negative = prepare_negative_data(negative_raw)

    combined = pd.concat(
        [positive, negative],
        ignore_index=True,
    )

    print(f"Combined rows: {len(combined):,}")

    combined = validate_dataset(combined)

    invalid_rows = combined[
        combined["exclusion_reason"] != ""
    ].copy()

    valid_rows = combined[
        combined["exclusion_reason"] == ""
    ].copy()

    valid_rows, duplicate_rows = remove_duplicates(
        valid_rows
    )

    valid_rows, overlap_rows = (
        remove_negative_positive_overlap(valid_rows)
    )

    excluded_rows = pd.concat(
        [
            invalid_rows,
            duplicate_rows,
            overlap_rows,
        ],
        ignore_index=True,
    )

    valid_rows = valid_rows.sort_values(
        by=[
            "protein_id",
            "position",
            "label",
        ]
    ).reset_index(drop=True)

    final_columns = [
        "protein_id",
        "position",
        "site_residue",
        "sequence_window",
        "window_start",
        "window_end",
        "window_length",
        "site_index",
        "label",
        "source",
    ]

    valid_rows[final_columns].to_csv(
        OUTPUT_FILE,
        index=False,
    )

    excluded_rows.to_csv(
        EXCLUDED_FILE,
        index=False,
    )

    positive_count = int(
        (valid_rows["label"] == 1).sum()
    )

    negative_count = int(
        (valid_rows["label"] == 0).sum()
    )

    print("\nFinished")
    print(f"Final rows: {len(valid_rows):,}")
    print(f"Positive rows: {positive_count:,}")
    print(f"Negative rows: {negative_count:,}")

    if positive_count > 0:
        ratio = negative_count / positive_count
        print(
            f"Negative-to-positive ratio: {ratio:.2f}:1"
        )

    print(
        f"\nTraining dataset saved as:\n{OUTPUT_FILE}"
    )

    print(
        f"\nExcluded rows saved as:\n{EXCLUDED_FILE}"
    )

    print("\nCounts by residue and label:")
    print(
        valid_rows.groupby(
            ["site_residue", "label"]
        ).size()
    )

    print("\nSite index values:")
    print(
        valid_rows["site_index"]
        .value_counts()
        .sort_index()
    )

    if not excluded_rows.empty:
        print("\nExclusion reasons:")
        print(
            excluded_rows["exclusion_reason"]
            .value_counts()
        )


if __name__ == "__main__":
    main()
