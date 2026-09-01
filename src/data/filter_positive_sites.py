import argparse

import pandas as pd


def parse_arguments() -> argparse.Namespace:
    """
    Read command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Filter a phosphosite dataset to retain valid canonical "
            "S/T/Y-centred phosphosites."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV file produced by generate_regions.py.",
    )

    parser.add_argument(
        "--valid-output",
        required=True,
        help="Output CSV file for valid phosphosite rows.",
    )

    parser.add_argument(
        "--excluded-output",
        required=True,
        help="Output CSV file for excluded rows.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    df = pd.read_csv(args.input)

    # Remove accidental whitespace from column names
    df.columns = df.columns.str.strip()

    required_columns = {
        "is_isoform",
        "site_residue",
        "error",
        "input_uniprot",
        "input_position",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Convert is_isoform values to proper Boolean values
    df["is_isoform"] = (
        df["is_isoform"]
        .astype("string")
        .str.strip()
        .str.upper()
        .map(
            {
                "TRUE": True,
                "FALSE": False,
            }
        )
    )

    # Clean the central residue values
    df["site_residue"] = (
        df["site_residue"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # Treat blank error cells as missing values
    df["error"] = df["error"].replace(
        r"^\s*$",
        pd.NA,
        regex=True,
    )

    # Keep rows that:
    # 1. have no processing error
    # 2. are canonical proteins rather than isoforms
    # 3. have S, T, or Y at the requested position
    valid_mask = (
        df["error"].isna()
        & df["is_isoform"].eq(False)
        & df["site_residue"].isin(["S", "T", "Y"])
    )

    valid_df = df.loc[valid_mask].copy()
    excluded_df = df.loc[~valid_mask].copy()

    # Remove duplicated canonical phosphosites
    valid_df = valid_df.drop_duplicates(
        subset=[
            "input_uniprot",
            "input_position",
        ]
    ).reset_index(drop=True)

    valid_df.to_csv(
        args.valid_output,
        index=False,
    )

    excluded_df.to_csv(
        args.excluded_output,
        index=False,
    )

    print(f"Original rows: {len(df):,}")
    print(f"Valid rows: {len(valid_df):,}")
    print(f"Excluded rows: {len(excluded_df):,}")
    print(f"Saved valid data to: {args.valid_output}")
    print(f"Saved excluded data to: {args.excluded_output}")


if __name__ == "__main__":
    main()
