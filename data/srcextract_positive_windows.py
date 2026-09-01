import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
import requests


@dataclass
class FastaRecord:
    """
    Stores one FASTA entry from UniProt.

    accession:
        UniProt accession, e.g. P04637 or P04637-2

    header:
        Full FASTA header line, excluding ">"

    sequence:
        Full amino acid sequence

    is_isoform:
        True if accession looks like an isoform, e.g. P04637-2
    """
    accession: str
    header: str
    sequence: str
    is_isoform: bool


def make_fasta_record(header: str, seq_lines: List[str]) -> FastaRecord:
    """
    Convert a FASTA header and sequence lines into a FastaRecord object.
    """
    parts = header.split("|")

    if len(parts) >= 2:
        accession = parts[1]
    else:
        accession = header.split()[0]

    sequence = "".join(seq_lines)
    is_isoform = "-" in accession

    return FastaRecord(
        accession=accession,
        header=header,
        sequence=sequence,
        is_isoform=is_isoform,
    )


def parse_fasta(fasta_text: str) -> List[FastaRecord]:
    """
    Parse UniProt FASTA text into a list of FastaRecord objects.

    A single UniProt query can return multiple FASTA entries if isoforms
    are included.
    """
    records = []
    header = None
    seq_lines = []

    for line in fasta_text.strip().splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith(">"):
            if header is not None:
                records.append(make_fasta_record(header, seq_lines))

            header = line[1:]
            seq_lines = []
        else:
            seq_lines.append(line)

    if header is not None:
        records.append(make_fasta_record(header, seq_lines))

    return records


def fetch_uniprot_records(
    uniprot_id: str,
    include_isoforms: bool = True,
    timeout: int = 20,
) -> List[FastaRecord]:
    """
    Fetch canonical and optionally isoform sequences from UniProt.

    Returns:
        A list of FastaRecord objects.
    """
    url = "https://rest.uniprot.org/uniprotkb/stream"

    params = {
        "query": f"accession:{uniprot_id}",
        "format": "fasta",
        "includeIsoform": str(include_isoforms).lower(),
    }

    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()

    records = parse_fasta(response.text)

    if not records:
        raise ValueError(f"No FASTA records returned for UniProt ID: {uniprot_id}")

    return records


def extract_linear_region(
    sequence: str,
    position: int,
    window: int = 10,
) -> Dict[str, Any]:
    """
    Extract amino acid sequence around a 1-indexed biological position.

    Example:
        sequence = ABCDEFGHIJK
        position = 6
        window = 2

        Output region = DEFGH

    Returns:
        Dictionary containing:
        - linear_region
        - window_start
        - window_end
        - site_residue
    """
    if not sequence:
        raise ValueError("Empty sequence")

    if position < 1:
        raise ValueError(f"Position must be >= 1, got {position}")

    if position > len(sequence):
        raise ValueError(
            f"Position {position} is outside sequence length {len(sequence)}"
        )

    # Biological positions are 1-indexed, Python strings are 0-indexed.
    idx = position - 1

    start_idx = max(0, idx - window)
    end_idx = min(len(sequence), idx + window + 1)

    return {
        "linear_region": sequence[start_idx:end_idx],
        "window_start": start_idx + 1,
        "window_end": end_idx,
        "site_residue": sequence[idx],
    }


def process_dataframe(
    df: pd.DataFrame,
    uniprot_col: str = "uniprot",
    position_col: str = "position",
    window: int = 10,
    include_isoforms: bool = True,
) -> pd.DataFrame:
    """
    Process an input dataframe containing UniProt IDs and phosphosite positions.

    Important:
        If a UniProt ID has multiple isoforms, one input row may produce
        multiple output rows.

    Output columns include:
        - input_row
        - input_uniprot
        - input_position
        - isoform_accession
        - is_isoform
        - fasta_header
        - full_sequence
        - sequence_length
        - linear_region
        - window_start
        - window_end
        - site_residue
        - error
    """
    required_cols = {uniprot_col, position_col}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    sequence_cache: Dict[str, List[FastaRecord]] = {}
    output_rows = []

    for row_index, row in df.iterrows():
        if row_index % 50 == 0:
            print(f"Processed {row_index} input rows...")
        raw_uniprot_id = row[uniprot_col]

        try:
            if pd.isna(raw_uniprot_id):
                raise ValueError("Missing UniProt ID")

            uniprot_id = str(raw_uniprot_id).strip()

            if not uniprot_id:
                raise ValueError("Empty UniProt ID")

            position = int(row[position_col])

            if uniprot_id not in sequence_cache:
                sequence_cache[uniprot_id] = fetch_uniprot_records(
                    uniprot_id,
                    include_isoforms=include_isoforms,
                )

            records = sequence_cache[uniprot_id]

            for record in records:
                try:
                    region_info = extract_linear_region(
                        record.sequence,
                        position,
                        window=window,
                    )

                    output_rows.append({
                        "input_row": row_index,
                        "input_uniprot": uniprot_id,
                        "input_position": position,
                        "isoform_accession": record.accession,
                        "is_isoform": record.is_isoform,
                        "fasta_header": record.header,
                        "full_sequence": record.sequence,
                        "sequence_length": len(record.sequence),
                        "linear_region": region_info["linear_region"],
                        "window_start": region_info["window_start"],
                        "window_end": region_info["window_end"],
                        "site_residue": region_info["site_residue"],
                        "error": None,
                    })

                except Exception as isoform_error:
                    output_rows.append({
                        "input_row": row_index,
                        "input_uniprot": uniprot_id,
                        "input_position": position,
                        "isoform_accession": record.accession,
                        "is_isoform": record.is_isoform,
                        "fasta_header": record.header,
                        "full_sequence": record.sequence,
                        "sequence_length": len(record.sequence),
                        "linear_region": None,
                        "window_start": None,
                        "window_end": None,
                        "site_residue": None,
                        "error": f"Isoform extraction error: {isoform_error}",
                    })

        except Exception as row_error:
            output_rows.append({
                "input_row": row_index,
                "input_uniprot": raw_uniprot_id,
                "input_position": row.get(position_col, None),
                "isoform_accession": None,
                "is_isoform": None,
                "fasta_header": None,
                "full_sequence": None,
                "sequence_length": None,
                "linear_region": None,
                "window_start": None,
                "window_end": None,
                "site_residue": None,
                "error": f"Row processing error: {row_error}",
            })

    return pd.DataFrame(output_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Extract amino acid sequence windows around phosphosite positions."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input Excel file containing UniProt ID and position columns.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file.",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Number of amino acids to extract on each side of the position.",
    )

    parser.add_argument(
        "--uniprot-col",
        default="uniprot",
        help="Column name containing UniProt IDs.",
    )

    parser.add_argument(
        "--position-col",
        default="position",
        help="Column name containing phosphosite positions.",
    )

    parser.add_argument(
        "--no-isoforms",
        action="store_true",
        help="Only fetch canonical sequence, not isoforms.",
    )

    args = parser.parse_args()

    df = pd.read_excel(args.input)

    output_df = process_dataframe(
        df,
        uniprot_col=args.uniprot_col,
        position_col=args.position_col,
        window=args.window,
        include_isoforms=not args.no_isoforms,
    )

    output_df.to_csv(args.output, index=False)

    n_errors = output_df["error"].notna().sum()

    print(f"Done. Saved to {args.output}")
    print(f"Total output rows: {len(output_df)}")
    print(f"Rows with errors: {n_errors}")


if __name__ == "__main__":
    main()
