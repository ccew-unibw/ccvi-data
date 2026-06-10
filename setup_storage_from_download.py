#!/usr/bin/env python3
"""Set up the data output folder from a downloaded CCVI zip."""

from argparse import ArgumentParser
from pathlib import Path
import shutil
from zipfile import ZipFile

import pandas as pd


SCORES_FILE = "ccvi_scores.parquet"
ZIP_TO_OUTPUT_FILES = {
    "base_grid.parquet": "base_grid_prio.parquet",
    "exposure_layers.parquet": "exposure.parquet",
}
INDEX_COLUMNS = ["pgid", "year", "quarter"]
EXPOSURE_SUFFIX = "_exposure"
RAW_SUFFIX = "_raw"


def parse_args():
    parser = ArgumentParser(
        description=(
            "Reconstruct the structure of the CCVI pipeline output folder from a downloaded full data zip."
            "Files are written to TARGET_FOLDER/output."
        )
    )
    parser.add_argument("zip_path", type=Path, help="Path to the downloaded CCVI full data zip file.")
    parser.add_argument(
        "target_folder",
        type=Path,
        help="Target storage folder. The script creates and writes into TARGET_FOLDER/output.",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Fully replace TARGET_FOLDER/output if it already exists and contains data.",
    )
    return parser.parse_args()


def get_component_column_groups(scores: pd.DataFrame) -> dict[str, list[str]]:
    component_ids = []
    raw_columns = []

    for column in scores.columns:
        if column in INDEX_COLUMNS or column.endswith(EXPOSURE_SUFFIX):
            continue
        if column.endswith(RAW_SUFFIX):
            raw_columns.append(column)
        else:
            component_ids.append(column)

    groups = {component_id: [component_id] for component_id in component_ids}
    for column in raw_columns:
        groups[column.removesuffix(RAW_SUFFIX)].append(column)

    return groups


def confirm_overwrite(output_folder: Path) -> None:
    answer = input(
        f"Overwrite {output_folder}? This will delete all existing data in that folder. "
        "Type 'yes' to continue: "
    )
    if answer != "yes":
        raise SystemExit("Overwrite cancelled.")


def output_from_ccvi_zip(archive: ZipFile, target_folder: Path, overwrite: bool = False) -> None:
    output_folder = target_folder / "output"
    if output_folder.exists() and any(output_folder.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output folder is not empty: {output_folder}")
        confirm_overwrite(output_folder)
        shutil.rmtree(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    scores = pd.read_parquet(archive.open(SCORES_FILE))
    assert list(scores.index.names) == INDEX_COLUMNS
    for component_id, columns in get_component_column_groups(scores).items():
        scores[columns].to_parquet(output_folder / f"{component_id}.parquet", compression="brotli")

    for zip_filename, output_filename in ZIP_TO_OUTPUT_FILES.items():
        df = pd.read_parquet(archive.open(zip_filename))
        df.to_parquet(output_folder / output_filename, compression="brotli")


def main() -> None:
    args = parse_args()
    if not args.zip_path.is_file():
        raise FileNotFoundError(f"Zip file not found: {args.zip_path}")

    with ZipFile(args.zip_path) as archive:
        for filename in [SCORES_FILE] + list(ZIP_TO_OUTPUT_FILES.keys()):
            assert filename in archive.namelist()
        output_from_ccvi_zip(archive, args.target_folder, args.overwrite)


if __name__ == "__main__":
    main()
