"""Append labeled examples from a CSV file.

Usage:
    python add_examples.py data/initial_examples.csv
"""

import argparse

from mixtrain import Dataset, MixClient

DATASET_NAME = "vlm-post-training-data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", help="CSV file containing labeled examples")
    args = parser.parse_args()

    Dataset(DATASET_NAME).append(
        Dataset.from_file(args.csv_file),
        copy_files=True,
    )
    print(f"Appended examples from '{args.csv_file}'")
    print(MixClient().frontend_url(f"/datasets/{DATASET_NAME}"))


if __name__ == "__main__":
    main()
