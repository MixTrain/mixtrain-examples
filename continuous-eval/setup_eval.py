"""Create the result dataset and Eval for the continuous VQA example."""

import pandas as pd
from mixtrain import Dataset, Eval, Image

from continuous_eval import eval_name, image_col, result_columns, result_dataset


def create_result_dataset() -> Dataset:
    empty_dataset = pd.DataFrame(columns=result_columns, dtype="string")
    return Dataset.from_pandas(empty_dataset).save(
        result_dataset,
        description="Continuous VQA eval",
        column_types={image_col: Image},
    )


def main() -> None:
    if not Dataset.exists(result_dataset):
        create_result_dataset()

    results = Dataset(result_dataset)

    if not Eval.exists(eval_name):
        Eval.from_dataset(results, name=eval_name, columns=result_columns)


if __name__ == "__main__":
    main()
