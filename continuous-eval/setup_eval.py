"""Create the result dataset and Eval for the continuous VQA example."""

import pandas as pd
from mixtrain import Dataset, Eval, Image, MixClient

from continuous_eval import (
    eval_name,
    image_col,
    result_columns,
    result_dataset,
    eval_set,
)


def main() -> None:
    if not Dataset.exists(eval_set):
        Dataset.from_file("initial_eval_set.csv").save(
            eval_set,
            description="A growing visual question-answer eval set",
            column_types={"image": Image},
            copy_files=True,
        )

    if not Dataset.exists(result_dataset):
        empty_dataset = pd.DataFrame(columns=result_columns, dtype="string")
        Dataset.from_pandas(empty_dataset).save(
            result_dataset,
            description="Continuous VQA eval",
            column_types={image_col: Image},
        )

    results = Dataset(result_dataset)

    if not Eval.exists(eval_name):
        Eval.from_dataset(results, name=eval_name, columns=result_columns)

    prefix = MixClient().frontend_url

    print("Resources:")
    print(f"- Eval: {prefix(f'/evaluations/{eval_name}')}")
    print(f"- Eval set: {prefix(f'/datasets/{eval_set}')}")
    print(f"- Results dataset: {prefix(f'/datasets/{result_dataset}')}")


if __name__ == "__main__":
    main()
