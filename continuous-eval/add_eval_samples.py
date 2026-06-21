"""Add a couple of new eval examples the eval dataset."""

from mixtrain import Dataset

from continuous_eval import eval_set, ground_truth_col, image_col, question_col

new_examples = [
    {
        image_col: "images/cat.jpg",
        question_col: "Is the image about a cat or a bicycle?",
        ground_truth_col: "cat",
    },
    {
        image_col: "images/bicycle.jpg",
        question_col: "Is the transport shown a bicycle or a car?",
        ground_truth_col: "bicycle",
    },
]


def main() -> None:
    Dataset(eval_set).append(new_examples)
    print(f"Appended {len(new_examples)} examples to '{eval_set}'")


if __name__ == "__main__":
    main()
