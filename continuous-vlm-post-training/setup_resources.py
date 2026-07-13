"""Create the datasets and Eval used by the continuous VLM example."""

from mixtrain import Checkpoint, Dataset, Eval, Image, MixClient

TRAINING_DATASET = "vlm-post-training-data"
CHECKPOINT_DATASET = "vlm-checkpoints"
RESULT_DATASET = "vlm-eval-results"
EVAL_NAME = "continuous-vlm-post-training"


def create_training_dataset() -> None:
    if Dataset.exists(TRAINING_DATASET):
        return

    Dataset.empty(
        ["sample_id", "split", "image", "prompt", "ground_truth"]
    ).with_column_types({"image": Image}).save(
        TRAINING_DATASET,
        description="Versioned multimodal data for continuous VLM post-training",
    )


def create_checkpoint_dataset() -> None:
    if Dataset.exists(CHECKPOINT_DATASET):
        return

    Dataset.empty(
        {
            "experiment_id": str,
            "base_model": str,
            "checkpoint": Checkpoint,
            "best_validation_loss": float,
            "dataset_version": int,
        }
    ).save(
        CHECKPOINT_DATASET,
        description="Best typed LoRA checkpoint for each completed VLM experiment",
    )


def create_result_dataset() -> None:
    schema: dict[str, type] = {
        "experiment_id": str,
        "image": Image,
        "prompt": str,
        "ground_truth": str,
        "base_answer": str,
        "finetuned_answer": str,
        "sample_id": str,
    }
    if Dataset.exists(RESULT_DATASET):
        return

    Dataset.empty(schema).save(
        RESULT_DATASET,
        description="Base-versus-checkpoint VQA results for each training experiment",
    )


def main() -> None:
    create_training_dataset()
    create_checkpoint_dataset()
    create_result_dataset()

    if not Eval.exists(EVAL_NAME):
        Eval.from_dataset(
            RESULT_DATASET,
            name=EVAL_NAME,
            description="Compare the base VLM with the fine-tuned checkpoint",
            columns=[
                "image",
                "prompt",
                "ground_truth",
                "base_answer",
                "finetuned_answer",
            ],
        )

    prefix = MixClient().frontend_url
    print("Resources:")
    print(f"- Eval: {prefix(f'/evaluations/{EVAL_NAME}')}")
    print(f"- Training data: {prefix(f'/datasets/{TRAINING_DATASET}')}")
    print(f"- Checkpoints: {prefix(f'/datasets/{CHECKPOINT_DATASET}')}")
    print(f"- Results: {prefix(f'/datasets/{RESULT_DATASET}')}")


if __name__ == "__main__":
    main()
