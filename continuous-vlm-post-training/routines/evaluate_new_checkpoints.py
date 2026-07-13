"""Evaluate completed LoRA checkpoints with generated serving models."""

from mixtrain import (
    Checkpoint,
    Dataset,
    Markdown,
    MixRoutine,
    Model,
    on_added_rows,
)

TRAINING_DATASET = "vlm-post-training-data"
CHECKPOINT_DATASET = "vlm-checkpoints"
RESULT_DATASET = "vlm-eval-results"
MODEL_PREFIX = "vlm-checkpoint-eval"


class EvaluateNewCheckpoints(MixRoutine):
    """Evaluate every completed training experiment as serving models.

    Tags: example, evaluation, vision, continuous
    """

    def run(
        self,
        new_checkpoints=on_added_rows(CHECKPOINT_DATASET),
    ) -> list[Markdown]:
        reports = []
        for checkpoint in new_checkpoints.select(["experiment_id"]):
            report = _evaluate_experiment(checkpoint["experiment_id"])
            print(f"Evaluated experiment {checkpoint['experiment_id']}")
            reports.append(report)
        return reports


def _model_name(experiment_id: str, candidate: str) -> str:
    return f"{MODEL_PREFIX}-{experiment_id}-{candidate}".replace("_", "-")


def _evaluate_experiment(experiment_id: str) -> Markdown:
    checkpoint = (
        Dataset(CHECKPOINT_DATASET)
        .filter(lambda row: row["experiment_id"] == experiment_id)
        .head(1)
        .to_pylist()[0]
    )
    base_model = checkpoint["base_model"]
    dataset_version = int(checkpoint["dataset_version"])

    candidates: dict[str, Checkpoint] = {
        "base": Checkpoint(base_model, task="image-text-to-text"),
        "finetuned": checkpoint["checkpoint"],
    }

    eval_data = Dataset(TRAINING_DATASET, version=dataset_version).filter(
        "split == 'eval'"
    )
    models = {
        candidate: Model.create(
            _model_name(experiment_id, candidate),
            checkpoint=checkpoint,
            sandbox={"gpu": "A10G"},
        )
        for candidate, checkpoint in candidates.items()
    }
    model_names = [model.name for model in models.values()]
    batch_rows = Model.batch(
        model_names,
        _inference_inputs(eval_data),
        input_columns=["prompt", "images", "max_tokens", "temperature"],
        error_col="model_errors",
    ).to_pylist()
    result_rows = _result_rows(experiment_id, batch_rows, models)

    # Publish both model answers for the experiment together.
    Dataset(RESULT_DATASET).append(result_rows)

    return _evaluation_report(experiment_id, dataset_version, len(result_rows))


def _inference_inputs(dataset: Dataset):
    for row in dataset:
        yield {
            **row,
            "images": row["image"],
            "max_tokens": 12,
            "temperature": 0.0,
        }


def _result_rows(
    experiment_id: str,
    batch_rows: list[dict],
    models: dict[str, Model],
) -> list[dict]:
    results = []
    for row in batch_rows:
        result = {
            "experiment_id": experiment_id,
            "sample_id": row["sample_id"],
            "image": row["image"],
            "prompt": row["prompt"],
            "ground_truth": row["ground_truth"],
        }
        for candidate, model in models.items():
            answer = row[model.name]
            result[f"{candidate}_answer"] = answer
        results.append(result)
    return results


def _evaluation_report(
    experiment_id: str,
    dataset_version: int,
    sample_count: int,
) -> Markdown:
    return Markdown(
        "\n".join(
            [
                "## Checkpoint evaluation complete",
                "",
                f"- **Experiment:** `{experiment_id}`",
                f"- **Held-out dataset:** v{dataset_version}",
                f"- **Examples:** {sample_count}",
            ]
        )
    )
