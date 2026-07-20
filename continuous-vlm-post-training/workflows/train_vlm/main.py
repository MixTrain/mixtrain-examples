"""Fine-tune a VLM with LoRA on two GPUs and register its best checkpoint."""

import tempfile
from pathlib import Path
from typing import TypedDict

from mixtrain import Checkpoint, Dataset, Markdown, MixFlow, Sandbox
from training import (
    epoch_metrics,
    load_log_history,
    prepare_splits,
    run_distributed_training,
)

DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_CHECKPOINT_DATASET = "vlm-checkpoints"
CHECKPOINT_ROOT = "/data/vlm-post-training-checkpoints"


class TrainingOutput(TypedDict):
    checkpoints: Dataset
    report: Markdown


class TrainVLM(MixFlow):
    """Distributed LoRA post-training that publishes its best checkpoint.

    Tags: example, training, vision, distributed, lora
    """

    _sandbox = Sandbox(
        image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime",
        gpu="A10G",
        gpu_per_node=2,
    )

    def run(
        self,
        training_data: Dataset,
        base_model: str = DEFAULT_BASE_MODEL,
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        checkpoint_dataset: str = DEFAULT_CHECKPOINT_DATASET,
    ) -> TrainingOutput:
        """Train on the pinned dataset version and register the best checkpoint.

        Args:
            training_data: Pinned version of the labeled source dataset
            base_model: Hugging Face image-text-to-text model to fine-tune
            num_epochs: Training epochs
            learning_rate: LoRA fine-tuning learning rate
            checkpoint_dataset: Registry dataset that receives the best checkpoint row
        """
        dataset_version = training_data.version
        experiment_id = f"vlm-data-v{dataset_version}"

        with tempfile.TemporaryDirectory(prefix="vlm-training-") as temp:
            root = Path(temp)
            dataset_path = root / "dataset"
            output_dir = root / "output"
            sample_count = prepare_splits(
                training_data.filter("split == 'train'"),
                dataset_path,
            )
            checkpoints = run_distributed_training(
                dataset_path=str(dataset_path),
                output_dir=str(output_dir),
                base_model=base_model,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
            )

            history = load_log_history(str(output_dir))
            metrics = {
                epoch: epoch_metrics(history, epoch)
                for epoch in range(1, num_epochs + 1)
            }
            measured = {
                epoch: values[1]
                for epoch, values in metrics.items()
                if values[1] is not None
            }
            best_epoch = min(measured, key=measured.get) if measured else num_epochs

            best_checkpoint = checkpoints[best_epoch - 1]
            _, best_validation_loss = metrics[best_epoch]
            checkpoint = Checkpoint.from_dir(
                str(best_checkpoint),
                f"{CHECKPOINT_ROOT}/{experiment_id}",
                task="image-text-to-text",
                source_run=experiment_id,
            )
            Dataset(checkpoint_dataset).append(
                [
                    {
                        "experiment_id": experiment_id,
                        "base_model": base_model,
                        "checkpoint": checkpoint,
                        "best_validation_loss": best_validation_loss,
                        "dataset_version": dataset_version,
                    }
                ]
            )

        report = _training_report(
            experiment_id,
            dataset_version,
            sample_count,
            base_model,
            metrics,
            best_epoch,
        )
        return {"checkpoints": Dataset(checkpoint_dataset), "report": report}


def _training_report(
    experiment_id: str,
    dataset_version: int,
    sample_count: int,
    base_model: str,
    metrics: dict[int, tuple[float | None, float | None]],
    best_epoch: int,
) -> Markdown:
    epoch_rows = "\n".join(
        f"| {epoch} | {training_loss or 'n/a'} | {validation_loss or 'n/a'} | "
        f"{'yes' if epoch == best_epoch else ''} |"
        for epoch, (training_loss, validation_loss) in metrics.items()
    )
    return Markdown(
        "\n".join(
            [
                "## VLM post-training complete",
                "",
                f"- **Experiment:** `{experiment_id}`",
                f"- **Training dataset:** v{dataset_version} ({sample_count} rows)",
                f"- **Base model:** `{base_model}`",
                "- **Distributed workers:** 2 GPUs",
                "",
                "| Epoch | Training loss | Validation loss | Best |",
                "| ---: | ---: | ---: | :---: |",
                epoch_rows,
            ]
        )
    )
