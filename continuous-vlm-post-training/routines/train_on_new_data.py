"""Submit VLM post-training whenever labeled training data is added."""

from mixtrain import Dataset, MixRoutine, Workflow, on_added_rows

TRAINING_DATASET = "vlm-post-training-data"
TRAINING_WORKFLOW = "train-vlm"


class TrainOnNewData(MixRoutine):
    """Start a new experiment when the training split grows.

    Tags: example, training, vision, continuous
    """

    def run(
        self,
        new_rows=on_added_rows(TRAINING_DATASET),
        training_workflow: str = TRAINING_WORKFLOW,
    ) -> Workflow | None:
        dataset_version = new_rows.version
        if new_rows.filter("split == 'train'").head(1).collect().num_rows == 0:
            print(f"Dataset v{dataset_version} added no training rows; skipping")
            return None

        run = Workflow(training_workflow).submit(
            inputs={
                "training_data": Dataset(TRAINING_DATASET, version=dataset_version),
            }
        )
        run_number = run["run_number"]
        print(
            f"Submitted '{training_workflow}' run #{run_number} "
            f"for dataset v{dataset_version}"
        )
        return Workflow(training_workflow, run_number=run_number)
