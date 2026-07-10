"""Continuous multimodal evaluation, driven by a dataset trigger.

A good eval set is never "done" — your team keeps adding hard cases as you find
them. This routine turns that growing eval set into a continuous evaluation:
every time new rows land in the eval set, the candidate vision-language models
are scored on just those rows, and a side-by-side Eval is refreshed so you can
review answers as they come in.

Create a Routine and you get a continuous evaluation:

    mixtrain routine create . --name continuous-vqa-eval
"""

from mixtrain import Dataset, Eval, MixRoutine, Model, on_added_rows

eval_set = "vqa-eval-set"
result_dataset = "vqa-eval-results"
eval_name = "continuous-vqa-eval"

# Replace these with your own model names
model_names = ["baseline_vlm", "candidate_vlm"]

image_col = "image"
question_col = "question"
ground_truth_col = "ground_truth"
input_columns = [image_col, question_col, ground_truth_col]
result_columns = [*input_columns, *model_names]


class ContinuousVQAEval(MixRoutine):
    """Re-score candidate VLMs whenever the eval set grows.

    The eval set is a multimodal VQA dataset: each row holds an ``image`` to
    reason over, a ``question`` to answer, and a ``ground_truth`` answer to grade
    against.

    Tags: example, eval, vision
    """

    def run(
        self,
        new_examples=on_added_rows(eval_set, batch_rows=1),
    ) -> Eval:
        """Score newly appended eval cases on the candidate models.

        Args:
            new_examples: Read-only Dataset containing rows appended since the last run
        """

        # Run the models on the new cases in parallel. Returns a Dataset.
        result = Model.batch(
            model_names, new_examples, input_columns=[image_col, question_col]
        )
        # Select the columns we need for eval
        results = result.select(result_columns)
        # Append to the results dataset, eval will update automatically

        Dataset(result_dataset).append(results)

        print(f"Eval '{eval_name}' updated with {len(results)} comparison(s)")
        return Eval(eval_name)
