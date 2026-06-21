# Continuous Multimodal Eval

Keep your vision-language models continuously scored against a *living* eval
set. A [Routine](https://mixtrain.ai/docs/guide/routines) watches the eval set
and, every time new cases are added, scores them on each candidate model and
refreshes a side-by-side [Eval](https://mixtrain.ai/docs/guide/evaluations) —
no cron jobs, no manual reruns. Since the datasets are versioned, you can track and time travel each modification on both eval dataset and eval results automatically.

## What you'll learn

- How to trigger a `MixRoutine` on dataset appends with `on_added_rows`
- How to process **only the new rows**
- How to run multiple models in parallel with `Model.batch()`
- How to accumulate multimodal results and keep an `Eval` fresh

## How it works

```
vqa-eval-set ──append──▶ ContinuousVQAEval ──append results──▶ vqa-eval-results ──▶ Eval
```

Only the rows appended since the last firing are passed to the routine, so cost
scales with what changed — not the size of the whole eval set.

## Prerequisites

- [mixtrain CLI installed and logged in](https://mixtrain.ai/docs/guide/quickstart)
- Replace `model_names` in `continuous_eval.py` with vision-language models you want to evaluate before creating the result dataset and Routine.

## Run it

1. Create the eval dataset from the bundled CSV and its 10 example rows and local
images. This uploads local images and creates a new dataset:

    ```bash
    mixtrain dataset create vqa-eval-set initial_eval_set.csv \
    --description "A growing visual question-answer eval set"
    ```

2. Create the Eval. Eval is tied to a result dataset, so this will create a new empty result dataset and link it to the Eval:

    ```bash
    python setup_eval.py
    ```

3. Create the routine. Its `on_added_rows` trigger starts from the current
`vqa-eval-set` version, so every future append will trigger the routine and score the new rows:

    ```bash
    mixtrain routine create . --name continuous-vqa-eval
    ```

4. Append new rows to the eval dataset to simulate dataset additions and trigger the routine. This will score the new rows and append the answers to the result dataset. You can append rows from UI or programmatically using the SDK:

    ```bash
    python add_eval_samples.py
    ```

Open `continuous-vqa-eval` in the app to compare model answers side by side. Each run will also have a link to the eval as well as the result dataset.

## Using your own data

The eval dataset format is defined in `continuous_eval.py`. It defaults to these columns:

- `image` — a local image path or remote image URL
- `question` — the question to answer
- `ground_truth` — the expected answer

The simplest path is to replace the rows in `initial_eval_set.csv` and create a new eval set:

| image | question | ground_truth |
| --- | --- | --- |
| `images/my-dog.jpg` | What animal is shown? | dog |
| `images/storefront.png` | What color is the door? | blue |

You can also append rows directly from the SDK:

```python
from mixtrain import Dataset

Dataset("vqa-eval-set").append(
    [
        {
            "image": "/absolute/path/to/photo.jpg",
            "question": "How many people are visible?",
            "ground_truth": "three",
        },
        {
            "image": "https://example.com/public-image.png",
            "question": "What is the dominant color?",
            "ground_truth": "green",
        },
    ]
)
```

## Learn more

- [Routines & triggers guide](https://mixtrain.ai/docs/guide/routines)
- [Evaluations guide](https://mixtrain.ai/docs/guide/evaluations)
