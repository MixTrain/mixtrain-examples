# Dataset Importer <a href="https://app.mixtrain.ai/new?from=https%3A%2F%2Fgithub.com%2FMixTrain%2Fmixtrain-examples%2Ftree%2Fmain%2Fdataset-importer&amp;type=workflow"><img src="https://mixtrain.ai/assets/run-with-mixtrain.svg" alt="Run with MixTrain" height="40" align="right"></a>

Import Hugging Face datasets into Mixtrain.

## What you'll learn

- How to import Hugging Face datasets into Mixtrain
- How to use Dataset argument in a workflow
- How to set column types for richer dataset column typing

## Prerequisites

- [mixtrain CLI installed and logged in](https://mixtrain.ai/docs/guide/quickstart)

## Run it

Create the workflow from this directory:

```bash
mixtrain workflow create . --name dataset-importer
```

Import a quick Hugging Face test dataset:

```bash
mixtrain workflow run dataset-importer \
  --input '{
    "hf_dataset": "cornell-movie-review-data/rotten_tomatoes",
    "target_name": "hf-rotten-tomatoes-test"
  }'
```

The workflow imports the data, saves the result as a Mixtrain dataset, and returns the created dataset. The dataset is linked directly from the workflow run in the Mixtrain UI.

## More examples

Import a dataset from Hugging Face and let Mixtrain infer column types:

```bash
mixtrain workflow run dataset-importer \
  --input '{
    "hf_dataset": "imdb",
    "split": "train",
    "target_name": "hf-imdb-test"
  }'
```

Import a multimodal VQA dataset from Hugging Face with image type:

```bash
mixtrain workflow run dataset-importer \
  --input '{
    "hf_dataset": "OneEyeDJ/Art-Vision-Question-Answering-Dataset",
    "target_name": "hf-art-vqa-test",
    "column_types": {
      "image_url": "image"
    }
  }'
```

The importer saves the Hugging Face split as a Mixtrain dataset. It also auto-detects common media URL columns, so `image_url` will be rendered as an image when the URLs have standard image extensions. Use `column_types` to override or customize display types. See the [Datasets guide](https://mixtrain.ai/docs/guide/datasets) for details.

## Inputs

- `hf_dataset`: Hugging Face dataset ID (required)
- `target_name`: output dataset name in Mixtrain (required)
- `description`: Description for the saved Mixtrain dataset (optional)
- `split`: Hugging Face split, such as `train`, `test`, or a slice like `train[:100]` (optional)
- `column_types`: optional display type overrides; by default, common media URL columns such as `image_url` are auto-detected. See the [Datasets guide](https://mixtrain.ai/docs/guide/datasets)
- `overwrite`: replace an existing dataset with the same name

## Hugging Face access

Public datasets work without extra setup. For private or gated Hugging Face datasets, add your Hugging Face token as a Mixtrain workspace secret named `HF_TOKEN`:

```bash
mixtrain secret set HF_TOKEN
```

Workflow runs automatically receive workspace secrets as environment variables, so this importer will use `HF_TOKEN` when loading Hugging Face datasets. See the [Mixtrain secrets guide](https://mixtrain.ai/docs/guide/secrets) for details.

## Learn more

- [Datasets guide](https://mixtrain.ai/docs/guide/datasets)
- [Workflows guide](https://mixtrain.ai/docs/guide/workflows)
