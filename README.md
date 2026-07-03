# mixtrain examples

Hands-on examples for [mixtrain](https://mixtrain.ai) — from first workflow to distributed training and RL.

## Getting started

Follow the [quickstart guide](https://mixtrain.ai/docs/guide/quickstart) to install the CLI and log in, then clone this repo:

```bash
git clone https://github.com/MixTrain/mixtrain-examples.git
cd mixtrain-examples
```

## Examples

| Example | Description | Level | Tags | Run |
|---|---|---|---|---|
| [`hello-workflow`](hello-workflow/) | Run a GPU workflow and check CUDA availability | Intro | `workflow` | [![Run with MixTrain](https://mixtrain.ai/assets/run-with-mixtrain.svg)](https://app.mixtrain.ai/new?from=https%3A%2F%2Fgithub.com%2FMixTrain%2Fmixtrain-examples%2Ftree%2Fmain%2Fhello-workflow&type=workflow) |
| [`dataset-importer`](dataset-importer/) | Import Hugging Face datasets into Mixtrain datasets | Intro | `workflow`, `dataset`, `huggingface` | [![Run with MixTrain](https://mixtrain.ai/assets/run-with-mixtrain.svg)](https://app.mixtrain.ai/new?from=https%3A%2F%2Fgithub.com%2FMixTrain%2Fmixtrain-examples%2Ftree%2Fmain%2Fdataset-importer&type=workflow) |
| [`parameter-sweep`](parameter-sweep/) | Launch multiple runs of another workflow by sweeping one numeric input | Intermediate | `workflow`, `orchestration` | [![Run with MixTrain](https://mixtrain.ai/assets/run-with-mixtrain.svg)](https://app.mixtrain.ai/new?from=https%3A%2F%2Fgithub.com%2FMixTrain%2Fmixtrain-examples%2Ftree%2Fmain%2Fparameter-sweep&type=workflow) |
| [`yolo-object-detection`](yolo-object-detection/) | Run YOLO11 object detection on images | Intermediate | `vision`, `image`, `docker-image` | [![Run with MixTrain](https://mixtrain.ai/assets/run-with-mixtrain.svg)](https://app.mixtrain.ai/new?from=https%3A%2F%2Fgithub.com%2FMixTrain%2Fmixtrain-examples%2Ftree%2Fmain%2Fyolo-object-detection&type=model) |
| [`yolo-video-tracking`](yolo-video-tracking/) | Run YOLO11 object tracking on videos | Intermediate | `vision`, `video`, `Dockerfile` | [![Run with MixTrain](https://mixtrain.ai/assets/run-with-mixtrain.svg)](https://app.mixtrain.ai/new?from=https%3A%2F%2Fgithub.com%2FMixTrain%2Fmixtrain-examples%2Ftree%2Fmain%2Fyolo-video-tracking&type=model) |

## Resources

- [Documentation](https://mixtrain.ai/docs)
- [Website](https://mixtrain.ai)
