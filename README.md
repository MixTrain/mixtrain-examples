# mixtrain examples

Hands-on examples for [mixtrain](https://mixtrain.ai) — from first workflow to distributed training and RL.

## Getting started

Follow the [quickstart guide](https://mixtrain.ai/docs/guide/quickstart) to install the CLI and log in, then clone this repo:

```bash
git clone https://github.com/MixTrain/mixtrain-examples.git
cd mixtrain-examples
```

## Examples

| Example | Description | Level | Run with MixTrain |
|---|---|---|---|
| [`hello-workflow`](hello-workflow/) | Run a GPU workflow and check CUDA availability | Intro | [![Run with MixTrain](https://mixtrain.ai/assets/run-with-mixtrain.svg)](https://app.mixtrain.ai/new?from=https%3A%2F%2Fgithub.com%2FMixTrain%2Fmixtrain-examples%2Ftree%2Fmain%2Fhello-workflow&type=workflow) |
| [`daily-arxiv-digest`](daily-arxiv-digest/) | Schedule a weekday digest of recent arXiv papers | Intro | [![Run with MixTrain](https://mixtrain.ai/assets/run-with-mixtrain.svg)](https://app.mixtrain.ai/new?from=https%3A%2F%2Fgithub.com%2FMixTrain%2Fmixtrain-examples%2Ftree%2Fmain%2Fdaily-arxiv-digest&type=routine) |
| [`dataset-importer`](dataset-importer/) | Import Hugging Face datasets into Mixtrain datasets | Intro | [![Run with MixTrain](https://mixtrain.ai/assets/run-with-mixtrain.svg)](https://app.mixtrain.ai/new?from=https%3A%2F%2Fgithub.com%2FMixTrain%2Fmixtrain-examples%2Ftree%2Fmain%2Fdataset-importer&type=workflow) |
| [`compare-image-models`](compare-image-models/) | Run your prompts through several image models and compare the results visually | Intro | [![Run with MixTrain](https://mixtrain.ai/assets/run-with-mixtrain.svg)](https://app.mixtrain.ai/new?from=https%3A%2F%2Fgithub.com%2FMixTrain%2Fmixtrain-examples%2Ftree%2Fmain%2Fcompare-image-models&type=workflow) |
| [`parameter-sweep`](parameter-sweep/) | Launch multiple runs of another workflow by sweeping one numeric input | Intermediate | [![Run with MixTrain](https://mixtrain.ai/assets/run-with-mixtrain.svg)](https://app.mixtrain.ai/new?from=https%3A%2F%2Fgithub.com%2FMixTrain%2Fmixtrain-examples%2Ftree%2Fmain%2Fparameter-sweep&type=workflow) |
| [`yolo-object-detection`](yolo-object-detection/) | Run YOLO11 object detection on images | Intermediate | [![Run with MixTrain](https://mixtrain.ai/assets/run-with-mixtrain.svg)](https://app.mixtrain.ai/new?from=https%3A%2F%2Fgithub.com%2FMixTrain%2Fmixtrain-examples%2Ftree%2Fmain%2Fyolo-object-detection&type=model) |
| [`yolo-video-tracking`](yolo-video-tracking/) | Run YOLO11 object tracking on videos | Intermediate | [![Run with MixTrain](https://mixtrain.ai/assets/run-with-mixtrain.svg)](https://app.mixtrain.ai/new?from=https%3A%2F%2Fgithub.com%2FMixTrain%2Fmixtrain-examples%2Ftree%2Fmain%2Fyolo-video-tracking&type=model) |
| [`continuous-eval`](continuous-eval/) | Auto-score VLMs whenever an eval set grows, via a dataset trigger | Intermediate | [![Run with MixTrain](https://mixtrain.ai/assets/run-with-mixtrain.svg)](https://app.mixtrain.ai/new?from=https%3A%2F%2Fgithub.com%2FMixTrain%2Fmixtrain-examples%2Ftree%2Fmain%2Fcontinuous-eval&type=routine&exclude=images**) |
| [`continuous-vlm-post-training`](continuous-vlm-post-training/) | Automatically retrain a VLM whenever new data is added and evaluate | Advanced | [See README](continuous-vlm-post-training/README.md) |

## Resources

- [Documentation](https://mixtrain.ai/docs)
- [Website](https://mixtrain.ai)
