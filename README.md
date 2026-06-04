# mixtrain examples

Hands-on examples for [mixtrain](https://mixtrain.ai) — from first workflow to distributed training and RL.

## Getting started

Follow the [quickstart guide](https://mixtrain.ai/docs/guide/quickstart) to install the CLI and log in, then clone this repo:

```bash
git clone https://github.com/MixTrain/mixtrain-examples.git
cd mixtrain-examples
```

## Examples

| Example | Description | Level | Tags |
|---|---|---|---|
| [`hello-workflow`](hello-workflow/) | Run a GPU workflow and check CUDA availability | Intro | `workflow` |
| [`dataset-importer`](dataset-importer/) | Import Hugging Face datasets into Mixtrain datasets | Intro | `workflow`, `dataset`, `huggingface` |
| [`parameter-sweep`](parameter-sweep/) | Launch multiple runs of another workflow by sweeping one numeric input | Intermediate | `workflow`, `orchestration` |
| [`yolo-object-detection`](yolo-object-detection/) | Run YOLO11 object detection on images | Intermediate | `vision`, `image`, `docker-image` |
| [`yolo-video-tracking`](yolo-video-tracking/) | Run YOLO11 object tracking on videos | Intermediate | `vision`, `video`, `Dockerfile` |

## Resources

- [Documentation](https://mixtrain.ai/docs)
- [Website](https://mixtrain.ai)
