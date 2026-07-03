<a href="https://app.mixtrain.ai/new?from=https%3A%2F%2Fgithub.com%2FMixTrain%2Fmixtrain-examples%2Ftree%2Fmain%2Fyolo-video-tracking&amp;type=model"><img src="https://mixtrain.ai/assets/run-with-mixtrain.svg" alt="Run with MixTrain" height="40" align="right"></a>

# YOLO Video Tracking

Object tracking on videos using YOLO11.

## What you'll learn

- How to create a model with a custom Dockerfile
- How to use `Video` input/output types
- How to use `Files` for cloud storage uploads

## Prerequisites

- [mixtrain CLI installed and logged in](https://mixtrain.ai/docs/guide/quickstart)

## Run it

```bash
mixtrain model create yolo_video_tracker.py Dockerfile --name yolo-tracker
mixtrain model run yolo-tracker \
  --input '{"video": "https://example.com/traffic.mp4"}'
```

You can pass any public or private storage URL (https://, gs://, s3://) for the video. Any argument in the `run()` method can be overridden via the CLI or UI.

```bash
mixtrain model run yolo-tracker \
  --input '{"video": "https://example.com/traffic.mp4", "tracker": "botsort", "classes": [0, 1, 2], "confidence_threshold": 0.3}'
```

Logs are streamed to the CLI and also available on the UI. The run URL will be printed on the CLI:

```
https://app.mixtrain.ai/<workspace>/models/yolo-tracker/runs/1
```

## Learn more

- [Models guide](https://mixtrain.ai/docs/guide/models)
- [Input/Output types](https://mixtrain.ai/docs/guide/types)
