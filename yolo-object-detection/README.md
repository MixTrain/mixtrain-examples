# YOLO Object Detection [![Run with MixTrain](https://mixtrain.ai/assets/run-with-mixtrain.svg)](https://app.mixtrain.ai/new?from=https%3A%2F%2Fgithub.com%2FMixTrain%2Fmixtrain-examples%2Ftree%2Fmain%2Fyolo-object-detection&type=model)

Run YOLO11 object detection on images.

## What you'll learn

- How to create a model with a published Docker image
- How to use `Image` input/output types
- How to upload results to cloud storage with `Files`

## Prerequisites

- [mixtrain CLI installed and logged in](https://mixtrain.ai/docs/guide/quickstart)

## Run it

```bash
mixtrain model create yolo_detector.py --name yolo-detector
mixtrain model run yolo-detector \
  --input '{"image": "https://example.com/photo.jpg"}'
```

You can pass any public or private storage URL (https://, gs://, s3://) for the image. Any argument in the `run()` method can be overridden via the CLI or UI.

```bash
mixtrain model run yolo-detector \
  --input '{"image": "https://example.com/photo.jpg", "confidence_threshold": 0.3}'
```

Logs are streamed to the CLI and also available on the UI. The run URL will be printed on the CLI:

```
https://app.mixtrain.ai/<workspace>/models/yolo-detector/runs/1
```

## Learn more

- [Models guide](https://mixtrain.ai/docs/guide/models)
- [Input/Output types](https://mixtrain.ai/docs/guide/types)
