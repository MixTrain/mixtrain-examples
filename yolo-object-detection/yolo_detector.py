"""YOLO11 Object Detection Model Example."""

import json
import os
import tempfile
import uuid
from datetime import UTC, datetime
from typing import Literal

from mixtrain import File, JSON, Image, MixModel, Sandbox


class YOLODetector(MixModel):
    """YOLO object detection model.

    Takes an image, runs YOLO11 detection, stores results to cloud storage,
    and returns annotated image with bounding boxes.
    """

    # Sandbox configuration - use ultralytics Docker image (use :latest for GPU support)
    _sandbox = Sandbox(image="ultralytics/ultralytics:latest-cpu")

    def setup(
        self,
        model_name: Literal[
            "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"
        ] = "yolo11n",
    ):
        from ultralytics import YOLO

        self.model = YOLO(f"{model_name}.pt")

    def run(
        self,
        image: Image,
        confidence_threshold: float = 0.5,
        output_path: str | None = None,
    ) -> dict:
        """Run object detection on input image.

        Args:
            image: Input image to process
            confidence_threshold: Minimum confidence threshold for detections
            output_path: Base path for storing results (auto-generated if not provided)

        Returns:
            dict with:
                - annotated_image: Image with bounding boxes drawn
                - detections: JSON with list of detected objects
                - detection_count: Number of objects detected
                - results_url: URL where results JSON is stored
        """
        # Use provided output_path or generate one
        base_path = output_path
        if not base_path:
            run_id = (
                f"yolo_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_"
                f"{uuid.uuid4().hex[:8]}"
            )
            base_path = f"detections/{run_id}"
        base_path = base_path.rstrip("/")

        # Download image using Image helper method
        input_path = image.to_file()
        ext = os.path.splitext(input_path)[1] or ".jpg"

        # Run YOLO detection
        results = self.model(input_path, conf=confidence_threshold)
        result = results[0]

        # Extract detection data
        detections = []
        for box in result.boxes:
            detections.append(
                {
                    "class": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                }
            )

        # Save annotated image (YOLO saves as same format)
        local_output = input_path.replace(ext, f"_annotated{ext}")
        result.save(local_output)

        # Upload annotated image to cloud storage
        annotated_image = Image.from_file(local_output).save(f"{base_path}/annotated{ext}")

        # Upload detection results JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "source_url": image.url,
                    "confidence_threshold": confidence_threshold,
                    "detections": detections,
                    "count": len(detections),
                },
                f,
                indent=2,
            )
            results_json_path = f.name

        results_info = File.from_file(
            results_json_path,
            content_type="application/json",
        ).save(f"{base_path}/results.json")

        return {
            "annotated_image": annotated_image,
            "detections": JSON(data=detections),
            "detection_count": len(detections),
            "results_url": results_info.url,
        }
