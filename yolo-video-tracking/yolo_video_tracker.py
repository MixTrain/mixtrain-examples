"""YOLO Video Object Tracking Example.

Demonstrates video object tracking using YOLO with ultralytics.
Supports ByteTrack and BOTSORT tracking algorithms.
"""

import json
import os
import subprocess
import tempfile
import uuid
from datetime import UTC, datetime
from typing import Literal

from mixtrain import JSON, Files, MixModel, Sandbox, Video


class YOLOVideoTracker(MixModel):
    """YOLO video object tracking model.

    Takes a video URL, runs YOLO tracking with ByteTrack or BOTSORT,
    stores results to cloud storage, and returns annotated video with
    tracking bounding boxes and IDs overlaid.
    """

    def setup(
        self,
        model_name: Literal[
            "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"
        ] = "yolo11n",
    ):
        from ultralytics import YOLO

        self.model = YOLO(f"{model_name}.pt")
        self.files = Files()

    def run(
        self,
        video: Video,
        confidence_threshold: float = 0.5,
        tracker: Literal["bytetrack", "botsort"] = "bytetrack",
        iou_threshold: float = 0.5,
        output_path: str | None = None,
        classes: list[int] | None = None,
    ) -> dict:
        """Run object tracking on input video.

        Args:
            video: Video to process
            confidence_threshold: Minimum confidence threshold for detections
            tracker: Tracking algorithm ('bytetrack' or 'botsort')
            iou_threshold: IOU threshold for tracking association
            output_path: Base path for storing results (auto-generated if not provided)
            classes: List of class IDs to track (e.g., [0, 2, 7] for person, car, truck).
                     Defaults to [0, 2, 7] if not provided.

        Returns:
            dict with:
                - annotated_video: Video with tracking bboxes and IDs overlaid
                - tracking_data: JSON with per-frame tracking data
                - track_count: Number of unique tracks
                - frame_count: Total frames processed
                - results_url: Cloud storage URL for tracking JSON
        """
        if classes is None:
            classes = [0, 2, 7]  # person, car, truck

        # Use provided output_path or generate one
        base_path = output_path
        if not base_path:
            run_id = (
                f"tracking_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_"
                f"{uuid.uuid4().hex[:8]}"
            )
            base_path = f"tracking/{run_id}"
        base_path = base_path.rstrip("/")

        # Download video using Video type's to_file() method
        input_path = video.to_file()

        # Create temp directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker_config = f"{tracker}.yaml"

            track_kwargs = {
                "source": input_path,
                "conf": confidence_threshold,
                "iou": iou_threshold,
                "tracker": tracker_config,
                "persist": True,
                "save": True,
                "project": temp_dir,
                "name": "track_output",
                "exist_ok": True,
            }

            if classes:
                track_kwargs["classes"] = classes

            results = self.model.track(**track_kwargs)

            # Extract tracking data from results
            tracking_frames = []
            all_track_ids = set()

            for frame_idx, result in enumerate(results):
                frame_detections = []

                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        track_id = int(boxes.id[i]) if boxes.id is not None else -1
                        class_id = int(boxes.cls[i])
                        confidence = float(boxes.conf[i])
                        bbox = boxes.xyxy[i].tolist()

                        if track_id >= 0:
                            all_track_ids.add(track_id)

                        frame_detections.append(
                            {
                                "track_id": track_id,
                                "class": result.names[class_id],
                                "class_id": class_id,
                                "confidence": round(confidence, 4),
                                "bbox": [round(x, 2) for x in bbox],
                            }
                        )

                tracking_frames.append(
                    {
                        "frame_number": frame_idx,
                        "detections": frame_detections,
                    }
                )

            tracking_data = {
                "source_url": video.url,
                "tracker": tracker,
                "confidence_threshold": confidence_threshold,
                "iou_threshold": iou_threshold,
                "total_tracks": len(all_track_ids),
                "total_frames": len(tracking_frames),
                "frames": tracking_frames,
            }

            # Find the output video (saved by YOLO)
            output_dir = os.path.join(temp_dir, "track_output")
            output_video_path = None

            for file in os.listdir(output_dir):
                if file.endswith((".mp4", ".avi", ".mov", ".mkv")):
                    output_video_path = os.path.join(output_dir, file)
                    break

            if not output_video_path:
                raise RuntimeError("Failed to generate annotated video")

            # Convert to MP4 if not already (ensures browser compatibility)
            if not output_video_path.lower().endswith(".mp4"):
                mp4_path = os.path.join(temp_dir, "output.mp4")
                self._convert_to_mp4(output_video_path, mp4_path)
                output_video_path = mp4_path

            # Upload annotated video to cloud storage
            video_info = self.files.upload(
                output_video_path,
                f"{base_path}/annotated_video.mp4",
                content_type="video/mp4",
            )

            # Upload tracking JSON to cloud storage
            json_path = os.path.join(temp_dir, "tracking_data.json")
            with open(json_path, "w") as f:
                json.dump(tracking_data, f, indent=2)

            results_info = self.files.upload(
                json_path,
                f"{base_path}/tracking_data.json",
                content_type="application/json",
            )

            return {
                "annotated_video": Video(url=video_info.url),
                "tracking_data": JSON(data=tracking_data),
                "track_count": len(all_track_ids),
                "frame_count": len(tracking_frames),
                "results_url": results_info.url,
            }

    def _convert_to_mp4(self, input_path: str, output_path: str) -> None:
        """Convert video to MP4 (H.264) format using ffmpeg."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
