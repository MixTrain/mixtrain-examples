"""Split a video into timestamped actions and caption each one."""

import os
import sys
import tempfile
from pathlib import Path
from typing import Literal

from mixtrain import MixFlow, Sandbox, Video


def trim(source: str, destination: Path, start_s: float, end_s: float | None) -> Path:
    """Re-encode the frames in [start_s, end_s), keeping their original timing."""
    import av

    with av.open(source) as reader, av.open(str(destination), "w") as writer:
        stream = reader.streams.video[0]
        output = writer.add_stream("libx264", rate=stream.average_rate)
        output.width = stream.codec_context.width
        output.height = stream.codec_context.height
        output.pix_fmt = "yuv420p"
        output.time_base = stream.time_base
        output.options = {"crf": "18"}
        if start_s > 0:
            reader.seek(int(start_s / stream.time_base), stream=stream, backward=True)
        first_pts = None
        for frame in reader.decode(stream):
            if frame.time < start_s:
                continue
            if end_s is not None and frame.time >= end_s:
                break
            first_pts = frame.pts if first_pts is None else first_pts
            frame.pts -= first_pts
            frame.time_base = stream.time_base
            for packet in output.encode(frame):
                writer.mux(packet)
        for packet in output.encode():
            writer.mux(packet)
    if first_pts is None:
        raise ValueError(f"No frames between {start_s}s and {end_s}s")
    return destination


class AutoSegmentCaption(MixFlow):
    """Find the actions in a video and caption each one with NVIDIA FLORA.

    Tags: video, robotics, captioning, data-pipeline
    """

    _sandbox = Sandbox(gpu="A100-80GB", cpu=8, memory=131072, timeout=3600)

    def run(
        self,
        video: Video,
        start_s: float = 0.0,
        end_s: float | None = None,
        profile: Literal["human", "yam", "dex", "tianji"] = "human",
        instruction: str = "",
        left_wrist: Video | None = None,
        right_wrist: Video | None = None,
        yam32: bool = False,
        include_audit: bool = False,
    ) -> dict:
        """Segment a clip into actions and caption each segment.

        Args:
            video: Main camera video, up to 120 seconds after trimming.
            start_s: Start of the clip to process, in seconds.
            end_s: End of the clip to process, in seconds. Defaults to the end of the video.
            profile: Embodiment: human (egocentric), yam, dex, or tianji.
            instruction: Optional task description, e.g. "Put the cube in the box."
            left_wrist: Optional left wrist camera, synchronized with video.
            right_wrist: Optional right wrist camera, synchronized with video.
            yam32: Use the 32-second, three-camera YAM layout.
            include_audit: Include the pipeline's intermediate records.
        """
        import torch

        if torch.cuda.get_device_properties(0).total_memory < 75 * 1024**3:
            raise RuntimeError("FLORA needs an 80 GB GPU, e.g. --gpu A100-80GB")

        sys.path.insert(0, os.environ.get("FLORA_RUNNER_DIR", "/opt/flora"))
        from batch_utils import DataEnginePipeline

        with tempfile.TemporaryDirectory() as workdir:

            def clip(camera: Video | None, name: str) -> Path | None:
                if camera is None:
                    return None
                return trim(camera.to_file(), Path(workdir) / name, start_s, end_s)

            return DataEnginePipeline().annotate(
                clip(video, "main.mp4"),
                profile=profile,
                left_wrist=clip(left_wrist, "left_wrist.mp4"),
                right_wrist=clip(right_wrist, "right_wrist.mp4"),
                yam32=yam32,
                instruction=instruction,
                include_audit=include_audit,
            )
