"""Download NVIDIA's FLORA pipeline code at a pinned revision. Weights download on first run."""

from pathlib import Path

REPO = "nvidia/Flora-TempSeg"
REVISION = "78a0e4ac75d33509f077fdf985ebb51153920e7c"


def download_runner(destination: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        REPO,
        revision=REVISION,
        local_dir=destination,
        allow_patterns=["batch_utils/*", "infer.py", "LICENSE.pdf"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(download_runner(str(args.destination)))
