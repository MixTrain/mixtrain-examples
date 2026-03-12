"""ML Hello World - Check CUDA/GPU availability."""

from mixtrain import MixFlow, Sandbox


class ExampleWorkflow(MixFlow):
    """ML Hello World workflow - checks CUDA device availability."""

    _sandbox = Sandbox(gpu="T4")

    def run(self):
        """Check CUDA availability and device info."""
        import subprocess

        print("Hello World from mixtrain!")

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            gpus = result.stdout.strip().split("\n")
            print(f"CUDA available: True")
            print(f"Device count: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  Device {i}: {gpu}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("CUDA available: False")
            print("No CUDA devices found")
