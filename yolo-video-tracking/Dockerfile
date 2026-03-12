FROM ultralytics/ultralytics:latest-cpu

# Install ffmpeg for H.264 video encoding
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
