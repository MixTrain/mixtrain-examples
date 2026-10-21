FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir huggingface_hub==1.33.0
COPY download_flora.py /tmp/download_flora.py
RUN python /tmp/download_flora.py /opt/flora

RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir -r /opt/flora/batch_utils/requirements.txt
RUN PYTHONPATH=/opt/flora python -c "import batch_utils.pipeline; from peft import PeftModel"

ENV FLORA_RUNNER_DIR=/opt/flora
ENV FLORA_MODEL_CACHE=/data/flora/models
ENV HF_HOME=/data/flora/huggingface
ENV TOKENIZERS_PARALLELISM=false
