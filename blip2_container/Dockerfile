FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y git-lfs && rm -rf /var/lib/apt/lists/*
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model/  /app/model/
COPY run.py  /app/

ENTRYPOINT ["python", "run.py"]
