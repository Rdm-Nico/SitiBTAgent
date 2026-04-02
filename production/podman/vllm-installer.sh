#!/bin/bash
# FILE DI CONFIGURAZIONE PER VLLM
# author: Nicolò Rossi
# questo file crea un pod Podman per avviare e monitorare l'istanza VLLM e la GPU
# Crea directory cache se non esiste
mkdir -p ~/.cache/huggingface

# Creiamo le variabili d'ambiente
GPU_ID="${GPU_ID:-gpu_1}"
COLLECTOR_ENDPOINT="${COLLECTOR_ENDPOINT:-http://172.18.31.215:4318}"


# Crea il pod
podman pod create --name vllm-services \
  -p 0.0.0.0:8000:8000 \
  -p 0.0.0.0:8001:8001

# Container per CHAT + Function Calling
podman run -d --pod vllm-services \
  --device nvidia.com/gpu=all \
  --name chat-model \
  --restart always \
  -v ~/.cache/huggingface:/root/.cache/huggingface:Z \
  -v /home/tecnico/training_Siti_agent_use_case/lora_modules:/lora_modules:Z\
  --env "HF_TOKEN=$HF_TOKEN" \
  --ipc=host \
  localhost/vllm/vllm-openai:working \
  --model unsloth/Qwen3-4B-Instruct-2507-bnb-4bit\
  --enable-lora \
  --lora-modules chat-tuned=/lora_modules/lora_adapter_rollout_10 \
  --gpu-memory-utilization 0.6 \
  --max-model-len 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --host 0.0.0.0 \
  --port 8000

# Container per EMBEDDINGS
podman run -d --pod vllm-services \
  --device nvidia.com/gpu=all \
  --name embedding-model \
  --restart always \
  -v ~/.cache/huggingface:/root/.cache/huggingface:Z \
  --env "HF_TOKEN=$HF_TOKEN" \
  --ipc=host \
  localhost/vllm/vllm-openai:working \
  --model google/embeddinggemma-300m \
  --max-model-len 2048 \
  --hf_overrides '{"matryoshka_dimensions":[128,256,512,768]}' \
  --gpu-memory-utilization 0.3 \
  --host 0.0.0.0 \
  --port 8001

# Container per GPU telemetry
podman run -d --pod vllm-services \
  --security-opt=label=disable \
  --device nvidia.com/gpu=all \
  --env "GPU_APPLICATION_NAME=172.18.31.178" \
  --env "GPU_ENVIROMENT=production" \
  --env "OTEL_EXPORTER_OTLP_ENDPOINT=$COLLECTOR_ENDPOINT" \
  --name otel-gpu-collector\
  --restart always \
  ghcr.io/openlit/otel-gpu-collector:latest


# sidecar per vllm monitoring
podman run -d --pod vllm-services \
  --name otel-agent \
  --env "HOSTNAME=$(hostname)" \
  --env "HOST_IP=172.18.31.178" \
  --env "COLLECTOR_ENDPOINT=$COLLECTOR_ENDPOINT" \
  -v $(pwd)/otel-agent-config.yaml:/etc/otelcol/config.yaml:ro,Z \
  --restart always \
  otel/opentelemetry-collector-contrib:latest \
  --config=/etc/otelcol/config.yaml

echo "✓ Servizi avviati!"

podman ps --pod --filter pod=vllm-services
