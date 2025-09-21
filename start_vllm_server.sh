#!/bin/bash

# Start vLLM server script for ReadingCorpus project
# This script starts the vLLM OpenAI-compatible API server

echo "🚀 Starting vLLM Server for ReadingCorpus"
echo "=========================================="

# Set environment variable to allow longer max model length
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Model path
MODEL_PATH="/root/autodl-tmp/ReadingCorpus/models/meta-llama/Meta-Llama-3-8B-Instruct/"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found at: $MODEL_PATH"
    echo "Please make sure the model is downloaded correctly."
    exit 1
fi

echo "📁 Model path: $MODEL_PATH"
echo "🔧 Starting server with max_model_len=8192..."

# Start the vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --max-model-len 8192 \
    --served-model-name llama8b \
    --host 0.0.0.0 \
    --port 8000

echo "✅ vLLM server started successfully!"
echo "🌐 API endpoint: http://localhost:8000/v1"
echo "📖 Model name: llama8b"
echo ""
echo "💡 You can now run AskCorpus.py with:"
echo "   python src/core/AskCorpus.py --dataset musique --corpus musique_sample_corpus.jsonl --workers 4"
echo "   python src/core/AskCorpus.py --dataset 2wiki --corpus 2wiki_sample_corpus.jsonl --workers 4"
echo "   python src/core/AskCorpus.py --dataset hotpotqa --corpus hotpotqa_sample_corpus.jsonl --workers 4"