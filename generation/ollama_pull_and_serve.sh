#!/usr/bin/env bash
set -e

# Start server in background
ollama serve &
PID=$!

until curl -sf http://localhost:11434/api/tags > /dev/null; do
  echo "wait for ollama..."
  sleep 1
done

echo "Ollama is running – load RAG Model: gte-qwen2-7b-instruct:f16"
ollama pull rjmalagon/gte-qwen2-7b-instruct:f16

kill $PID

# Start server in foreground
exec ollama serve
