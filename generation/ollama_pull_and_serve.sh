#!/usr/bin/env bash
set -e

# Start server im Hintergrund
ollama serve &
PID=$!

# Warte bis Ollama erreichbar ist
until curl -sf http://localhost:11434/api/tags > /dev/null; do
  echo "wait for ollama..."
  sleep 1
done

echo "Ollama is running – load RAG Model: gte-qwen2-7b-instruct:f16"
ollama pull rjmalagon/gte-qwen2-7b-instruct:f16

# Stoppe vorherige serve-process, dann neu starten
kill $PID

# Nun Server im Vordergrund
exec ollama serve