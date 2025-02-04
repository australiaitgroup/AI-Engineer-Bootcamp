#!/bin/sh
echo $MODEL
# Start the ollama server in the background
ollama serve &

# Give the server a short delay to initialize (adjust as needed)
sleep 5

# # Pull the nemotron-mini model
ollama pull nemotron-mini
wait