#!/bin/bash

source ~/.pyenv/versions/venv/bin/activate

echo "🔍 Checking for processes on port 5000..."
PID=$(lsof -t -i:5000)

if [ -n "$PID" ]; then
  echo "Killing process on port 5000 (PID: $PID)..."
  kill -9 $PID
else
  echo "No process currently using port 5000."
fi

echo "🚀 Starting Gunicorn server..."
~/.pyenv/versions/venv/bin/gunicorn FlaskServer.wsgi:app \
  --bind 0.0.0.0:5000 \
  --workers 1 \
  --timeout 120 \
  --log-level info

