#!/bin/bash

trap "kill 0 2>/dev/null; wait 2>/dev/null" EXIT

echo "[backend] Starting Flask..."
./venv/bin/python -m backend.app &
BACKEND_PID=$!

echo "[backend] Waiting for port 5000..."
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:5000 >/dev/null 2>&1; then
        echo "[backend] Ready."
        break
    fi
    sleep 1
done
if ! curl -s http://127.0.0.1:5000 >/dev/null 2>&1; then
    echo "[backend] Failed to start"
    exit 1
fi

echo "[frontend] Starting frontend dev server..."
cd frontend && npm run dev &
FRONTEND_PID=$!

if ! wait -n; then
    echo "[frontend] Failed to start"
    exit 1
fi

wait
