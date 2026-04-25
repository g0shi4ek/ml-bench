#!/usr/bin/env bash
# scripts/measure_docker.sh
# Измеряет размеры Docker-образов и RAM-потребление во время выполнения.

set -e

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"
OUTPUT="$RESULTS_DIR/docker_metrics.json"

echo "╔══════════════════════════════════════════════╗"
echo "║  ML-Bench Docker Metrics Measurement         ║"
echo "╚══════════════════════════════════════════════╝"

# ── 1. Сборка образов ──────────────────────────────────────────────────────
echo ""
echo "[1/4] Сборка образов..."
docker compose build --quiet 2>/dev/null || {
    echo "  Ошибка сборки. Убедитесь, что Docker запущен."
    exit 1
}
echo "  Готово."

# ── 2. Размеры образов ────────────────────────────────────────────────────
echo ""
echo "[2/4] Размеры Docker-образов:"

GO_IMAGE_SIZE=$(docker image inspect ml-bench/go-bench:latest \
    --format='{{.Size}}' 2>/dev/null || echo "0")
PY_IMAGE_SIZE=$(docker image inspect ml-bench/python-bench:latest \
    --format='{{.Size}}' 2>/dev/null || echo "0")

GO_SIZE_MB=$(echo "scale=1; $GO_IMAGE_SIZE / 1048576" | bc)
PY_SIZE_MB=$(echo "scale=1; $PY_IMAGE_SIZE / 1048576" | bc)

echo "  Go     образ: ${GO_SIZE_MB} МБ"
echo "  Python образ: ${PY_SIZE_MB} МБ"

if [ "$GO_IMAGE_SIZE" -gt 0 ] 2>/dev/null; then
    RATIO=$(echo "scale=1; $PY_IMAGE_SIZE / $GO_IMAGE_SIZE" | bc)
    echo "  Соотношение:  Python / Go = ${RATIO}x"
fi

echo ""
echo "[3/4] Время холодного старта (cold start):"

# Go: alpine образ имеет shell, замеряем время запуска бенчмарка
# Запускаем с --n=1 для минимального времени выполнения
GO_START=$(docker run --rm \
    -v "$(pwd)/results:/app/results" \
    ml-bench/go-bench:latest \
    --n 1 --output /dev/null 2>&1 | head -1 || echo "N/A")
echo "  Go  cold start: запуск успешен"
echo "    Первая строка вывода: ${GO_START}"

# Python: замеряем время импорта
PY_START_SEC=$(docker run --rm \
    ml-bench/python-bench:latest \
    python -c "
import time
start = time.time()
import torch
import sklearn
import numpy
elapsed = (time.time() - start) * 1000
print(f'{elapsed:.0f}')
" 2>/dev/null || echo "N/A")
echo "  Python cold start (import torch+sklearn+numpy): ~${PY_START_SEC} мс"

# ── 4. RAM-потребление ────────────────────────────────────────────────────
echo ""
echo "[4/4] Замер RAM:"
echo ""
echo "  Для замера RAM во время выполнения бенчмарков:"
echo ""
echo "  Терминал 1:"
echo "    docker compose up go-bench python-bench"
echo ""
echo "  Терминал 2 (пока бенчмарки работают):"
echo "    docker stats --no-stream --format 'table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}'"
echo ""

# ── Запись метрик в JSON ──────────────────────────────────────────────────
RATIO_VAL="0"
if [ "$GO_IMAGE_SIZE" -gt 0 ] 2>/dev/null; then
    RATIO_VAL=$(echo "scale=1; $PY_IMAGE_SIZE / $GO_IMAGE_SIZE" | bc)
fi

cat > "$OUTPUT" << EOF
{
  "docker_image_sizes": {
    "go_bench_bytes": $GO_IMAGE_SIZE,
    "go_bench_mb": $GO_SIZE_MB,
    "python_bench_bytes": $PY_IMAGE_SIZE,
    "python_bench_mb": $PY_SIZE_MB,
    "ratio_python_go": $RATIO_VAL
  },
  "notes": {
    "go_base_image":     "alpine:3.19 (multi-stage build)",
    "go_builder_image":  "golang:1.22-alpine",
    "python_base_image": "python:3.11-slim",
    "python_torch":      "CPU-only (pytorch.org/whl/cpu)",
    "measurement_date":  "$(date -Iseconds 2>/dev/null || date)"
  }
}
EOF

echo ""
echo "  Метрики сохранены: $OUTPUT"
echo ""
echo "┌──────────────────────────────────────────────────┐"
echo "│  Полный запуск бенчмарков + сравнение:            │"
echo "│    docker compose up --build                     │"
echo "│                                                  │"
echo "│  Только замер размеров:                          │"
echo "│    docker images | grep ml-bench                 │"
echo "└──────────────────────────────────────────────────┘"
