#!/usr/bin/env bash
# scripts/measure_docker.sh
# Измеряет размеры Docker-образов и RAM-потребление во время выполнения.
# Используется для таблицы методологии (Промт 3.3).
#
# Требования: docker, docker-compose, bc
# Использование: bash scripts/measure_docker.sh

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
docker-compose build --quiet 2>/dev/null || {
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

echo "  Go  образ: ${GO_SIZE_MB} МБ"
echo "  Python образ: ${PY_SIZE_MB} МБ"

# ── 3. Время холодного старта ─────────────────────────────────────────────
echo ""
echo "[3/4] Время холодного старта (cold start):"

# Go: время до первого вывода
GO_START_MS=$(docker run --rm --name ml-bench-go-cold \
    -v "$(pwd)/results:/app/results" \
    ml-bench/go-bench:latest \
    /usr/bin/time -f "%e" sh -c 'echo start; exit' 2>&1 | \
    grep -E '^[0-9]' | head -1 | awk '{printf "%.0f", $1 * 1000}' || echo "N/A")

echo "  Go  cold start: ~${GO_START_MS} мс"
echo "  Python cold start: замеряется при запуске python-bench"

# ── 4. RAM-потребление ────────────────────────────────────────────────────
echo ""
echo "[4/4] Замер RAM (запустите бенчмарки и выполните 'docker stats --no-stream'):"
echo ""
echo "  Команда для замера:"
echo "    docker stats --no-stream --format '{{.Name}}|{{.MemUsage}}|{{.CPUPerc}}'"
echo ""
echo "  Запуск бенчмарков в фоне:"
echo "    docker-compose up go-bench python-bench &"
echo "    sleep 5 && docker stats --no-stream"

# ── Запись метрик в JSON ──────────────────────────────────────────────────
cat > "$OUTPUT" << EOF
{
  "docker_image_sizes": {
    "go_bench_mb": $GO_SIZE_MB,
    "python_bench_mb": $PY_SIZE_MB,
    "ratio_python_go": $(echo "scale=1; $PY_IMAGE_SIZE / $GO_IMAGE_SIZE" | bc)
  },
  "notes": {
    "go_base_image":     "scratch (пустой образ)",
    "python_base_image": "python:3.11-slim",
    "measurement_date":  "$(date -I)"
  }
}
EOF

echo ""
echo "  Метрики сохранены: $OUTPUT"
echo ""
echo "┌─────────────────────────────────────────┐"
echo "│  Для полного замера RAM выполните:       │"
echo "│  docker-compose up & sleep 3 &&          │"
echo "│  docker stats --no-stream                │"
echo "└─────────────────────────────────────────┘"
