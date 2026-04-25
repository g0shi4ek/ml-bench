# ── Этап 1: сборка ──────────────────────────────────────────────────────────
FROM golang:1.21-alpine AS builder

WORKDIR /build

# Кешируем зависимости отдельно от исходников
COPY go-bench/go.mod go-bench/go.sum ./
RUN go mod download

COPY go-bench/ ./

# Собираем бинарник бенчмарка
# CGO_ENABLED=0 — чистый Go без C-зависимостей → минимальный образ
# -ldflags="-s -w" — убираем символы отладки → уменьшаем размер
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w" \
    -o /bench-runner \
    ./cmd/bench/

# ── Этап 2: минимальный runtime-образ ────────────────────────────────────────
# scratch — пустой образ, только наш бинарник
# Итоговый размер образа: ~10-15 МБ (против ~2-5 ГБ для Python+PyTorch)
FROM scratch

WORKDIR /app
COPY --from=builder /bench-runner /app/bench-runner
COPY go-bench/testdata/ /app/testdata/

# Директория для результатов — монтируется как volume
VOLUME ["/app/results"]

ENTRYPOINT ["/app/bench-runner"]
CMD ["--output", "/app/results/go_results.json", "--n", "10000"]
