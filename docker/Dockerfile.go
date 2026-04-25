FROM golang:1.24-alpine AS builder

RUN apk add --no-cache gcc musl-dev

WORKDIR /build

COPY go-bench/go.mod go-bench/go.sum ./
RUN go mod download

COPY go-bench/ ./

# CGO_ENABLED=1 — golearn требует CGO для оптимизированного KNN
RUN CGO_ENABLED=1 GOOS=linux go build \
    -ldflags="-s -w" \
    -o /bench-runner \
    ./cmd/bench/

FROM alpine:3.19

ENV ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.24

WORKDIR /app
COPY --from=builder /bench-runner /app/bench-runner
COPY go-bench/testdata/ /app/testdata/

VOLUME ["/app/results"]

ENTRYPOINT ["/app/bench-runner"]
CMD ["--output", "/app/results/go_results.json", "--n", "10000"]
