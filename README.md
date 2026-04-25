# ml-bench: Go vs Python ML Library Benchmark

**НИРС: «Анализ библиотек для машинного обучения на Golang и Python»**  
Направление подготовки: 09.03.04 Программная инженерия, 3 курс

---

## Структура проекта

```
ml-bench/
├── go-bench/                      # Go-реализации
│   ├── cmd/bench/main.go          # CLI-запуск всех бенчмарков
│   ├── internal/
│   │   ├── knn/                   # Промт 3.1: KNN инференс (golearn)
│   │   ├── gonum_bench/           # Промт 3.1-ДОП: Gonum vs срезы
│   │   └── gorgonia_bench/        # Промт 3.2-ДОП: Gorgonia autograd
│   ├── testdata/iris.csv
│   └── go.mod
├── python-bench/
│   ├── bench_knn.py               # Промт 3.2: KNN (scikit-learn)
│   ├── bench_numpy.py             # NumPy vs pure Python
│   ├── compare_results.py         # Визуализация сравнения
│   └── requirements.txt
├── docker/
│   ├── Dockerfile.go
│   └── Dockerfile.python
├── docker-compose.yml
├── scripts/
│   └── measure_docker.sh          # Замер размеров образов и RAM
└── results/                       # JSON + PNG результатов
```

---

## Быстрый старт

### Вариант 1: Docker (рекомендуется для воспроизводимости)

```bash
# Сборка и запуск всех бенчмарков
docker-compose up --build

# Просмотр результатов
ls results/

# Размеры образов (Промт 3.3 — таблица методологии)
docker images | grep ml-bench
```

### Вариант 2: Локальный запуск

**Go-бенчмарки:**
```bash
cd go-bench

# KNN инференс (golearn) — стандартный testing.B
go test -bench=BenchmarkKNNInference -benchmem -benchtime=10s ./internal/knn/

# Gonum vs naive srices
go test -bench=. -benchmem -count=5 ./internal/gonum_bench/

# Gorgonia autograd (тест + бенчмарк)
go test -v -run TestGorgoniaAutograd ./internal/gorgonia_bench/
go test -bench=BenchmarkGorgoniaInference -benchmem ./internal/gorgonia_bench/

# CLI-запуск с записью в JSON
go run ./cmd/bench/ --output ../results/go_results.json
```

**Python-бенчмарки:**
```bash
cd python-bench
pip install -r requirements.txt

python bench_knn.py          # KNN (Промт 3.2)
python bench_numpy.py        # NumPy vs pure Python

# Сравнение (запускать после обоих бенчмарков)
python compare_results.py
```

---

## Измеряемые метрики

| Метрика | Go инструмент | Python инструмент |
|---|---|---|
| Время инференса (ns/op) | `testing.B` | `time.perf_counter` |
| Аллокации памяти (allocs/op) | `testing.B -benchmem` | `tracemalloc` |
| Размер Docker-образа (МБ) | `docker image inspect` | `docker image inspect` |
| RAM во время выполнения | `docker stats` | `docker stats` |
| Время холодного старта | `time docker run` | `time docker run` |

---

## Ожидаемые результаты

| Критерий | Победитель | Причина |
|---|---|---|
| Латентность инференса (единичный запрос) | ≈ паритет | KNN линеен, Go GC накладки малы |
| Размер Docker-образа | **Go** (~12 МБ vs ~600 МБ) | scratch vs python:3.11-slim |
| Аллокации памяти | **Go** | явное управление; меньше объектов |
| Матричное умножение (large) | **Python/NumPy** | BLAS/LAPACK SIMD-оптимизация |
| Кросс-компиляция | **Go** | один бинарник без зависимостей |
| Разнообразие алгоритмов | **Python** | зрелая экосистема sklearn |

---

## Зависимости

**Go:** `github.com/sjwhitworth/golearn`, `gorgonia.org/gorgonia`, `gonum.org/v1/gonum`  
**Python:** `scikit-learn`, `numpy`, `matplotlib`  
**Инфраструктура:** Docker 24+, docker-compose v2
