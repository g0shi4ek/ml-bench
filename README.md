# ml-bench: Go vs Python ML Library Benchmark

**НИРС: «Анализ библиотек для машинного обучения на Golang и Python»**

### 1. Отдельные бенчмарки Go (в папке internal)

Запуск отдельных бенчмарков для детального анализа каждой библиотеки:

```bash
cd go-bench

export GOMODCACHE="$(pwd)/.gomodcache"
export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26

# KNN инференс (golearn)
go test -bench=. -benchmem -benchtime=10s ./internal/knn/

# Gonum vs naive slices (линейная алгебра)
go test -bench=. -benchmem -count=5 ./internal/gonum_bench/

# Gorgonia autograd (автоматическое дифференцирование)
go test -bench=. -benchmem  -count=5 ./internal/gorgonia_bench/

cd ..
```

### 2. Полный локальный запуск

Запуск всех бенчмарков с сохранением результатов в JSON и генерацией графиков сравнения:

#### 2.1. Go-бенчмарки

```bash
cd go-bench

# Настройка окружения (однократно)
export GOMODCACHE="$(pwd)/.gomodcache"
export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26

# Запуск всех бенчмарков с сохранением в JSON
go run ./cmd/bench/ --output ../results/go_results.json --n 10000

cd ..
```

#### 2.2. Python-бенчмарки

```bash
cd python-bench

# Создание виртуального окружения (однократно)
bash setup_venv.sh

# Активация окружения
source .venv/bin/activate

# Запуск всех бенчмарков
python bench_knn.py          # KNN (scikit-learn)
python bench_numpy.py        # NumPy vs pure Python
python bench_autograd.py     # PyTorch autograd

# Деактивация окружения
deactivate

cd ..
```

#### 2.3. Сравнение результатов

```bash
cd python-bench
source .venv/bin/activate
python compare_results.py    # Генерация графиков сравнения
deactivate
cd ..
```

**Результаты:** Все результаты сохраняются в папку `results/`:
- `go_results.json` — результаты Go-бенчмарков
- `python_results.json` — результаты Python-бенчмарков + графики
- `comparison_*.png` — графики сравнения

### 3. Запуск в Docker

Полная изоляция в Docker-контейнерах с автоматическим сравнением результатов:

```bash
# Сборка и запуск всех бенчмарков в контейнерах
docker compose up --build go-bench python-bench compare

```