"""
Бенчмарк инференса KNN-классификатора на Python (scikit-learn).
Функционально эквивалентен Go-реализации в go-bench/internal/knn/.

Использование:
    cd python-bench
    python bench_knn.py

Или из корня проекта:
    python python-bench/bench_knn.py

Требования:
    pip install scikit-learn numpy matplotlib

Интерпретация результатов:
    - mean_ns:   среднее время одного предсказания в наносекундах.
                 Сравнивается с "ns/op" из Go-бенчмарка.
                 Чем меньше — тем ниже латентность при обслуживании запросов.

    - std_ns:    стандартное отклонение в нс. Высокое значение (>10% от mean)
                 указывает на нестабильность производительности — нежелательно
                 для SLA-чувствительных сервисов.

    - p50/p95/p99: перцентили латентности (медиана, 95-й и 99-й перцентиль).
                 В продакшн системах SLA обычно задаётся как p99 < X мс.

    - Python GIL: в отличие от Go-горутин, Python-потоки ограничены GIL.
                 Для CPU-bound инференса реальный параллелизм достигается
                 через multiprocessing, а не threading.
"""

import time
import statistics
import json
import platform
import sys
from pathlib import Path

from typing import Optional

import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Определяем корень проекта относительно расположения скрипта
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
# В Docker скрипты лежат в /app/, поэтому parent = / и results = /results (неверно).
# Если parent — корень ФС, используем SCRIPT_DIR/results (= /app/results в Docker).
if PROJECT_ROOT == Path("/"):
    RESULTS_DIR = SCRIPT_DIR / "results"
else:
    RESULTS_DIR = PROJECT_ROOT / "results"


# ──────────────────────────────────────────────
# 1. Подготовка данных и модели
# ──────────────────────────────────────────────

def prepare_model():
    """Загружает Iris, обучает KNN(k=3), возвращает модель и тестовые данные."""
    iris = load_iris()
    X, y = iris.data, iris.target

    # Разбивка 70/30 — аналогично Go-версии
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    # n_neighbors=3 соответствует Go-реализации knn.NewKnnClassifier(..., 3)
    clf = KNeighborsClassifier(n_neighbors=3, metric="euclidean", algorithm="auto")
    clf.fit(X_train, y_train)

    # Проверка качества
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy на тестовой выборке: {acc:.4f}")

    # Один тестовый сэмпл (первая строка X_test) — аналог Go-версии
    single_sample = X_test[0:1]  # shape (1, 4)
    return clf, single_sample, X_test


# ──────────────────────────────────────────────
# 2. Замер инференса одного сэмпла
# ──────────────────────────────────────────────

def benchmark_single_inference(clf, sample, n_iterations: int = 10_000) -> dict:
    """
    Замеряет время n_iterations вызовов clf.predict(sample).

    Использует time.perf_counter — наиболее точный монотонный таймер
    в Python (разрешение ~100 нс на Linux, ~15 нс на Windows с HPET).

    Аналог testing.B в Go: измеряет только вызов predict,
    исключая накладные расходы на инициализацию.
    """
    # Прогрев (warm-up): JIT scikit-learn / NumPy оптимизирует первые вызовы
    for _ in range(200):
        clf.predict(sample)

    timings_ns = []

    for _ in range(n_iterations):
        t_start = time.perf_counter()
        clf.predict(sample)
        t_end = time.perf_counter()
        # Переводим в наносекунды (1 сек = 1e9 нс)
        timings_ns.append((t_end - t_start) * 1e9)

    mean_ns  = statistics.mean(timings_ns)
    stdev_ns = statistics.stdev(timings_ns)
    p50_ns   = float(np.percentile(timings_ns, 50))
    p95_ns   = float(np.percentile(timings_ns, 95))
    p99_ns   = float(np.percentile(timings_ns, 99))
    min_ns   = min(timings_ns)
    max_ns   = max(timings_ns)

    return {
        "name":       "KNNInference_single",
        "iterations": n_iterations,
        "mean_ns":    round(mean_ns, 2),
        "stdev_ns":   round(stdev_ns, 2),
        "p50_ns":     round(p50_ns, 2),
        "p95_ns":     round(p95_ns, 2),
        "p99_ns":     round(p99_ns, 2),
        "min_ns":     round(min_ns, 2),
        "max_ns":     round(max_ns, 2),
        "mean_us":    round(mean_ns / 1e3, 3),
        "mean_ms":    round(mean_ns / 1e6, 6),
        "timings_ns": timings_ns,  # сырые данные для построения гистограммы
    }


# ──────────────────────────────────────────────
# 3. Замер инференса батча
# ──────────────────────────────────────────────

def benchmark_batch_inference(clf, X_test: np.ndarray, n_iterations: int = 1000) -> dict:
    """Замеряет инференс на всей тестовой выборке (батч)."""
    # Прогрев
    for _ in range(50):
        clf.predict(X_test)

    timings_ns = []
    for _ in range(n_iterations):
        t_start = time.perf_counter()
        clf.predict(X_test)
        t_end = time.perf_counter()
        timings_ns.append((t_end - t_start) * 1e9)

    mean_ns = statistics.mean(timings_ns)
    return {
        "name":          "KNNInference_batch",
        "batch_size":    len(X_test),
        "iterations":    n_iterations,
        "mean_ns":       round(mean_ns, 2),
        "ns_per_sample": round(mean_ns / len(X_test), 2),
    }


# ──────────────────────────────────────────────
# 4. Построение гистограммы распределения латентности
# ──────────────────────────────────────────────

def plot_latency_histogram(timings_ns: list,
                           output_path: Optional[str] = None):
    """Строит гистограмму распределения времени инференса."""
    if output_path is None:
        output_path = str(RESULTS_DIR / "latency_hist.png")

    try:
        import matplotlib
        matplotlib.use("Agg")  # неинтерактивный бэкенд — работает без GUI
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        timings_us = [t / 1e3 for t in timings_ns]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(timings_us, bins=80, color="#4C72B0", edgecolor="white", alpha=0.85)

        mean_us = statistics.mean(timings_us)
        p99_us  = float(np.percentile(timings_us, 99))

        ax.axvline(mean_us, color="#DD4444", linestyle="--", linewidth=1.5,
                   label=f"mean = {mean_us:.1f} µs")
        ax.axvline(p99_us, color="#FF8800", linestyle=":",  linewidth=1.5,
                   label=f"p99  = {p99_us:.1f} µs")

        ax.set_xlabel("Латентность инференса (µs)", fontsize=12)
        ax.set_ylabel("Число измерений", fontsize=12)
        ax.set_title(
            "Распределение латентности KNN инференса\n"
            "(Python / scikit-learn, n=10 000)",
            fontsize=13,
        )
        ax.legend()
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(str(out), dpi=150)
        plt.close(fig)
        print(f"  Гистограмма сохранена: {out}")
    except ImportError:
        print("  matplotlib не установлен — гистограмма не построена.")


# ──────────────────────────────────────────────
# 5. Запись результатов в JSON
# ──────────────────────────────────────────────

def save_results(single_result: dict, batch_result: dict,
                 output_path: Optional[str] = None):
    """Сохраняет результаты в JSON для последующего сравнения с Go."""
    if output_path is None:
        output_path = str(RESULTS_DIR / "python_results.json")

    # Убираем сырые timings из JSON (слишком большой массив)
    single_clean = {k: v for k, v in single_result.items() if k != "timings_ns"}

    report = {
        "system": {
            "python_version": sys.version,
            "platform":       platform.platform(),
            "processor":      platform.processor(),
        },
        "results": [single_clean, batch_result],
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Результаты сохранены: {out}")


# ──────────────────────────────────────────────
# 6. Точка входа
# ──────────────────────────────────────────────

def main():
    N_SINGLE = 10_000  # число итераций — совпадает с Go-бенчмарком
    N_BATCH  = 1_000

    print("=" * 55)
    print("  Python KNN Benchmark (scikit-learn)")
    print("=" * 55)

    print(f"\n[1/4] Подготовка модели...")
    clf, single_sample, X_test = prepare_model()

    print(f"\n[2/4] Инференс одного сэмпла (n={N_SINGLE:,})...")
    single_result = benchmark_single_inference(clf, single_sample, N_SINGLE)

    print(f"\n[3/4] Инференс батча (size={len(X_test)}, n={N_BATCH:,})...")
    batch_result = benchmark_batch_inference(clf, X_test, N_BATCH)

    print("\n[4/4] Сохранение результатов...")
    plot_latency_histogram(single_result["timings_ns"])
    save_results(single_result, batch_result)

    # ── Итоговый вывод ──
    print("\n" + "=" * 55)
    print("  РЕЗУЛЬТАТЫ")
    print("=" * 55)
    print(f"  Одиночный инференс:")
    print(f"    mean   = {single_result['mean_ns']:>10.1f} ns/op")
    print(f"    stdev  = {single_result['stdev_ns']:>10.1f} ns")
    print(f"    p50    = {single_result['p50_ns']:>10.1f} ns")
    print(f"    p95    = {single_result['p95_ns']:>10.1f} ns")
    print(f"    p99    = {single_result['p99_ns']:>10.1f} ns")
    print(f"    mean   = {single_result['mean_us']:>10.3f} µs/op")
    print(f"\n  Примечание: для сравнения с Go запустите:")
    print(f"    cd go-bench && go test -bench=BenchmarkKNNInference -benchmem ./internal/knn/")
    print("=" * 55)


if __name__ == "__main__":
    main()
