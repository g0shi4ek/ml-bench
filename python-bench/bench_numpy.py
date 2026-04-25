"""
Бенчмарк матричного умножения NumPy — аналог Gonum-бенчмарка.
Сравнивает: numpy @ (matmul с BLAS DGEMM) vs чистый Python (циклы).

Использование:
    cd python-bench
    python bench_numpy.py

Или из корня проекта:
    python python-bench/bench_numpy.py
"""

import time
import statistics
import json
import platform
from pathlib import Path

from typing import Optional

import numpy as np


# Определяем корень проекта относительно расположения скрипта
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
# В Docker скрипты лежат в /app/, поэтому parent = / и results = /results (неверно).
# Если parent — корень ФС, используем SCRIPT_DIR/results (= /app/results в Docker).
if PROJECT_ROOT == Path("/"):
    RESULTS_DIR = SCRIPT_DIR / "results"
else:
    RESULTS_DIR = PROJECT_ROOT / "results"


def benchmark_numpy_matmul(n: int = 10_000) -> dict:
    """NumPy @ (matmul) — использует BLAS DGEMM под капотом."""
    X = np.random.rand(150, 4).astype(np.float64)
    W = np.random.rand(4, 1).astype(np.float64)

    # Прогрев
    for _ in range(200):
        _ = X @ W

    timings = []
    for _ in range(n):
        t0 = time.perf_counter()
        result = X @ W
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1e9)
        _ = result[0, 0]

    return {
        "name":    "NumPy_matmul_150x4",
        "mean_ns": round(statistics.mean(timings), 2),
        "std_ns":  round(statistics.stdev(timings), 2),
        "p99_ns":  round(float(np.percentile(timings, 99)), 2),
    }


def benchmark_pure_python(n: int = 10_000) -> dict:
    """Чистый Python — аналог наивной реализации на Go-срезах."""
    X = [[float(i * j + 0.1) for j in range(4)] for i in range(150)]
    W = [0.1 * i for i in range(4)]

    # Прогрев
    for _ in range(50):
        result = [sum(X[r][c] * W[c] for c in range(4)) for r in range(150)]
        _ = result[0]

    timings = []
    for _ in range(n):
        t0 = time.perf_counter()
        result = [sum(X[r][c] * W[c] for c in range(4)) for r in range(150)]
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1e9)
        _ = result[0]

    return {
        "name":    "PurePython_matmul_150x4",
        "mean_ns": round(statistics.mean(timings), 2),
        "std_ns":  round(statistics.stdev(timings), 2),
        "p99_ns":  round(float(np.percentile(timings, 99)), 2),
    }


def plot_comparison(numpy_result: dict, python_result: dict,
                    output_path: Optional[str] = None):
    if output_path is None:
        output_path = str(RESULTS_DIR / "matmul_comparison.png")

    try:
        import matplotlib
        matplotlib.use("Agg")  # неинтерактивный бэкенд — работает без GUI
        import matplotlib.pyplot as plt

        labels = ["NumPy\n(BLAS)", "Pure Python\n(loops)"]
        means  = [numpy_result["mean_ns"], python_result["mean_ns"]]
        stds   = [numpy_result["std_ns"],  python_result["std_ns"]]
        colors = ["#2196F3", "#FF5722"]

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(labels, means, yerr=stds, color=colors,
                      capsize=6, edgecolor="white", width=0.4)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(means) * 0.02,
                    f"{mean:.0f} ns", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

        ax.set_ylabel("Среднее время (ns)", fontsize=12)
        ax.set_title(
            "Умножение матрицы 150×4 на вектор 4×1\n"
            "NumPy (BLAS) vs Pure Python",
            fontsize=12,
        )
        ax.set_ylim(0, max(means) * 1.25)

        if numpy_result["mean_ns"] > 0:
            speedup = python_result["mean_ns"] / numpy_result["mean_ns"]
            ax.text(0.98, 0.95, f"NumPy быстрее в {speedup:.0f}×",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=12, color="#1565C0",
                    bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.8))

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(str(out), dpi=150)
        plt.close(fig)
        print(f"  График сохранён: {out}")
    except ImportError:
        print("  matplotlib не установлен — график не построен.")


def save_results_to_json(numpy_result: dict, python_result: dict) -> None:
    """Сохраняет результаты в общий python_results.json."""
    # Определяем директорию результатов
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    if project_root == Path("/"):
        results_dir = script_dir / "results"
    else:
        results_dir = project_root / "results"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "python_results.json"
    
    # Загружаем существующие результаты или создаём новые
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {
            "system": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "numpy_version": np.__version__,
            },
            "results": []
        }
    
    # Добавляем результаты NumPy
    numpy_entry = {
        "name": "NumPyMatMul_150x4",
        "mean_ns": numpy_result["mean_ns"],
        "std_ns": numpy_result["std_ns"],
        "p99_ns": numpy_result["p99_ns"],
    }
    
    # Добавляем результаты Pure Python
    python_entry = {
        "name": "PurePythonMatMul_150x4",
        "mean_ns": python_result["mean_ns"],
        "std_ns": python_result["std_ns"],
        "p99_ns": python_result["p99_ns"],
    }
    
    # Обновляем или добавляем результаты
    for entry in [numpy_entry, python_entry]:
        existing_idx = None
        for i, r in enumerate(data["results"]):
            if r.get("name") == entry["name"]:
                existing_idx = i
                break
        
        if existing_idx is not None:
            data["results"][existing_idx] = entry
        else:
            data["results"].append(entry)
    
    # Сохраняем
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"\n  Результаты сохранены в: {output_path}")


def main():
    print("=" * 50)
    print("  NumPy vs Pure Python Matrix Benchmark")
    print("=" * 50)

    print("\n[1/2] NumPy matmul (n=10000)...")
    numpy_r = benchmark_numpy_matmul(10_000)
    print(f"  NumPy : {numpy_r['mean_ns']:.1f} ns/op")

    print("\n[2/2] Pure Python (n=10000)...")
    python_r = benchmark_pure_python(10_000)
    print(f"  Python: {python_r['mean_ns']:.1f} ns/op")

    if numpy_r["mean_ns"] > 0:
        speedup = python_r["mean_ns"] / numpy_r["mean_ns"]
        print(f"\n  NumPy быстрее чистого Python в {speedup:.1f}×")
        print("  (аналогично Gonum vs наивные срезы в Go)")

    plot_comparison(numpy_r, python_r)
    
    # Сохраняем в общий python_results.json
    save_results_to_json(numpy_r, python_r)


if __name__ == "__main__":
    main()
