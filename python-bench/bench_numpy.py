"""
Бенчмарк матричного умножения NumPy — аналог Gonum-бенчмарка.
Сравнивает: numpy.dot (BLAS), numpy @ (matmul), чистый Python.

Использование:
    python bench_numpy.py
"""

import time
import statistics
import json
import numpy as np
from pathlib import Path


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
                    output_path: str = "results/matmul_comparison.png"):
    try:
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
                    f"{mean:.0f} ns", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_ylabel("Среднее время (ns)", fontsize=12)
        ax.set_title("Умножение матрицы 150×4 на вектор 4×1\nNumPy (BLAS) vs Pure Python", fontsize=12)
        ax.set_ylim(0, max(means) * 1.25)

        speedup = python_result["mean_ns"] / numpy_result["mean_ns"]
        ax.text(0.98, 0.95, f"NumPy быстрее в {speedup:.0f}×",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=12, color="#1565C0",
                bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.8))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"  График сохранён: {output_path}")
    except ImportError:
        print("  matplotlib не установлен.")


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

    speedup = python_r["mean_ns"] / numpy_r["mean_ns"]
    print(f"\n  NumPy быстрее чистого Python в {speedup:.1f}×")
    print("  (аналогично Gonum vs наивные срезы в Go)")

    plot_comparison(numpy_r, python_r)

    results = {"numpy": numpy_r, "pure_python": python_r}
    Path("results").mkdir(exist_ok=True)
    with open("results/python_numpy_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Результаты: results/python_numpy_results.json")


if __name__ == "__main__":
    main()
