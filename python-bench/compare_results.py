"""
Скрипт сравнения результатов Go и Python бенчмарков.
Загружает JSON-файлы из results/ и строит итоговые графики.

Использование:
    python compare_results.py
"""

import json
import sys
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_comparison_bar(go_ns: float, py_ns: float,
                        title: str, output: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # ── График 1: абсолютные значения ──
        langs  = ["Go\n(golearn KNN)", "Python\n(sklearn KNN)"]
        values = [go_ns, py_ns]
        colors = ["#00ADD8", "#3572A5"]   # официальные цвета Go и Python

        bars = ax1.bar(langs, values, color=colors, width=0.45, edgecolor="white")
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(values) * 0.02,
                     f"{val:,.0f} ns", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax1.set_ylabel("Среднее время инференса (ns/op)", fontsize=11)
        ax1.set_title(f"{title}\nАбсолютное время", fontsize=11)
        ax1.set_ylim(0, max(values) * 1.3)

        winner = "Go" if go_ns < py_ns else "Python"
        ratio  = max(values) / min(values)
        ax1.text(0.5, 0.92,
                 f"{winner} быстрее в {ratio:.1f}×",
                 transform=ax1.transAxes, ha="center",
                 fontsize=12, color="#1B5E20",
                 bbox=dict(boxstyle="round", facecolor="#E8F5E9", alpha=0.9))

        # ── График 2: нормированное сравнение ──
        baseline = max(go_ns, py_ns)
        norm_vals = [v / baseline * 100 for v in values]
        bars2 = ax2.barh(langs, norm_vals, color=colors, edgecolor="white")
        for bar, nv, v in zip(bars2, norm_vals, values):
            ax2.text(nv + 1, bar.get_y() + bar.get_height() / 2,
                     f"{nv:.1f}%  ({v:,.0f} ns)",
                     va="center", fontsize=10)

        ax2.set_xlim(0, 130)
        ax2.set_xlabel("% от максимального (меньше — лучше)", fontsize=11)
        ax2.set_title("Нормированное сравнение", fontsize=11)

        fig.suptitle("Сравнение производительности инференса KNN\nGo (golearn) vs Python (scikit-learn)",
                     fontsize=13, fontweight="bold", y=1.02)

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output, dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt_close
        plt_close.close(fig)
        print(f"  График сохранён: {output}")

    except ImportError:
        print("  matplotlib не установлен — график не построен.")


def main():
    print("=" * 55)
    print("  Сравнение результатов Go vs Python")
    print("=" * 55)

    go_path  = Path("results/go_results.json")
    py_path  = Path("results/python_results.json")

    if not go_path.exists() or not py_path.exists():
        print("\n  Сначала запустите бенчмарки:")
        print("    go run ./go-bench/cmd/bench/")
        print("    python python-bench/bench_knn.py")
        sys.exit(1)

    go_data  = load_json(go_path)
    py_data  = load_json(py_path)

    # Извлечь результат KNN single из обоих отчётов
    go_knn  = next(r for r in go_data["results"]  if "KNN" in r["name"])
    py_knn  = next(r for r in py_data["results"]  if "KNN" in r["name"])

    go_ns = go_knn["ns_per_op"]
    py_ns = py_knn["mean_ns"]

    print(f"\n  Go  KNN (single): {go_ns:>10,.1f} ns/op")
    print(f"  Py  KNN (single): {py_ns:>10,.1f} ns/op")
    winner = "Go" if go_ns < py_ns else "Python"
    ratio  = max(go_ns, py_ns) / min(go_ns, py_ns)
    print(f"\n  Победитель (латентность): {winner} (в {ratio:.1f}× быстрее)")

    plot_comparison_bar(go_ns, py_ns,
                        title="KNN Inference, single sample, n=10000",
                        output="results/comparison_knn.png")

    print("\n  Готово! Проверьте папку results/")


if __name__ == "__main__":
    main()
