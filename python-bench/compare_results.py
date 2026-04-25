import json
import sys
from pathlib import Path


# Определяем корень проекта относительно расположения скрипта
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
# В Docker скрипты лежат в /app/, поэтому parent = / и results = /results (неверно).
# Если parent — корень ФС, используем SCRIPT_DIR/results (= /app/results в Docker).
if PROJECT_ROOT == Path("/"):
    RESULTS_DIR = SCRIPT_DIR / "results"
else:
    RESULTS_DIR = PROJECT_ROOT / "results"


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_comparison_bar(go_ns: float, py_ns: float,
                        title: str, output: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # ── График 1: абсолютные значения ──
        langs  = ["Go\n(golearn KNN)", "Python\n(sklearn KNN)"]
        values = [go_ns, py_ns]
        colors = ["#00ADD8", "#3572A5"]

        bars = ax1.bar(langs, values, color=colors, width=0.45, edgecolor="white")
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(values) * 0.02,
                     f"{val:,.0f} ns", ha="center", va="bottom",
                     fontsize=11, fontweight="bold")

        ax1.set_ylabel("Среднее время инференса (ns/op)", fontsize=11)
        ax1.set_title(f"{title}\nАбсолютное время", fontsize=11)
        ax1.set_ylim(0, max(values) * 1.3)

        winner = "Go" if go_ns < py_ns else "Python"
        ratio  = max(values) / min(values) if min(values) > 0 else float("inf")
        ax1.text(0.5, 0.92,
                 f"{winner} быстрее в {ratio:.1f}×",
                 transform=ax1.transAxes, ha="center",
                 fontsize=12, color="#1B5E20",
                 bbox=dict(boxstyle="round", facecolor="#E8F5E9", alpha=0.9))

        # ── График 2: нормированное сравнение ──
        baseline = max(go_ns, py_ns) if max(go_ns, py_ns) > 0 else 1
        norm_vals = [v / baseline * 100 for v in values]
        bars2 = ax2.barh(langs, norm_vals, color=colors, edgecolor="white")
        for bar, nv, v in zip(bars2, norm_vals, values):
            ax2.text(nv + 1, bar.get_y() + bar.get_height() / 2,
                     f"{nv:.1f}%  ({v:,.0f} ns)",
                     va="center", fontsize=10)

        ax2.set_xlim(0, 130)
        ax2.set_xlabel("% от максимального (меньше — лучше)", fontsize=11)
        ax2.set_title("Нормированное сравнение", fontsize=11)

        fig.suptitle(
            "Сравнение производительности инференса KNN\n"
            "Go (golearn) vs Python (scikit-learn)",
            fontsize=13, fontweight="bold", y=1.02,
        )

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  График сохранён: {output_path}")

    except ImportError:
        print("  matplotlib не установлен — график не построен.")


def plot_autograd_comparison(go_ns: float, py_ns: float,
                            title: str, output: str):
    """Строит график сравнения автоградиента."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # ── График 1: абсолютные значения ──
        langs  = ["Go\n(Gorgonia)", "Python\n(PyTorch)"]
        values = [go_ns, py_ns]
        colors = ["#00ADD8", "#FF6F00"]   # Go и PyTorch цвета

        bars = ax1.bar(langs, values, color=colors, width=0.45, edgecolor="white")
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(values) * 0.02,
                     f"{val:,.0f} ns", ha="center", va="bottom",
                     fontsize=11, fontweight="bold")

        ax1.set_ylabel("Среднее время (ns/op)", fontsize=11)
        ax1.set_title(f"{title}\nАбсолютное время", fontsize=11)
        ax1.set_ylim(0, max(values) * 1.3)

        winner = "Go" if go_ns < py_ns else "Python"
        ratio  = max(values) / min(values) if min(values) > 0 else float("inf")
        ax1.text(0.5, 0.92,
                 f"{winner} быстрее в {ratio:.1f}×",
                 transform=ax1.transAxes, ha="center",
                 fontsize=12, color="#1B5E20",
                 bbox=dict(boxstyle="round", facecolor="#E8F5E9", alpha=0.9))

        # ── График 2: нормированное сравнение ──
        baseline = max(go_ns, py_ns) if max(go_ns, py_ns) > 0 else 1
        norm_vals = [v / baseline * 100 for v in values]
        bars2 = ax2.barh(langs, norm_vals, color=colors, edgecolor="white")
        for bar, nv, v in zip(bars2, norm_vals, values):
            ax2.text(nv + 1, bar.get_y() + bar.get_height() / 2,
                     f"{nv:.1f}%  ({v:,.0f} ns)",
                     va="center", fontsize=10)

        ax2.set_xlim(0, 130)
        ax2.set_xlabel("% от максимального (меньше — лучше)", fontsize=11)
        ax2.set_title("Нормированное сравнение", fontsize=11)

        fig.suptitle(
            "Сравнение производительности автоградиента\n"
            "Go (Gorgonia) vs Python (PyTorch)",
            fontsize=13, fontweight="bold", y=1.02,
        )

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  График сохранён: {output_path}")

    except ImportError:
        print("  matplotlib не установлен — график не построен.")


def plot_matmul_comparison(go_gonum: float, go_naive: float,
                          py_numpy: float, py_pure: float,
                          title: str, output: str):
    """Строит график сравнения матричного умножения."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # ── График 1: Go (Gonum vs Naive) ──
        go_labels = ["Gonum\n(BLAS)", "Naive\n(Slices)"]
        go_values = [go_gonum, go_naive]
        go_colors = ["#00ADD8", "#FF6F00"]

        bars1 = ax1.bar(go_labels, go_values, color=go_colors, width=0.4, edgecolor="white")
        for bar, val in zip(bars1, go_values):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(go_values) * 0.02,
                     f"{val:,.0f} ns", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")

        ax1.set_ylabel("Среднее время (ns/op)", fontsize=11)
        ax1.set_title("Go: Gonum vs Naive", fontsize=11)
        ax1.set_ylim(0, max(go_values) * 1.3)

        if go_naive > 0 and go_gonum > 0:
            if go_gonum < go_naive:
                speedup = go_naive / go_gonum
                winner = "Gonum"
            else:
                speedup = go_gonum / go_naive
                winner = "Naive"
            ax1.text(0.5, 0.92,
                     f"{winner} быстрее в {speedup:.1f}×",
                     transform=ax1.transAxes, ha="center",
                     fontsize=11, color="#1B5E20",
                     bbox=dict(boxstyle="round", facecolor="#E8F5E9", alpha=0.9))

        # ── График 2: Python (NumPy vs Pure) ──
        py_labels = ["NumPy\n(BLAS)", "Pure Python\n(Loops)"]
        py_values = [py_numpy, py_pure]
        py_colors = ["#3572A5", "#FFD43B"]

        bars2 = ax2.bar(py_labels, py_values, color=py_colors, width=0.4, edgecolor="white")
        for bar, val in zip(bars2, py_values):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(py_values) * 0.02,
                     f"{val:,.0f} ns", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")

        ax2.set_ylabel("Среднее время (ns/op)", fontsize=11)
        ax2.set_title("Python: NumPy vs Pure Python", fontsize=11)
        ax2.set_ylim(0, max(py_values) * 1.3)

        if py_pure > 0 and py_numpy > 0:
            speedup = py_pure / py_numpy
            ax2.text(0.5, 0.92,
                     f"NumPy быстрее в {speedup:.1f}×",
                     transform=ax2.transAxes, ha="center",
                     fontsize=11, color="#1B5E20",
                     bbox=dict(boxstyle="round", facecolor="#E8F5E9", alpha=0.9))

        fig.suptitle(
            "Сравнение матричного умножения (150×4 @ 4×1)\n"
            "Оптимизированные библиотеки vs наивные реализации",
            fontsize=13, fontweight="bold", y=1.02,
        )

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  График сохранён: {output_path}")

    except ImportError:
        print("  matplotlib не установлен — график не построен.")


def main():
    print("=" * 55)
    print("  Сравнение результатов Go vs Python")
    print("=" * 55)

    go_path = RESULTS_DIR / "go_results.json"
    py_path = RESULTS_DIR / "python_results.json"

    if not go_path.exists() or not py_path.exists():
        missing = []
        if not go_path.exists():
            missing.append(f"    {go_path}")
        if not py_path.exists():
            missing.append(f"    {py_path}")
        print("\n  Не найдены файлы результатов:")
        for m in missing:
            print(m)
        print("\n  Сначала запустите бенчмарки:")
        print("    cd go-bench && go run ./cmd/bench/ --output ../results/go_results.json")
        print("    cd python-bench && python bench_knn.py")
        sys.exit(1)

    go_data = load_json(go_path)
    py_data = load_json(py_path)

    # ── Сравнение KNN ──
    print("\n" + "=" * 55)
    print("  1. Сравнение KNN инференса")
    print("=" * 55)

    go_knn = next((r for r in go_data["results"] if "KNN" in r["name"]), None)
    py_knn = next((r for r in py_data["results"] if "KNN" in r["name"]), None)

    if go_knn is not None and py_knn is not None:
        go_ns = go_knn["ns_per_op"]
        py_ns = py_knn["mean_ns"]

        print(f"\n  Go  KNN (single): {go_ns:>10,.1f} ns/op")
        print(f"  Py  KNN (single): {py_ns:>10,.1f} ns/op")

        if min(go_ns, py_ns) > 0:
            winner = "Go" if go_ns < py_ns else "Python"
            ratio = max(go_ns, py_ns) / min(go_ns, py_ns)
            print(f"\n  Победитель (латентность): {winner} (в {ratio:.1f}× быстрее)")

        output_png = str(RESULTS_DIR / "comparison_knn.png")
        plot_comparison_bar(go_ns, py_ns,
                            title="KNN Inference, single sample, n=10000",
                            output=output_png)
    else:
        print("\n  ⚠ Не найдены результаты KNN")

    # ── Сравнение матричного умножения ──
    print("\n" + "=" * 55)
    print("  2. Сравнение матричного умножения")
    print("=" * 55)

    go_gonum = next((r for r in go_data["results"] if "Gonum" in r["name"] and "MatMul" in r["name"]), None)
    go_naive = next((r for r in go_data["results"] if "Naive" in r["name"] and "Mul" in r["name"]), None)
    py_numpy = next((r for r in py_data["results"] if "NumPy" in r["name"] and "MatMul" in r["name"]), None)
    py_pure = next((r for r in py_data["results"] if "PurePython" in r["name"] and "MatMul" in r["name"]), None)

    if go_gonum is not None and go_naive is not None and py_numpy is not None and py_pure is not None:
        print(f"\n  Go  Gonum: {go_gonum['ns_per_op']:>10,.1f} ns/op")
        print(f"  Go  Naive: {go_naive['ns_per_op']:>10,.1f} ns/op")
        print(f"  Py  NumPy: {py_numpy['mean_ns']:>10,.1f} ns/op")
        print(f"  Py  Pure:  {py_pure['mean_ns']:>10,.1f} ns/op")

        # Сравнение Gonum vs NumPy
        if go_gonum["ns_per_op"] > 0 and py_numpy["mean_ns"] > 0:
            winner = "Go" if go_gonum["ns_per_op"] < py_numpy["mean_ns"] else "Python"
            ratio = max(go_gonum["ns_per_op"], py_numpy["mean_ns"]) / min(go_gonum["ns_per_op"], py_numpy["mean_ns"])
            print(f"\n  Gonum vs NumPy: {winner} (в {ratio:.1f}× быстрее)")

        output_png = str(RESULTS_DIR / "comparison_matmul.png")
        plot_matmul_comparison(go_gonum["ns_per_op"], go_naive["ns_per_op"],
                              py_numpy["mean_ns"], py_pure["mean_ns"],
                              title="Matrix Multiplication 150×4 @ 4×1",
                              output=output_png)
    else:
        print("\n  ⚠ Не найдены результаты матричного умножения")
        if go_gonum is None:
            print("    - Go Gonum: не найден")
        if go_naive is None:
            print("    - Go Naive: не найден")
        if py_numpy is None:
            print("    - Python NumPy: не найден")
        if py_pure is None:
            print("    - Python Pure: не найден")

    # ── Сравнение автоградиента ──
    print("\n" + "=" * 55)
    print("  3. Сравнение автоградиента")
    print("=" * 55)

    go_autograd = next((r for r in go_data["results"] if "Gorgonia" in r["name"] and "Autograd" in r["name"]), None)
    py_autograd = next((r for r in py_data["results"] if "Autograd" in r["name"]), None)

    if go_autograd is not None and py_autograd is not None:
        go_ns = go_autograd["ns_per_op"]
        py_ns = py_autograd["mean_ns"]

        print(f"\n  Go  Gorgonia: {go_ns:>10,.1f} ns/op")
        print(f"  Py  PyTorch:  {py_ns:>10,.1f} ns/op")

        if min(go_ns, py_ns) > 0:
            winner = "Go" if go_ns < py_ns else "Python"
            ratio = max(go_ns, py_ns) / min(go_ns, py_ns)
            print(f"\n  Победитель (автоградиент): {winner} (в {ratio:.1f}× быстрее)")

        output_png = str(RESULTS_DIR / "comparison_autograd.png")
        plot_autograd_comparison(go_ns, py_ns,
                                title="Autograd: z = (x+y)*(x-y), n=10000",
                                output=output_png)
    else:
        print("\n  ⚠ Не найдены результаты автоградиента")
        if go_autograd is None:
            print("    - Go: запустите go-bench/cmd/bench/main.go")
        if py_autograd is None:
            print("    - Python: запустите python-bench/bench_autograd.py")

    # ── Итоговая сводка ──
    print("\n" + "=" * 55)
    print("  Итоговая сводка")
    print("=" * 55)

    print("\n  Все результаты:")
    print(f"  Go бенчмарков: {len(go_data['results'])}")
    print(f"  Python бенчмарков: {len(py_data['results'])}")

    print("\n  Графики:")
    print("  - comparison_knn.png")
    print("  - comparison_matmul.png")
    print("  - comparison_autograd.png")

    print("\n  Готово! Проверьте папку results/")


if __name__ == "__main__":
    main()
