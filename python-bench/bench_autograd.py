"""
bench_autograd.py — Аналог Gorgonia-теста автоградиента на Python (PyTorch).

Промт 3.2-ДОП: Построение вычислительного графа z = (x+y)*(x-y)
с автоматическим дифференцированием (autograd).

Аналитические производные:
    z = x² - y²
    dz/dx = 2x
    dz/dy = -2y

PyTorch vs Gorgonia:
    - PyTorch: eager mode (define-by-run), автоград «из коробки»,
      граф строится неявно при выполнении операций.
    - Gorgonia: define-and-run (как TF 1.x), граф строится явно
      через NewGraph/NewScalar/Add/Sub/Mul, затем выполняется
      через TapeMachine.

Запуск:
    cd python-bench
    source .venv/bin/activate
    python bench_autograd.py
    python bench_autograd.py -n 50000
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple, List

import torch
import platform


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PyTorch autograd: z = (x+y)*(x-y)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_autograd(x_val: float, y_val: float) -> Tuple[float, float, float]:
    """
    Вычисляет z = (x+y)*(x-y) и градиенты dz/dx, dz/dy через PyTorch autograd.

    Аналог Go-кода (Gorgonia):
        g := gorgonia.NewGraph()
        x := gorgonia.NewScalar(g, Float64, WithValue(x_val))
        y := gorgonia.NewScalar(g, Float64, WithValue(y_val))
        sum := gorgonia.Must(gorgonia.Add(x, y))
        diff := gorgonia.Must(gorgonia.Sub(x, y))
        z := gorgonia.Must(gorgonia.Mul(sum, diff))
        _, _ = gorgonia.Grad(z, x, y)
        vm := gorgonia.NewTapeMachine(g)
        vm.RunAll()
        // x.Grad() → dz/dx, y.Grad() → dz/dy

    В PyTorch это 3 строки:
        x = torch.tensor(x_val, requires_grad=True)
        z = (x + y) * (x - y)
        z.backward()
    """
    x = torch.tensor(x_val, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(y_val, dtype=torch.float64, requires_grad=True)

    z = (x + y) * (x - y)
    z.backward()

    return z.item(), x.grad.item(), y.grad.item()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Параметрические тесты (аналог TestGorgoniaAutogradParametric)
# ═══════════════════════════════════════════════════════════════════════════════

# (x, y, expected_z, expected_dz_dx, expected_dz_dy)
TEST_CASES: List[Tuple[float, float, float, float, float]] = [
    (5.0, 3.0, 16.0, 10.0, -6.0),
    (1.0, 0.0, 1.0, 2.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (10.0, 7.0, 51.0, 20.0, -14.0),
    (-3.0, 2.0, 5.0, -6.0, -4.0),
]


def run_parametric_tests() -> bool:
    """
    Запускает параметрические тесты.
    Аналог TestGorgoniaAutogradParametric в Go.
    """
    print("\n=== Параметрические тесты (PyTorch autograd) ===\n")
    all_passed = True

    for x, y, exp_z, exp_dx, exp_dy in TEST_CASES:
        z, dx, dy = compute_autograd(x, y)
        ok = (
            abs(z - exp_z) < 1e-9
            and abs(dx - exp_dx) < 1e-9
            and abs(dy - exp_dy) < 1e-9
        )
        status = "✓" if ok else "✗"
        print(
            f"  {status} x={x:6.1f}, y={y:6.1f} → "
            f"z={z:8.1f}, dz/dx={dx:8.1f}, dz/dy={dy:8.1f}"
        )
        if not ok:
            print(f"    ОЖИДАЛОСЬ: z={exp_z}, dz/dx={exp_dx}, dz/dy={exp_dy}")
            all_passed = False

    return all_passed


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Бенчмарк
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_autograd(n_iterations: int = 100_000) -> dict:
    """
    Бенчмарк PyTorch autograd: замеряет время одного прямого + обратного прохода.

    Аналог BenchmarkGorgoniaWithGrad в Go.
    """
    # Прогрев (JIT-компиляция, кеширование)
    for _ in range(1000):
        compute_autograd(5.0, 3.0)

    timings = []
    for _ in range(n_iterations):
        start = time.perf_counter_ns()
        compute_autograd(5.0, 3.0)
        elapsed = time.perf_counter_ns() - start
        timings.append(elapsed)

    avg_ns = sum(timings) / len(timings)
    return {
        "method": "pytorch_autograd",
        "iterations": n_iterations,
        "avg_ns_per_op": avg_ns,
        "avg_ms_per_op": avg_ns / 1e6,
        "min_ns": min(timings),
        "max_ns": max(timings),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Главная функция
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Бенчмарк автоградиента PyTorch: аналог Gorgonia-теста"
    )
    parser.add_argument(
        "-n", type=int, default=100_000,
        help="Число итераций бенчмарка (по умолчанию: 100000)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  PyTorch Autograd: z = (x+y)*(x-y)")
    print("  Аналитически: z = x² - y², dz/dx = 2x, dz/dy = -2y")
    print(f"  PyTorch version: {torch.__version__}")
    print("=" * 60)

    # --- Демо (аналог TestGorgoniaAutograd) ---
    print("\n=== Демо: x=5, y=3 ===\n")

    z, dx, dy = compute_autograd(5.0, 3.0)
    print(f"  z = (5+3)*(5-3) = {z:.1f}  (ожидается 16.0)")
    print(f"  dz/dx = 2*5 = {dx:.1f}  (ожидается 10.0)")
    print(f"  dz/dy = -2*3 = {dy:.1f}  (ожидается -6.0)")

    # --- Параметрические тесты ---
    all_passed = run_parametric_tests()
    if all_passed:
        print("\n  Все тесты пройдены ✓")
    else:
        print("\n  ЕСТЬ ОШИБКИ ✗")
        sys.exit(1)

    # --- Бенчмарк ---
    print(f"\n=== Бенчмарк PyTorch autograd (n={args.n}) ===\n")

    result = benchmark_autograd(args.n)
    print(f"  {result['avg_ns_per_op']:.0f} ns/op | "
          f"{result['avg_ms_per_op']:.4f} ms/op")
    print(f"  min={result['min_ns']} ns, max={result['max_ns']} ns")

    # --- Сохранение результатов в JSON ---
    save_results_to_json(result)

    # --- Сравнение API ---
    print("\n" + "=" * 60)
    print("  Сравнение API: PyTorch vs Gorgonia")
    print("=" * 60)
    print("""
  Python (PyTorch):                    Go (Gorgonia):
  ─────────────────                    ──────────────
  x = torch.tensor(5.0,               g := gorgonia.NewGraph()
        requires_grad=True)            x := gorgonia.NewScalar(g,
  y = torch.tensor(3.0,                     Float64, WithValue(5.0))
        requires_grad=True)            y := gorgonia.NewScalar(g, ...)
  z = (x + y) * (x - y)               sum := gorgonia.Must(gorgonia.Add(x, y))
  z.backward()                         diff := gorgonia.Must(gorgonia.Sub(x, y))
  print(x.grad)  # 10.0               z := gorgonia.Must(gorgonia.Mul(sum, diff))
                                        _, _ = gorgonia.Grad(z, x, y)
                                        vm := gorgonia.NewTapeMachine(g)
                                        vm.RunAll()
                                        fmt.Println(x.Grad())  // 10.0

  PyTorch: 3 строки кода, eager mode (define-by-run).
  Gorgonia: 10+ строк, define-and-run (как TensorFlow 1.x).
""")


def save_results_to_json(result: dict) -> None:
    """Сохраняет результаты бенчмарка в JSON файл."""
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
                "torch_version": torch.__version__,
            },
            "results": []
        }
    
    # Добавляем результат автоградиента
    autograd_result = {
        "name": "PyTorchAutograd",
        "mean_ns": result["avg_ns_per_op"],
        "ms_per_op": result["avg_ms_per_op"],
        "min_ns": result["min_ns"],
        "max_ns": result["max_ns"],
        "iterations": result["iterations"],
    }
    
    # Проверяем, есть ли уже результат автоградиента
    existing_idx = None
    for i, r in enumerate(data["results"]):
        if "Autograd" in r.get("name", ""):
            existing_idx = i
            break
    
    if existing_idx is not None:
        data["results"][existing_idx] = autograd_result
    else:
        data["results"].append(autograd_result)
    
    # Сохраняем
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"\n  Результаты сохранены в: {output_path}")


if __name__ == "__main__":
    main()
