import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

BENCHMARKS = [
    ("KNN инференс (scikit-learn)", ["python", "bench_knn.py"]),
    ("NumPy матричное умножение", ["python", "bench_numpy.py"]),
    ("PyTorch автоградиент", ["python", "bench_autograd.py", "-n", "10000"]),
]


def main() -> None:
    print("=" * 60)
    print("  Python ML Benchmarks — полный запуск")
    print("=" * 60)

    failed = []

    for i, (name, cmd) in enumerate(BENCHMARKS, 1):
        print(f"\n{'─' * 60}")
        print(f"  [{i}/{len(BENCHMARKS)}] {name}")
        print(f"{'─' * 60}\n")

        result = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            env={**__import__("os").environ},
        )

        if result.returncode != 0:
            print(f"\n  ⚠ {name}: завершился с ошибкой (код {result.returncode})")
            failed.append(name)
        else:
            print(f"\n  ✓ {name}: успешно")

    print(f"\n{'=' * 60}")
    if failed:
        print(f"  Ошибки в {len(failed)} бенчмарках:")
        for f in failed:
            print(f"    ✗ {f}")
        sys.exit(1)
    else:
        print(f"  Все {len(BENCHMARKS)} бенчмарков выполнены успешно ✓")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
