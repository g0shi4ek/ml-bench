#!/usr/bin/env bash
# Создание виртуального окружения и установка зависимостей
# для Python-бенчмарков проекта ml-bench.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "=== Настройка виртуального окружения ==="
echo "  Директория: ${VENV_DIR}"

# Проверяем наличие Python 3
if ! command -v python3 &>/dev/null; then
    echo "ОШИБКА: python3 не найден. Установите Python 3.9+."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1)
echo "  Python: ${PYTHON_VERSION}"

if [ -d "${VENV_DIR}" ]; then
    echo "  venv уже существует, пересоздаём..."
    rm -rf "${VENV_DIR}"
fi

python3 -m venv "${VENV_DIR}"
echo "  venv создан."

source "${VENV_DIR}/bin/activate"

echo "  Обновляем pip..."
pip install --upgrade pip --quiet

echo "  Устанавливаем зависимости из requirements.txt..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

echo ""
echo "Готово"
