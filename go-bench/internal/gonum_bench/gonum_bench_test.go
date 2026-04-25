// Package gonumbench демонстрирует использование gonum/mat для операций
// линейной алгебры и сравнивает производительность с наивной реализацией
// на срезах [][]float64.
//
// Запуск:
//
//	go test -bench=. -benchmem -count=5 ./internal/gonum_bench/
package gonumbench

import (
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

const (
	rows = 150 // число объектов (аналог датасета Iris: 150 строк)
	cols = 4   // число признаков (Iris: 4 признака)
)

// ----- Вспомогательные функции инициализации -----

// makeGonumMatrices создаёт матрицу признаков X (150x4) и вектор весов W (4x1)
// с помощью gonum/mat.NewDense. Значения заполняются случайными числами.
func makeGonumMatrices() (*mat.Dense, *mat.Dense) {
	dataX := make([]float64, rows*cols)
	for i := range dataX {
		dataX[i] = rand.Float64()
	}
	dataW := make([]float64, cols*1)
	for i := range dataW {
		dataW[i] = rand.Float64()
	}
	X := mat.NewDense(rows, cols, dataX)
	W := mat.NewDense(cols, 1, dataW)
	return X, W
}

// makeSliceMatrices создаёт эквивалентные матрицы как [][]float64.
func makeSliceMatrices() ([][]float64, []float64) {
	X := make([][]float64, rows)
	for i := range X {
		X[i] = make([]float64, cols)
		for j := range X[i] {
			X[i][j] = rand.Float64()
		}
	}
	W := make([]float64, cols)
	for i := range W {
		W[i] = rand.Float64()
	}
	return X, W
}

// naiveMul наивное матрично-векторное произведение на срезах.
// Сложность O(rows*cols). Результат: вектор длиной rows.
func naiveMul(X [][]float64, W []float64) []float64 {
	result := make([]float64, len(X))
	for i, row := range X {
		sum := 0.0
		for j, v := range row {
			sum += v * W[j]
		}
		result[i] = sum
	}
	return result
}

// ----- Бенчмарки -----

// BenchmarkGonumMatMul замеряет время умножения X(150x4) * W(4x1) через gonum/mat.
//
// Gonum под капотом использует оптимизированный BLAS-бэкенд (netlib/blas или
// gonum/blas/native), который применяет блочные алгоритмы и, при наличии
// соответствующей сборки с CGO, — Intel MKL / OpenBLAS с SIMD-инструкциями.
// Даже без CGO встроенная реализация gonum/blas/native использует
// ручную развёртку циклов (loop unrolling) для small dense matrices.
func BenchmarkGonumMatMul(b *testing.B) {
	X, W := makeGonumMatrices()
	result := mat.NewDense(rows, 1, nil) // предвыделяем — избегаем аллокаций в цикле

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// result = X * W
		// Mul использует DGEMM из BLAS — оптимизированное ядро
		result.Mul(X, W)
	}

	// Используем result, чтобы компилятор не оптимизировал цикл
	_ = result.At(0, 0)
}

// BenchmarkNaiveSliceMul замеряет наивное матрично-векторное произведение
// на чистых Go-срезах без какой-либо BLAS-оптимизации.
//
// Ожидаемый результат: значительно медленнее Gonum для больших матриц,
// поскольку:
// 1. Нет оптимизации доступа к памяти (cache locality хуже для [][]float64)
// 2. Нет SIMD-векторизации (компилятор Go генерирует её частично,
//    но хуже вручную написанного BLAS-кода)
// 3. make([]float64, rows) внутри naiveMul создаёт аллокацию каждый вызов
func BenchmarkNaiveSliceMul(b *testing.B) {
	X, W := makeSliceMatrices()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		result := naiveMul(X, W)
		_ = result[0]
	}
}

// BenchmarkGonumMatMulPrealloc демонстрирует важность предварительного
// выделения результирующей матрицы для минимизации GC-давления.
func BenchmarkGonumMatMulPrealloc(b *testing.B) {
	X, W := makeGonumMatrices()

	b.Run("with_prealloc", func(b *testing.B) {
		result := mat.NewDense(rows, 1, nil)
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			result.Mul(X, W)
		}
		_ = result.At(0, 0)
	})

	b.Run("without_prealloc", func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			// Каждую итерацию создаём новую матрицу — нагрузка на GC
			result := mat.NewDense(rows, 1, nil)
			result.Mul(X, W)
			_ = result.At(0, 0)
		}
	})
}
