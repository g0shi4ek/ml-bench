// Package knn реализует бенчмарк инференса KNN-классификатора
// с использованием библиотеки github.com/sjwhitworth/golearn.
//
// Запуск бенчмарка:
//
//	go test -bench=. -benchmem -count=5 -benchtime=10s ./internal/knn/
package knn

import (
	"testing"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/knn"
)

// loadIrisAndTrain загружает датасет Iris, разбивает на train/test
// и обучает KNN-классификатор (k=3, метрика — евклидово расстояние).
// Возвращает обученный классификатор и один тестовый экземпляр.
func loadIrisAndTrain() (*knn.KNNClassifier, base.FixedDataGrid) {
	raw, err := base.ParseCSVToInstances("../../testdata/iris.csv", true)
	if err != nil {
		panic("не удалось загрузить iris.csv: " + err.Error())
	}

	// Разбивка 70% обучение / 30% тест
	train, test := base.InstancesTrainTestSplit(raw, 0.70)

	cls := knn.NewKnnClassifier("euclidean", "linear", 3)
	cls.Fit(train)

	singleSample := base.NewInstancesViewFromVisible(test, []int{0}, test.AllAttributes())

	return cls, singleSample
}

func loadIrisAndTrainBatch(batchSize int) (*knn.KNNClassifier, base.FixedDataGrid) {
	raw, err := base.ParseCSVToInstances("../../testdata/iris.csv", true)
	if err != nil {
		panic("не удалось загрузить iris.csv: " + err.Error())
	}

	train, test := base.InstancesTrainTestSplit(raw, 0.70)
	cls := knn.NewKnnClassifier("euclidean", "linear", 3)
	cls.Fit(train)

	_, testRows := test.Size()
	if testRows < batchSize {
		batchSize = testRows
	}

	rowIndices := make([]int, batchSize)
	for i := range rowIndices {
		rowIndices[i] = i
	}
	batch := base.NewInstancesViewFromVisible(test, rowIndices, test.AllAttributes())

	return cls, batch
}

// BenchmarkKNNInference замеряет время выполнения одного предсказания
func BenchmarkKNNInference(b *testing.B) {
	cls, sample := loadIrisAndTrain()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := cls.Predict(sample)
		if err != nil {
			b.Fatalf("Predict error: %v", err)
		}
	}
}

// BenchmarkKNNInferenceBatch32 замеряет инференс для батча из 32 сэмплов
func BenchmarkKNNInferenceBatch32(b *testing.B) {
	cls, batch := loadIrisAndTrainBatch(32)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := cls.Predict(batch)
		if err != nil {
			b.Fatalf("Predict batch error: %v", err)
		}
	}
}

// BenchmarkKNNInferenceFullTest — бенчмарк на всём тестовом наборе
func BenchmarkKNNInferenceFullTest(b *testing.B) {
	raw, err := base.ParseCSVToInstances("../../testdata/iris.csv", true)
	if err != nil {
		panic(err)
	}

	train, test := base.InstancesTrainTestSplit(raw, 0.70)
	cls := knn.NewKnnClassifier("euclidean", "linear", 3)
	cls.Fit(train)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := cls.Predict(test)
		if err != nil {
			b.Fatalf("Predict error: %v", err)
		}
	}
}