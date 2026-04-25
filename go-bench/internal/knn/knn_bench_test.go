// Package knn реализует бенчмарк инференса KNN-классификатора
// с использованием библиотеки github.com/sjwhitworth/golearn.
//
// Запуск бенчмарка:
//
//	go test -bench=. -benchmem -benchtime=10s ./internal/knn/
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
	// Загрузка датасета Iris из CSV
	raw, err := base.ParseCSVToInstances("../../testdata/iris.csv", true)
	if err != nil {
		panic("не удалось загрузить iris.csv: " + err.Error())
	}

	// Разбивка 70% обучение / 30% тест
	train, test := base.InstancesTrainTestSplit(raw, 0.70)

	// Создание и обучение классификатора
	cls := knn.NewKnnClassifier("euclidean", "linear", 3)
	cls.Fit(train)

	// Получаем первую строку через Slice
	// DenseInstances имеет метод Slice для извлечения подвыборки
	var singleSample base.FixedDataGrid
	if denseTest, ok := test.(*base.DenseInstances); ok {
		singleSample = denseTest.Slice(0, 1)
	} else {
		panic("test is not *base.DenseInstances")
	}

	return cls, singleSample
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
	raw, err := base.ParseCSVToInstances("../../testdata/iris.csv", true)
	if err != nil {
		panic(err)
	}
	
	train, test := base.InstancesTrainTestSplit(raw, 0.70)
	cls := knn.NewKnnClassifier("euclidean", "linear", 3)
	cls.Fit(train)

	// Берём первые 32 строки тестовой выборки (или меньше)
	_, testRows := test.Size()
	batchSize := 32
	if testRows < batchSize {
		batchSize = testRows
	}

	// Используем Slice для создания батча
	var batch base.FixedDataGrid
	if denseTest, ok := test.(*base.DenseInstances); ok {
		batch = denseTest.Slice(0, batchSize)
	} else {
		panic("test is not *base.DenseInstances")
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := cls.Predict(batch)
		if err != nil {
			b.Fatalf("Predict batch error: %v", err)
		}
	}
}

// BenchmarkKNNInferenceFullTest - простой бенчмарк на всём тестовом наборе
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