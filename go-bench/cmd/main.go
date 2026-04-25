// cmd/bench — запускает все бенчмарки и сохраняет результаты в JSON.
//
// Использование:
//
//	go run ./cmd/bench/ --output results/go_results.json
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"time"

	"gonum.org/v1/gonum/mat"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/knn"
)

// BenchResult хранит результаты одного бенчмарка.
type BenchResult struct {
	Name        string    `json:"name"`
	Iterations  int       `json:"iterations"`
	NsPerOp     float64   `json:"ns_per_op"`
	MsPerOp     float64   `json:"ms_per_op"`
	AllocsPerOp float64   `json:"allocs_per_op"`
	BytesPerOp  float64   `json:"bytes_per_op"`
	Timestamp   time.Time `json:"timestamp"`
	GoVersion   string    `json:"go_version"`
	GOOS        string    `json:"goos"`
	GOARCH      string    `json:"goarch"`
}

// SystemInfo содержит информацию об окружении.
type SystemInfo struct {
	GoVersion string `json:"go_version"`
	GOOS      string `json:"goos"`
	GOARCH    string `json:"goarch"`
	NumCPU    int    `json:"num_cpu"`
	GOMAXPROCS int   `json:"gomaxprocs"`
}

// Report — итоговый отчёт.
type Report struct {
	System  SystemInfo   `json:"system"`
	Results []BenchResult `json:"results"`
}

func main() {
	outputPath := flag.String("output", "results/go_results.json", "путь к файлу результатов")
	iterations := flag.Int("n", 10000, "число итераций для ручных бенчмарков")
	flag.Parse()

	sysInfo := SystemInfo{
		GoVersion:   runtime.Version(),
		GOOS:        runtime.GOOS,
		GOARCH:      runtime.GOARCH,
		NumCPU:      runtime.NumCPU(),
		GOMAXPROCS:  runtime.GOMAXPROCS(0),
	}

	fmt.Printf("Система: %s %s %s\n", sysInfo.GoVersion, sysInfo.GOOS, sysInfo.GOARCH)
	fmt.Printf("CPU: %d ядер, GOMAXPROCS=%d\n\n", sysInfo.NumCPU, sysInfo.GOMAXPROCS)

	var results []BenchResult

	// --- Бенчмарк 1: KNN Inference ---
	fmt.Printf("[1/3] Бенчмарк KNN инференса (n=%d)...\n", *iterations)
	knnResult := benchKNN(*iterations)
	results = append(results, knnResult)
	fmt.Printf("  KNN  : %.0f ns/op | %.3f ms/op\n", knnResult.NsPerOp, knnResult.MsPerOp)

	// --- Бенчмарк 2: Gonum матричное умножение ---
	fmt.Printf("[2/3] Бенчмарк Gonum матричного умножения (n=%d)...\n", *iterations)
	gonumResult := benchGonum(*iterations)
	results = append(results, gonumResult)
	fmt.Printf("  Gonum: %.0f ns/op | %.3f ms/op\n", gonumResult.NsPerOp, gonumResult.MsPerOp)

	// --- Бенчмарк 3: Naive slice умножение ---
	fmt.Printf("[3/3] Бенчмарк наивного умножения срезов (n=%d)...\n", *iterations)
	naiveResult := benchNaiveSlice(*iterations)
	results = append(results, naiveResult)
	fmt.Printf("  Naive: %.0f ns/op | %.3f ms/op\n", naiveResult.NsPerOp, naiveResult.MsPerOp)

	// Speedup Gonum vs Naive
	if naiveResult.NsPerOp > 0 {
		speedup := naiveResult.NsPerOp / gonumResult.NsPerOp
		fmt.Printf("\nGonum быстрее naive в %.1fx раз\n", speedup)
	}

	// --- Запись результатов ---
	report := Report{System: sysInfo, Results: results}
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "json marshal error: %v\n", err)
		os.Exit(1)
	}

	if err := os.WriteFile(*outputPath, data, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "write error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("\nРезультаты сохранены в: %s\n", *outputPath)
}

func benchKNN(n int) BenchResult {
	raw, err := base.ParseCSVToInstances("testdata/iris.csv", true)
	if err != nil {
		panic("iris.csv not found. Run from project root: cp testdata/iris.csv .")
	}
	train, test := base.InstancesTrainTestSplit(raw, 0.70)
	cls := knn.NewKnnClassifier("euclidean", "linear", 3)
	cls.Fit(train)
	sample, _ := base.GetInstancesFromRange(test, 0, 1)

	// Прогрев
	for i := 0; i < 100; i++ {
		_, _ = cls.Predict(sample)
	}

	start := time.Now()
	for i := 0; i < n; i++ {
		_, _ = cls.Predict(sample)
	}
	elapsed := time.Since(start)

	nsPerOp := float64(elapsed.Nanoseconds()) / float64(n)
	return BenchResult{
		Name:       "KNNInference_single",
		Iterations: n,
		NsPerOp:    nsPerOp,
		MsPerOp:    nsPerOp / 1e6,
		Timestamp:  time.Now(),
		GoVersion:  runtime.Version(),
		GOOS:       runtime.GOOS,
		GOARCH:     runtime.GOARCH,
	}
}

func benchGonum(n int) BenchResult {
	dataX := make([]float64, 150*4)
	dataW := make([]float64, 4)
	for i := range dataX { dataX[i] = rand.Float64() }
	for i := range dataW { dataW[i] = rand.Float64() }
	X := mat.NewDense(150, 4, dataX)
	W := mat.NewDense(4, 1, dataW)
	result := mat.NewDense(150, 1, nil)

	// Прогрев
	for i := 0; i < 100; i++ {
		result.Mul(X, W)
	}

	start := time.Now()
	for i := 0; i < n; i++ {
		result.Mul(X, W)
	}
	elapsed := time.Since(start)
	_ = result.At(0, 0)

	nsPerOp := float64(elapsed.Nanoseconds()) / float64(n)
	return BenchResult{
		Name:       "GonumMatMul_150x4",
		Iterations: n,
		NsPerOp:    nsPerOp,
		MsPerOp:    nsPerOp / 1e6,
		Timestamp:  time.Now(),
		GoVersion:  runtime.Version(),
		GOOS:       runtime.GOOS,
		GOARCH:     runtime.GOARCH,
	}
}

func benchNaiveSlice(n int) BenchResult {
	X := make([][]float64, 150)
	for i := range X {
		X[i] = make([]float64, 4)
		for j := range X[i] { X[i][j] = rand.Float64() }
	}
	W := make([]float64, 4)
	for i := range W { W[i] = rand.Float64() }

	// Прогрев
	for i := 0; i < 100; i++ {
		result := make([]float64, 150)
		for row := range X {
			for j := range W {
				result[row] += X[row][j] * W[j]
			}
		}
		_ = result[0]
	}

	start := time.Now()
	for i := 0; i < n; i++ {
		result := make([]float64, 150)
		for row := range X {
			for j := range W {
				result[row] += X[row][j] * W[j]
			}
		}
		_ = result[0]
	}
	elapsed := time.Since(start)

	nsPerOp := float64(elapsed.Nanoseconds()) / float64(n)
	return BenchResult{
		Name:       "NaiveSliceMul_150x4",
		Iterations: n,
		NsPerOp:    nsPerOp,
		MsPerOp:    nsPerOp / 1e6,
		Timestamp:  time.Now(),
		GoVersion:  runtime.Version(),
		GOOS:       runtime.GOOS,
		GOARCH:     runtime.GOARCH,
	}
}
