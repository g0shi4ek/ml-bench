// Package gorgoniabench демонстрирует построение вычислительного графа
// в Gorgonia и автоматическое дифференцирование (autograd).
//
// Задача: вычислить z = (x + y) * (x - y) и получить градиенты dz/dx, dz/dy.
//
// Математически:
//
//	z  = (x + y)(x - y) = x² - y²
//	dz/dx = 2x
//	dz/dy = -2y
//
// При x=5, y=3: z=16, dz/dx=10, dz/dy=-6
//
// Запуск:
//
//	go test -v -run TestGorgoniaAutograd ./internal/gorgonia_bench/
//	go test -bench=BenchmarkGorgoniaInference -benchmem ./internal/gorgonia_bench/
package gorgoniabench

import (
	"fmt"
	"testing"

	g "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// buildGraph строит вычислительный граф для z = (x+y)*(x-y).
//
// Архитектура памяти Gorgonia (аналог TensorFlow 1.x):
//
//  1. Граф (ExprGraph) — направленный ациклический граф (DAG), где узлы
//     хранят операции и форму тензора, но НЕ значения. Это схема вычисления.
//
//  2. Значения (Value) — присваиваются узлам через WithValue() или
//     вычисляются VM при исполнении. Хранятся в отдельной области памяти.
//
//  3. TapeMachine — «лента Тьюринга» для записи операций. При RunAll()
//     последовательно исполняет узлы графа в топологическом порядке,
//     записывая промежуточные значения на «ленту» для обратного прохода.
//     Это прямой аналог Session.run() в TensorFlow 1.x.
//
// Ключевое отличие от PyTorch (define-by-run / eager):
// в Gorgonia граф строится явно ДО исполнения, что соответствует
// парадигме define-and-run TF 1.x. Это ограничивает динамические
// архитектуры, но открывает возможности статической оптимизации графа.
func buildGraph(xVal, yVal float64) (
	graph *g.ExprGraph,
	zNode *g.Node,
	xNode, yNode *g.Node,
) {
	// Шаг 1: создать пустой граф — контейнер для узлов и рёбер
	graph = g.NewGraph()

	// Шаг 2: создать входные узлы-скаляры типа float64.
	// g.WithValue() задаёт начальное значение узла.
	// g.WithName() нужен для читаемой визуализации графа (GraphViz).
	xNode = g.NewScalar(graph, tensor.Float64,
		g.WithName("x"),
		g.WithValue(xVal),
	)
	yNode = g.NewScalar(graph, tensor.Float64,
		g.WithName("y"),
		g.WithValue(yVal),
	)

	// Шаг 3: построить граф вычислений.
	// g.Must() — вспомогательная обёртка: паникует при ошибке построения.
	// Каждый вызов Add/Sub/Mul добавляет узел в граф и возвращает его.
	sum  := g.Must(g.Add(xNode, yNode))  // узел: x + y
	diff := g.Must(g.Sub(xNode, yNode))  // узел: x - y
	zNode = g.Must(g.Mul(sum, diff))     // узел: (x+y)*(x-y)

	return graph, zNode, xNode, yNode
}

// TestGorgoniaAutograd проверяет корректность автоматического дифференцирования.
func TestGorgoniaAutograd(t *testing.T) {
	graph, zNode, xNode, yNode := buildGraph(5.0, 3.0)

	// Шаг 4: объявить, по каким переменным нужны градиенты.
	// Gorgonia добавит в граф узлы обратного прохода.
	// Метод Grad() возвращает узлы-градиенты в том же порядке.
	grads, err := g.Grad(zNode, xNode, yNode)
	if err != nil {
		t.Fatalf("Grad: %v", err)
	}

	// Шаг 5: создать TapeMachine и исполнить граф.
	// WithLeaveOnTape() — оставлять значения на ленте после RunAll()
	// (нужно для последующего чтения градиентов через .Deriv()).
	vm := g.NewTapeMachine(graph, g.BindDualValues(xNode, yNode))
	defer vm.Close() // освобождает ресурсы ленты и GPU-буферы (если есть)

	if err := vm.RunAll(); err != nil {
		t.Fatalf("RunAll: %v", err)
	}

	// Шаг 6: прочитать результаты.
	zVal := zNode.Value().Data().(float64)
	dzdx := grads[0].Value().Data().(float64)
	dzdy := grads[1].Value().Data().(float64)

	fmt.Printf("\n=== Gorgonia Autograd Demo ===\n")
	fmt.Printf("x = 5.0, y = 3.0\n")
	fmt.Printf("z = (x+y)*(x-y) = %.1f  (ожидается 16.0)\n", zVal)
	fmt.Printf("dz/dx = 2x = %.1f  (ожидается 10.0)\n", dzdx)
	fmt.Printf("dz/dy = -2y = %.1f  (ожидается -6.0)\n", dzdy)

	// Проверки корректности
	if zVal != 16.0 {
		t.Errorf("z: got %.1f, want 16.0", zVal)
	}
	if dzdx != 10.0 {
		t.Errorf("dz/dx: got %.1f, want 10.0", dzdx)
	}
	if dzdy != -6.0 {
		t.Errorf("dz/dy: got %.1f, want -6.0", dzdy)
	}
}

// BenchmarkGorgoniaInference замеряет инференс (только прямой проход)
// без пересборки графа — как это происходит в продакшн-сервисе.
func BenchmarkGorgoniaInference(b *testing.B) {
	// Граф строится один раз ДО цикла бенчмарка
	graph, zNode, xNode, yNode := buildGraph(5.0, 3.0)
	vm := g.NewTapeMachine(graph)
	defer vm.Close()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// В реальном инференс-сервисе здесь менялись бы значения входных узлов:
		// g.Let(xNode, newXValue)
		// g.Let(yNode, newYValue)
		// Для чистоты бенчмарка измеряем только RunAll
		_ = xNode
		_ = yNode
		vm.Reset() // сброс ленты для повторного исполнения
		if err := vm.RunAll(); err != nil {
			b.Fatalf("RunAll: %v", err)
		}
		_ = zNode.Value()
	}
}
