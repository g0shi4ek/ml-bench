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
// ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.26 go test -bench=. -benchmem -count=5 ./internal/gorgonia_bench/
package gorgoniabench

import (
	"fmt"
	"math"
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
//  4. Управление памятью узлов:
//     - Каждый узел (*Node) содержит указатель на Value (интерфейс).
//     - При исполнении TapeMachine выделяет память под промежуточные
//       значения и градиенты (dual values).
//     - BindDualValues() привязывает к узлам «двойственные значения»
//       (dual values) — пару (значение, градиент), как в forward-mode AD.
//     - После vm.RunAll() градиенты доступны через node.Deriv().
//     - vm.Close() освобождает все ресурсы ленты.
//
// Ключевое отличие от PyTorch (define-by-run / eager):
// в Gorgonia граф строится явно ДО исполнения, что соответствует
// парадигме define-and-run TF 1.x. Это ограничивает динамические
// архитектуры (условные ветвления, циклы зависящие от данных),
// но открывает возможности статической оптимизации графа
// (constant folding, dead code elimination, operator fusion).
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
	sum := g.Must(g.Add(xNode, yNode))   // узел: x + y
	diff := g.Must(g.Sub(xNode, yNode))  // узел: x - y
	zNode = g.Must(g.Mul(sum, diff))     // узел: (x+y)*(x-y)

	return graph, zNode, xNode, yNode
}

// TestGorgoniaAutograd проверяет корректность автоматического дифференцирования.
//
// Алгоритм:
// 1. Строим граф z = (x+y)*(x-y)
// 2. Вызываем g.Grad(z, x, y) — Gorgonia добавляет узлы обратного прохода
// 3. Создаём TapeMachine с BindDualValues для x и y
// 4. Исполняем граф (RunAll)
// 5. Читаем градиенты через node.Deriv()
func TestGorgoniaAutograd(t *testing.T) {
	graph, zNode, xNode, yNode := buildGraph(5.0, 3.0)

	// Шаг 4: объявить, по каким переменным нужны градиенты.
	// Gorgonia добавит в граф узлы обратного прохода (reverse-mode AD).
	// Это аналог loss.backward() в PyTorch, но выполняется на этапе
	// построения графа, а не при исполнении.
	_, err := g.Grad(zNode, xNode, yNode)
	if err != nil {
		t.Fatalf("Grad: %v", err)
	}

	// Шаг 5: создать TapeMachine и исполнить граф.
	// BindDualValues() — привязать к узлам x и y «двойственные значения»,
	// чтобы после исполнения можно было прочитать градиенты через Deriv().
	vm := g.NewTapeMachine(graph, g.BindDualValues(xNode, yNode))
	defer vm.Close() // освобождает ресурсы ленты и GPU-буферы (если есть)

	if err := vm.RunAll(); err != nil {
		t.Fatalf("RunAll: %v", err)
	}

	// Шаг 6: прочитать результаты.
	// zNode.Value() — значение прямого прохода
	// xNode.Deriv() — градиент dz/dx, записанный при обратном проходе
	zVal := zNode.Value().Data().(float64)

	xDeriv, err := xNode.Grad()
	if err != nil {
		t.Fatalf("xNode.Grad(): %v", err)
	}
	dzdx := xDeriv.Data().(float64)

	yDeriv, err := yNode.Grad()
	if err != nil {
		t.Fatalf("yNode.Grad(): %v", err)
	}
	dzdy := yDeriv.Data().(float64)

	fmt.Printf("\n=== Gorgonia Autograd Demo ===\n")
	fmt.Printf("x = 5.0, y = 3.0\n")
	fmt.Printf("z = (x+y)*(x-y) = %.1f  (ожидается 16.0)\n", zVal)
	fmt.Printf("dz/dx = 2x = %.1f  (ожидается 10.0)\n", dzdx)
	fmt.Printf("dz/dy = -2y = %.1f  (ожидается -6.0)\n", dzdy)

	const eps = 1e-10
	if math.Abs(zVal-16.0) > eps {
		t.Errorf("z: got %.6f, want 16.0", zVal)
	}
	if math.Abs(dzdx-10.0) > eps {
		t.Errorf("dz/dx: got %.6f, want 10.0", dzdx)
	}
	if math.Abs(dzdy-(-6.0)) > eps {
		t.Errorf("dz/dy: got %.6f, want -6.0", dzdy)
	}
}

// TestGorgoniaAutogradParametric проверяет автоградиент для нескольких
// наборов входных значений, подтверждая формулы dz/dx=2x, dz/dy=-2y.
func TestGorgoniaAutogradParametric(t *testing.T) {
	testCases := []struct {
		x, y           float64
		wantZ          float64
		wantDzDx       float64
		wantDzDy       float64
	}{
		{x: 5, y: 3, wantZ: 16, wantDzDx: 10, wantDzDy: -6},
		{x: 1, y: 0, wantZ: 1, wantDzDx: 2, wantDzDy: 0},
		{x: 0, y: 0, wantZ: 0, wantDzDx: 0, wantDzDy: 0},
		{x: 10, y: 7, wantZ: 51, wantDzDx: 20, wantDzDy: -14},
		{x: -3, y: 2, wantZ: 5, wantDzDx: -6, wantDzDy: -4},
	}

	for _, tc := range testCases {
		name := fmt.Sprintf("x=%.0f_y=%.0f", tc.x, tc.y)
		t.Run(name, func(t *testing.T) {
			graph, zNode, xNode, yNode := buildGraph(tc.x, tc.y)

			_, err := g.Grad(zNode, xNode, yNode)
			if err != nil {
				t.Fatalf("Grad: %v", err)
			}

			vm := g.NewTapeMachine(graph, g.BindDualValues(xNode, yNode))
			defer vm.Close()

			if err := vm.RunAll(); err != nil {
				t.Fatalf("RunAll: %v", err)
			}

			zVal := zNode.Value().Data().(float64)
			xGrad, _ := xNode.Grad()
			yGrad, _ := yNode.Grad()
			dzdx := xGrad.Data().(float64)
			dzdy := yGrad.Data().(float64)

			const eps = 1e-10
			if math.Abs(zVal-tc.wantZ) > eps {
				t.Errorf("z: got %.6f, want %.1f", zVal, tc.wantZ)
			}
			if math.Abs(dzdx-tc.wantDzDx) > eps {
				t.Errorf("dz/dx: got %.6f, want %.1f", dzdx, tc.wantDzDx)
			}
			if math.Abs(dzdy-tc.wantDzDy) > eps {
				t.Errorf("dz/dy: got %.6f, want %.1f", dzdy, tc.wantDzDy)
			}
		})
	}
}

// BenchmarkGorgoniaInference замеряет инференс (только прямой проход)
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

// BenchmarkGorgoniaWithGrad замеряет полный цикл: прямой + обратный проход.
func BenchmarkGorgoniaWithGrad(b *testing.B) {
	graph, zNode, xNode, yNode := buildGraph(5.0, 3.0)

	_, err := g.Grad(zNode, xNode, yNode)
	if err != nil {
		b.Fatalf("Grad: %v", err)
	}

	vm := g.NewTapeMachine(graph, g.BindDualValues(xNode, yNode))
	defer vm.Close()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		vm.Reset()
		if err := vm.RunAll(); err != nil {
			b.Fatalf("RunAll: %v", err)
		}
		_ = zNode.Value()
	}
}
