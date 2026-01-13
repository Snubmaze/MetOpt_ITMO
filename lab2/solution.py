import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import os
from typing import Callable, Tuple, List

class GlobalExtremumFinder:
    """
    Класс для поиска глобального минимума липшицевой функции
    методом ломаных (метод Пиявского)
    """
    
    def __init__(self, func_str: str, a: float, b: float, eps: float):
        """
        Инициализация параметров задачи
        
        :param func_str: строка с функцией (например, "x + np.sin(3.14159*x)")
        :param a: левая граница отрезка
        :param b: правая граница отрезка
        :param eps: требуемая точность
        """
        self.func_str = func_str
        self.a = a
        self.b = b
        self.eps = eps
        self.func = self._parse_function(func_str)
        self.L = None  # Константа Липшица
        self.points = []  # История точек испытаний
        self.iterations = 0
        self.computation_time = 0
        
    def _parse_function(self, func_str: str) -> Callable:
        """Преобразование строки в функцию"""
        def f(x):
            # Добавляем поддержку pi
            return eval(func_str, {"x": x, "np": np, "sin": np.sin, 
                                   "cos": np.cos, "exp": np.exp, 
                                   "sqrt": np.sqrt, "abs": np.abs,
                                   "pi": np.pi})
        return f
    
    def _estimate_lipschitz(self, samples: int = 1000) -> float:
        """
        Оценка константы Липшица численным методом
        
        :param samples: количество точек для оценки
        :return: оценка константы Липшица
        """
        x_samples = np.linspace(self.a, self.b, samples)
        y_samples = np.array([self.func(x) for x in x_samples])
        
        # Вычисляем максимальный наклон
        max_slope = 0
        for i in range(len(x_samples) - 1):
            slope = abs((y_samples[i+1] - y_samples[i]) / 
                       (x_samples[i+1] - x_samples[i]))
            max_slope = max(max_slope, slope)
        
        # Увеличиваем оценку для надежности
        return max_slope * 2.0 if max_slope > 0 else 1.0
    
    def _calculate_R(self, i: int, points: List[Tuple[float, float]]) -> float:
        """
        Вычисление характеристики R для интервала i
        
        :param i: индекс интервала
        :param points: отсортированный список точек (x, f(x))
        :return: значение характеристики
        """
        x_prev, y_prev = points[i]
        x_next, y_next = points[i + 1]
        
        # Характеристика по методу Пиявского
        R = (x_next - x_prev) / 2 - (y_next + y_prev) / (2 * self.L)
        
        return R
    
    def solve(self) -> Tuple[float, float]:
        """
        Решение задачи поиска глобального минимума
        
        :return: кортеж (x_min, f_min) - аргумент и значение минимума
        """
        start_time = time.time()
        
        # Оценка константы Липшица
        self.L = self._estimate_lipschitz()
        print(f"Оценка константы Липшица: L = {self.L:.4f}")
        
        # Инициализация: вычисляем функцию на концах отрезка
        points = [(self.a, self.func(self.a)), (self.b, self.func(self.b))]
        self.points.append(points.copy())
        self.iterations = 0
        
        while True:
            self.iterations += 1
            
            # Сортируем точки по x
            points.sort(key=lambda p: p[0])
            
            # Находим интервал с максимальной характеристикой
            max_R = float('-inf')
            max_idx = 0
            
            for i in range(len(points) - 1):
                R = self._calculate_R(i, points)
                if R > max_R:
                    max_R = R
                    max_idx = i
            
            # Новая точка испытания
            x_prev, y_prev = points[max_idx]
            x_next, y_next = points[max_idx + 1]
            
            x_new = (x_prev + x_next) / 2 - (y_next - y_prev) / (2 * self.L)
            y_new = self.func(x_new)
            
            points.insert(max_idx + 1, (x_new, y_new))
            self.points.append(points.copy())
            
            # Проверка критерия останова
            if x_next - x_prev < self.eps:
                break
            
            # Ограничение на количество итераций
            if self.iterations > 10000:
                print("Достигнуто максимальное число итераций")
                break
        
        self.computation_time = time.time() - start_time
        
        # Находим минимум среди всех точек
        min_point = min(points, key=lambda p: p[1])
        
        return min_point
    
    def visualize(self, x_min: float, f_min: float):
        """
        Визуализация результатов
        
        :param x_min: аргумент минимума
        :param f_min: значение минимума
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # График исходной функции
        x_plot = np.linspace(self.a, self.b, 1000)
        y_plot = [self.func(x) for x in x_plot]
        
        # Верхний график: функция и точки испытаний
        ax1 = axes[0]
        ax1.plot(x_plot, y_plot, 'b-', linewidth=2, label='Исходная функция')
        
        # Все точки испытаний
        final_points = self.points[-1]
        x_trials = [p[0] for p in final_points]
        y_trials = [p[1] for p in final_points]
        ax1.scatter(x_trials, y_trials, c='orange', s=50, 
                   zorder=5, label='Точки испытаний')
        
        # Минимум
        ax1.scatter([x_min], [f_min], c='red', s=200, marker='*', 
                   zorder=6, label=f'Минимум: ({x_min:.4f}, {f_min:.4f})')
        
        # Построение ломаной (верхняя оценка)
        self._draw_piecewise_linear(ax1, final_points)
        
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('f(x)', fontsize=12)
        ax1.set_title('Поиск глобального минимума методом ломаных', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Нижний график: сходимость
        ax2 = axes[1]
        
        # Значения минимума на каждой итерации
        min_values = []
        for pts in self.points:
            current_min = min(pts, key=lambda p: p[1])[1]
            min_values.append(current_min)
        
        ax2.plot(range(len(min_values)), min_values, 'g-', 
                linewidth=2, marker='o', markersize=4)
        ax2.axhline(y=f_min, color='r', linestyle='--', 
                   label=f'Найденный минимум: {f_min:.6f}')
        ax2.set_xlabel('Итерация', fontsize=12)
        ax2.set_ylabel('Текущий минимум', fontsize=12)
        ax2.set_title('Сходимость алгоритма', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def _draw_piecewise_linear(self, ax, points):
        """
        Отрисовка вспомогательной ломаной (верхней оценки функции)
        
        :param ax: объект axes для рисования
        :param points: список точек
        """
        # Строим верхнюю оценку для каждого интервала
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            # Вершина "конуса"
            x_mid = (x1 + x2) / 2 - (y2 - y1) / (2 * self.L)
            y_mid = (y1 + y2) / 2 - self.L * abs(x2 - x1) / 2
            
            # Рисуем две линии конуса
            x_cone = [x1, x_mid, x2]
            y_cone = [y1, y_mid, y2]
            
            if i == 0:
                ax.plot(x_cone, y_cone, 'r--', alpha=0.5, linewidth=1, 
                       label='Вспомогательная ломаная')
            else:
                ax.plot(x_cone, y_cone, 'r--', alpha=0.5, linewidth=1)
    
    def print_results(self, x_min: float, f_min: float):
        """Вывод результатов в консоль"""
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
        print("="*60)
        print(f"Функция: f(x) = {self.func_str}")
        print(f"Отрезок: [{self.a}, {self.b}]")
        print(f"Точность: ε = {self.eps}")
        print(f"Константа Липшица: L = {self.L:.6f}")
        print("-"*60)
        print(f"Аргумент минимума: x* = {x_min:.8f}")
        print(f"Значение минимума: f(x*) = {f_min:.8f}")
        print(f"Количество итераций: {self.iterations}")
        print(f"Время вычисления: {self.computation_time:.6f} сек")
        print("="*60)


def read_input_from_file(filename: str = "input.txt") -> Tuple[str, float, float, float]:
    """
    Чтение входных данных из файла
    
    Формат файла:
    строка 1: функция (например, x + sin(pi*x))
    строка 2: левая граница отрезка
    строка 3: правая граница отрезка
    строка 4: точность epsilon
    
    :param filename: имя файла с входными данными
    :return: кортеж (функция_строка, a, b, epsilon)
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 4:
            raise ValueError("Недостаточно данных в файле. Ожидается 4 строки.")
        
        function_str = lines[0].strip()
        left_bound = float(lines[1].strip())
        right_bound = float(lines[2].strip())
        epsilon = float(lines[3].strip())
        
        # Валидация данных
        if left_bound >= right_bound:
            raise ValueError("Левая граница должна быть меньше правой")
        if epsilon <= 0:
            raise ValueError("Точность должна быть положительным числом")
        
        print(f"Данные успешно прочитаны из файла '{filename}':")
        print(f"  Функция: {function_str}")
        print(f"  Отрезок: [{left_bound}, {right_bound}]")
        print(f"  Точность: {epsilon}\n")
        
        return function_str, left_bound, right_bound, epsilon
        
    except FileNotFoundError:
        print(f"Ошибка: файл '{filename}' не найден!")
        print("Создаю файл с примером данных...")
        create_example_input_file(filename)
        raise
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        raise


def create_example_input_file(filename: str = "input.txt"):
    """
    Создание примера файла с входными данными
    
    :param filename: имя файла для создания
    """
    example_content = """x + sin(pi*x)
-2
4
0.01"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(example_content)
    
    print(f"Файл '{filename}' создан с примером данных.")
    print("Пожалуйста, запустите программу снова.")


if __name__ == "__main__":
    print("Программа поиска глобального минимума липшицевой функции")
    print("="*60 + "\n")
    
    try:
        function_string, left_bound, right_bound, precision = read_input_from_file("input.txt")
    except:
        print("\nПрограмма завершена с ошибкой.")
        exit(1)
    
    print("Запуск алгоритма оптимизации...\n")
    
    # Создание объекта и решение задачи
    finder = GlobalExtremumFinder(function_string, left_bound, right_bound, precision)
    x_min, f_min = finder.solve()
    
    finder.print_results(x_min, f_min)
    
    finder.visualize(x_min, f_min)