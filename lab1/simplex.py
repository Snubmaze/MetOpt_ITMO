from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict

EPS = 1e-9


class LPTask:
    """
    Класс описывает задачу линейного программирования.
    Отвечает только за чтение и хранение данных.
    """

    def __init__(self, c: np.ndarray, A: np.ndarray,
                 signs: List[str], b: np.ndarray):
        self.c = c
        self.A = A
        self.signs = signs
        self.b = b

    @staticmethod
    def from_file(filename: str) -> "LPTask":
        with open(f"./lab1/{filename}") as f:
            lines = [l.strip() for l in f if l.strip()]

        first = lines[0].split()
        c = np.array(list(map(float, first[1:])))

        A, signs, b = [], [], []

        for line in lines[1:]:
            parts = line.split()
            A.append(list(map(float, parts[:-2])))
            signs.append(parts[-2])
            b.append(float(parts[-1]))

        return LPTask(c, np.array(A), signs, np.array(b))


class SimplexSolver:
    """
    Реализует двухфазный симплекс-метод.
    """

    def __init__(self, task: LPTask):
        self.task = task

    @staticmethod
    def _pivot(T: np.ndarray, r: int, c: int) -> None:
        T[r] /= T[r, c]
        for i in range(len(T)):
            if i != r:
                T[i] -= T[i, c] * T[r]

    def _simplex(self, T: np.ndarray, basis: List[int]) -> Tuple[np.ndarray, List[int]]:
        while True:
            if np.all(T[-1, :-1] >= -EPS):
                break

            col = int(np.argmin(T[-1, :-1]))

            ratios = [
                T[i, -1] / T[i, col] if T[i, col] > EPS else np.inf
                for i in range(len(T) - 1)
            ]

            if min(ratios) == np.inf:
                raise ValueError("Целевая функция не ограничена")

            row = int(np.argmin(ratios))
            basis[row] = col
            self._pivot(T, row, col)

        return T, basis

    def solve(self) -> Tuple[Dict[str, float], float]:
        c, A, signs, b = self.task.c, self.task.A, self.task.signs, self.task.b
        m, n = A.shape

        names = [f"x{i+1}" for i in range(n)]
        basis: List[int] = []

        # Канонизация
        for i, s in enumerate(signs):
            col = [0] * m
            col[i] = 1

            if s == "<=":
                A = np.column_stack([A, col])
                names.append(f"s{i+1}")
                basis.append(len(names) - 1)

            elif s == ">=":
                A = np.column_stack([A, [-x for x in col]])
                names.append(f"s{i+1}")
                A = np.column_stack([A, col])
                names.append(f"a{i+1}")
                basis.append(len(names) - 1)

            elif s == "=":
                A = np.column_stack([A, col])
                names.append(f"a{i+1}")
                basis.append(len(names) - 1)

        A = np.column_stack([A, b])

        # ---------- Фаза I ----------
        T = A.copy()
        F1 = np.zeros(len(names) + 1)

        for i, nm in enumerate(names):
            if nm.startswith("a"):
                F1[i] = 1

        T = np.vstack([T, F1])

        for i in range(len(basis)):
            T[-1] -= T[i]

        T, basis = self._simplex(T, basis)

        if abs(T[-1, -1]) > EPS:
            raise ValueError("Допустимых решений нет")

        # ---------- Фаза II ----------
        keep = [i for i, nm in enumerate(names) if not nm.startswith("a")]
        T = T[:, keep + [len(names)]]
        names2 = [names[i] for i in keep]

        F2 = [-x for x in c] + [0] * (len(names2) - len(c)) + [0]
        T[-1] = F2

        for i, b in enumerate(basis):
            if b in keep:
                j = keep.index(b)
                T[-1] += T[i] * T[-1, j]

        T, basis = self._simplex(T, basis)

        # ---------- Результат ----------
        solution: Dict[str, float] = dict.fromkeys(names2, 0.0)

        for i, b in enumerate(basis):
            if b in keep:
                solution[names2[keep.index(b)]] = T[i, -1]

        Z = T[-1, -1]
        return solution, Z


task = LPTask.from_file("task.txt")
solver = SimplexSolver(task)

solution, Z = solver.solve()

coords = []
i = 1
while f"x{i}" in solution:
    coords.append(solution[f"x{i}"])
    i += 1

coords_str = ", ".join(str(round(x,4)) for x in coords)

print("Оптимальная точка:")
print(f"({coords_str})")

print("Значение целевой функции:")
print(f"Z({coords_str}) = {round(Z,4)}")


