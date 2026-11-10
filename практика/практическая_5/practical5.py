# Практическая работа (урок 5).
# Переписать обработку большого массива с циклов на NumPy; сравнить время и читаемость.
# Задачи 1–15. Запуск: python practical5.py
#
# Сделано по шаблону compare_simple.py: в каждой задаче два способа
# (цикл Python и NumPy), замер time.perf_counter() 3 раза, берём минимум,
# печатаем время, ускорение и проверку, что результаты совпадают.
# Читаемость оцениваем по SLOC (строки кода функции) и по самому коду.
# Выводы по каждой задаче — комментарии «# Вывод (задача N)» после неё.

import inspect
import math
import sys
import time
import numpy as np


# ── Замер: запускаем 3 раза, берём лучший (минимум) ────────
def measure(fn, *args, repeat=3):
    best = float("inf")
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(*args)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return result, best


# ── Сколько строк кода в функции (без def, пустых и комментариев) ──
def sloc(fn):
    lines = inspect.getsource(fn).splitlines()[1:]
    return sum(1 for s in lines if s.strip() and not s.strip().startswith("#"))


# ── Печать результата, как в шаблоне ───────────────────────
def report(f_loop, f_numpy, t_loop, t_numpy, same):
    print(f"цикл:   {t_loop*1000:10.2f} мс   SLOC = {sloc(f_loop)}")
    print(f"NumPy:  {t_numpy*1000:10.2f} мс   SLOC = {sloc(f_numpy)}")
    print(f"ускорение: {t_loop / t_numpy:.1f}×")
    print(f"результаты совпадают: {same}")


def title(text):
    print("\n" + "=" * 60 + "\n" + text + "\n" + "=" * 60)


# ════════════════════════════════════════════════════════════
# БЛОК 1. ОСНОВНЫЕ ОПЕРАЦИИ
# ════════════════════════════════════════════════════════════

# ── Задача 1. Сумма массива ────────────────────────────────
title("Задача 1. Сумма массива")


def loop_sum(arr):
    total = 0.0
    for x in arr:
        total += x
    return total


def numpy_sum(arr):
    return arr.sum()


print(f"{'n':>10} {'цикл, мс':>12} {'NumPy, мс':>12} {'ускорение':>10}  совпадают")
for n in [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]:
    rng = np.random.default_rng(42)
    a = rng.normal(size=n)
    r_loop,  t_loop  = measure(loop_sum,  a)
    r_numpy, t_numpy = measure(numpy_sum, a)
    print(f"{n:>10} {t_loop*1000:12.3f} {t_numpy*1000:12.4f} {t_loop/t_numpy:9.1f}×  "
          f"{np.isclose(r_loop, r_numpy)}")
print(f"SLOC: цикл {sloc(loop_sum)}, NumPy {sloc(numpy_sum)}")

# Вывод (задача 1). Ускорение растёт с n: при n = 100 всего ≈4× (у вызова NumPy
# есть постоянные накладные расходы ~1 мкс), больше 10× оно становится уже при
# n = 1000, а при n = 10⁵ доходит до ≈300×. При n = 10⁶–10⁷ ускорение падает
# до ≈100×: массив не помещается в кеш процессора, и NumPy упирается в скорость
# памяти, а цикл всегда тратит ≈60 нс на элемент. a.sum() — 1 строка вместо 4.


# ── Задача 2. Среднее и стандартное отклонение ─────────────
title("Задача 2. Среднее и std")
rng = np.random.default_rng(42)
a = rng.normal(size=1_000_000)


def loop_mean_std(arr):
    n = len(arr)
    s = 0.0
    s2 = 0.0
    for x in arr:
        s += x
        s2 += x * x
    mean = s / n
    var = s2 / n - mean * mean
    return mean, math.sqrt(max(var, 0.0))


def numpy_mean_std(arr):
    return arr.mean(), arr.std()


r_loop,  t_loop  = measure(loop_mean_std,  a)
r_numpy, t_numpy = measure(numpy_mean_std, a)
same = abs(r_loop[0] - r_numpy[0]) < 1e-12 and abs(r_loop[1] - r_numpy[1]) < 1e-12
report(loop_mean_std, numpy_mean_std, t_loop, t_numpy, same)
print(f"|Δmean| = {abs(r_loop[0] - r_numpy[0]):.1e}, |Δstd| = {abs(r_loop[1] - r_numpy[1]):.1e}")

# Проверка устойчивости: сдвигаем данные на 1e8 — std не должно измениться.
b = a + 1e8
print(f"std после сдвига на 1e8: настоящее {a.std():.10f}, "
      f"цикл {loop_mean_std(b)[1]:.10f}, NumPy {b.std():.10f}")

# Вывод (задача 2). NumPy быстрее ≈45–65×, результаты совпадают до 1e-14 (лучше
# порога 1e-12). Формула «в лоб» E[x²] − E[x]² неустойчива: после сдвига данных
# на 1e8 она даёт std = 0 вместо 1 (вычитание близких чисел ~1e16), а NumPy
# считает двумя проходами — сначала mean, потом среднее (x − mean)² с попарным
# суммированием — и ошибки не даёт. Читаемость: 1 строка против 9.


# ── Задача 3. y = sin(x) + x² ──────────────────────────────
title("Задача 3. Поэлементное преобразование sin(x) + x²")
rng = np.random.default_rng(42)
a = rng.normal(size=1_000_000)


def loop_sin_sq(arr):
    result = []
    for x in arr:
        result.append(math.sin(x) + x * x)
    return result


def numpy_sin_sq(arr):
    return np.sin(arr) + arr**2


r_loop,  t_loop  = measure(loop_sin_sq,  a)
r_numpy, t_numpy = measure(numpy_sin_sq, a)
report(loop_sin_sq, numpy_sin_sq, t_loop, t_numpy, np.allclose(r_loop, r_numpy))

# Память: список = указатели (8 байт) + объекты float (24 байта каждый).
list_bytes = sys.getsizeof(r_loop) + sum(sys.getsizeof(v) for v in r_loop)
print(f"память цикла (list):   {list_bytes / 1e6:.1f} МБ")
print(f"память NumPy (ответ):  {r_numpy.nbytes / 1e6:.1f} МБ "
      f"+ 2 временных массива (np.sin(a) и a**2) по {a.nbytes / 1e6:.0f} МБ")

# Вывод (задача 3). NumPy быстрее ≈15× (меньше, чем в других задачах: sin сам по
# себе дорогой). Цикл строит список ≈40 МБ (указатель + отдельный объект на каждое
# число), NumPy — один массив 8 МБ, но np.sin(a) + a**2 создаёт 2 временных массива
# (np.sin(a) и a**2), то есть в пике ≈24 МБ. Код NumPy совпадает с формулой.


# ── Задача 4. Clipping ─────────────────────────────────────
title("Задача 4. Условная замена (clipping)")
rng = np.random.default_rng(42)
a = rng.normal(size=1_000_000)


def loop_clip(arr):
    result = []
    for x in arr:
        if x < 0:
            result.append(0.0)
        elif x > 1:
            result.append(1.0)
        else:
            result.append(x)
    return result


def numpy_clip(arr):
    return np.clip(arr, 0, 1)


r_loop,  t_loop  = measure(loop_clip,  a)
r_numpy, t_numpy = measure(numpy_clip, a)
report(loop_clip, numpy_clip, t_loop, t_numpy, np.array_equal(r_loop, r_numpy))
print(f"изменено элементов: {np.sum((a < 0) | (a > 1))} из {a.size}")

# Вывод (задача 4). np.clip быстрее ≈55×, результаты совпадают побитово, изменено
# ≈66% элементов (для N(0,1): P(x<0) + P(x>1) ≈ 0.5 + 0.16). У цикла 1 for и 2 if
# (9 строк), у NumPy 0 циклов и 0 if — 1 строка, и название функции сразу
# говорит, что происходит.


# ════════════════════════════════════════════════════════════
# БЛОК 2. АГРЕГАЦИИ ПО ОСЯМ
# ════════════════════════════════════════════════════════════

# ── Задача 5. Сумма по строкам матрицы ─────────────────────
title("Задача 5. Сумма по строкам матрицы")
rng = np.random.default_rng(42)
m = rng.normal(size=(100_000, 10))


def loop_sum_rows(mat):
    result = [0.0] * mat.shape[0]
    for i in range(mat.shape[0]):
        s = 0.0
        for j in range(mat.shape[1]):
            s += mat[i, j]
        result[i] = s
    return result


def numpy_sum_rows(mat):
    return mat.sum(axis=1)


r_loop,  t_loop  = measure(loop_sum_rows,  m)
r_numpy, t_numpy = measure(numpy_sum_rows, m)
report(loop_sum_rows, numpy_sum_rows, t_loop, t_numpy, np.allclose(r_loop, r_numpy))

# Вывод (задача 5). NumPy быстрее ≈100×, SLOC 7 → 1. m.sum(axis=1) читается как
# сама задача — «сумма по строкам», а цикл — это индексы i, j и счётчик s, где
# легко ошибиться. Поддерживать проще NumPy: для суммы по столбцам достаточно
# поменять axis.


# ── Задача 6. Среднее по столбцам и нормировка ─────────────
title("Задача 6. Вычитание среднего по столбцам")
rng = np.random.default_rng(42)
m = rng.normal(size=(100_000, 10))


def loop_center(mat):
    n, k = mat.shape
    means = [0.0] * k
    for j in range(k):
        for i in range(n):
            means[j] += mat[i, j]
        means[j] /= n
    result = np.empty((n, k))
    for i in range(n):
        for j in range(k):
            result[i, j] = mat[i, j] - means[j]
    return result


def numpy_center(mat):
    return mat - mat.mean(axis=0)


r_loop,  t_loop  = measure(loop_center,  m)
r_numpy, t_numpy = measure(numpy_center, m)
report(loop_center, numpy_center, t_loop, t_numpy, np.allclose(r_loop, r_numpy))
print(f"max |среднее столбца| после нормировки: {np.abs(r_numpy.mean(axis=0)).max():.1e}")
idx_loop = inspect.getsource(loop_center).count("[")
idx_numpy = inspect.getsource(numpy_center).count("[")
print(f"индексаций [..] в коде: цикл {idx_loop}, NumPy {idx_numpy}")

# Вывод (задача 6). NumPy быстрее ≈110–130×, средние после нормировки ≈ 0 (1e-16,
# ошибка округления). Broadcasting сам «растягивает» вектор средних (10,) на
# матрицу (n, 10): 1 строка без индексов против 11 строк и 7 индексаций у цикла.


# ── Задача 7. Максимум и его индекс ────────────────────────
title("Задача 7. Максимум и его индекс")
rng = np.random.default_rng(42)
a = rng.normal(size=1_000_000)


def loop_max(arr):
    max_val = arr[0]
    max_idx = 0
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
            max_idx = i
    return max_val, max_idx


def numpy_max(arr):
    return arr.max(), arr.argmax()


r_loop,  t_loop  = measure(loop_max,  a)
r_numpy, t_numpy = measure(numpy_max, a)
report(loop_max, numpy_max, t_loop, t_numpy,
       r_loop[0] == r_numpy[0] and r_loop[1] == r_numpy[1])

# Теперь данные с NaN.
for pos in [0, 500_000]:
    b = a.copy()
    b[pos] = np.nan
    print(f"NaN на позиции {pos:>6}: цикл → {loop_max(b)[0]:.4f}, "
          f"a.max() → {b.max():.4f}, np.nanmax → {np.nanmax(b):.4f}")

# Вывод (задача 7). NumPy быстрее ≈200–280×, ответ тот же. С NaN цикл ведёт себя
# непредсказуемо: если NaN первый, цикл возвращает nan (все сравнения с NaN
# дают False), а если NaN в середине — молча его пропускает. a.max() честно
# возвращает nan, а np.nanmax даёт правильный ответ: nan-функции — защита от
# пропусков, которые в реальных данных бывают всегда.


# ════════════════════════════════════════════════════════════
# БЛОК 3. ФИЛЬТРАЦИЯ И МАСКИ
# ════════════════════════════════════════════════════════════

# ── Задача 8. Фильтрация по порогу ─────────────────────────
title("Задача 8. Фильтрация по порогу")
rng = np.random.default_rng(42)
a = rng.normal(size=1_000_000)


def loop_filter(arr):
    return [x for x in arr.tolist() if x > 0]


def numpy_filter(arr):
    return arr[arr > 0]


r_loop,  t_loop  = measure(loop_filter,  a)
r_numpy, t_numpy = measure(numpy_filter, a)
report(loop_filter, numpy_filter, t_loop, t_numpy, np.array_equal(r_loop, r_numpy))
list_bytes = sys.getsizeof(r_loop) + sum(sys.getsizeof(v) for v in r_loop)
print(f"list:    {len(r_loop)} элементов, {list_bytes / 1e6:.1f} МБ "
      f"({list_bytes / len(r_loop):.0f} байт на элемент)")
print(f"ndarray: {r_numpy.size} элементов, {r_numpy.nbytes / 1e6:.1f} МБ "
      f"({r_numpy.itemsize} байт на элемент)")
print(f"ndarray компактнее в {list_bytes / r_numpy.nbytes:.1f} раза")

# Вывод (задача 8). NumPy быстрее ≈12×. Список занимает 16 МБ (32 байта на
# элемент: указатель 8 + объект float 24), ndarray — 4 МБ (8 байт), то есть в
# 4 раза компактнее. К тому же результат цикла — список, и для дальнейших
# вычислений его пришлось бы снова превращать в массив.


# ── Задача 9. Замена по маске (in-place) ───────────────────
title("Задача 9. Модуль на месте (in-place)")
rng = np.random.default_rng(42)
a = rng.normal(size=1_000_000)


def loop_abs(arr):
    for i in range(len(arr)):
        if arr[i] < 0:
            arr[i] = -arr[i]
    return arr


def numpy_abs(arr):
    arr[arr < 0] *= -1
    return arr


def numpy_abs_out(arr):
    return np.abs(arr, out=arr)


# Функции меняют массив, поэтому перед каждым прогоном — свежая копия
# (копирование в замер не входит).
def measure_inplace(fn, arr, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        b = arr.copy()
        t0 = time.perf_counter()
        result = fn(b)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return result, b, best


r_loop,  b_loop,  t_loop  = measure_inplace(loop_abs,      a)
r_numpy, b_numpy, t_numpy = measure_inplace(numpy_abs,     a)
r_out,   b_out,   t_out   = measure_inplace(numpy_abs_out, a)
report(loop_abs, numpy_abs, t_loop, t_numpy, np.array_equal(r_loop, r_numpy))
print(f"np.abs(a, out=a): {t_out*1000:.2f} мс, ускорение {t_loop / t_out:.1f}×")
print(f"новый массив не создан: цикл {r_loop is b_loop}, "
      f"a[a<0]*=-1 {r_numpy is b_numpy}, np.abs(out=) {r_out is b_out}")

# Вывод (задача 9). Все варианты работают на месте — возвращается тот же объект,
# новый массив не создаётся. a[a < 0] *= -1 быстрее цикла ≈12×, а np.abs(a, out=a)
# — в сотни раз, потому что не строит маску и временную копию выбранных
# элементов. In-place важен для больших данных: если массив занимает половину
# памяти, на копию её просто не хватит.


# ── Задача 10. Комбинированная маска ───────────────────────
title("Задача 10. Комбинированная маска")
rng = np.random.default_rng(42)
a = rng.normal(size=1_000_000)


def loop_band(arr):
    return [x for x in arr.tolist() if -1 < x < 1]


def numpy_band(arr):
    return arr[(arr > -1) & (arr < 1)]


r_loop,  t_loop  = measure(loop_band,  a)
r_numpy, t_numpy = measure(numpy_band, a)
report(loop_band, numpy_band, t_loop, t_numpy, np.array_equal(r_loop, r_numpy))

# Что будет без скобок:
try:
    a > -1 & a < 1
except Exception as e:
    print(f"a > -1 & a < 1  →  {type(e).__name__}: {e}")
try:
    -1 < a < 1
except Exception as e:
    print(f"-1 < a < 1      →  {type(e).__name__}: {e}")

# Вывод (задача 10). NumPy быстрее ≈15–18×. Без скобок a > -1 & a < 1 не работает:
# & сильнее сравнений, получается a > (-1 & a) < 1 и TypeError, а цепочка
# -1 < a < 1 требует от массива одно True/False → ValueError. Поэтому каждое
# условие берут в скобки и соединяют &; здесь запись цикла -1 < x < 1 понятнее.


# ════════════════════════════════════════════════════════════
# БЛОК 4. РЕАЛЬНЫЕ КЕЙСЫ
# ════════════════════════════════════════════════════════════

# ── Задача 11. Матрица расстояний ──────────────────────────
title("Задача 11. Расстояния между точками")
rng = np.random.default_rng(42)
X = rng.normal(size=(1000, 3))
Y = rng.normal(size=(1000, 3))


def loop_dist(X, Y):
    n, m = X.shape[0], Y.shape[0]
    D = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            D[i, j] = math.sqrt((X[i, 0] - Y[j, 0]) ** 2 + (X[i, 1] - Y[j, 1]) ** 2
                                + (X[i, 2] - Y[j, 2]) ** 2)
    return D


def numpy_dist(X, Y):
    return np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2))


def numpy_dist_matmul(X, Y):
    # ||x - y||² = ||x||² + ||y||² - 2·x·y  →  нет временного массива (n, m, 3)
    d2 = (X**2).sum(axis=1)[:, None] + (Y**2).sum(axis=1)[None, :] - 2 * X @ Y.T
    return np.sqrt(np.maximum(d2, 0))


r_loop,  t_loop  = measure(loop_dist,  X, Y)
r_numpy, t_numpy = measure(numpy_dist, X, Y)
report(loop_dist, numpy_dist, t_loop, t_numpy, np.allclose(r_loop, r_numpy))

tmp = X[:, None, :] - Y[None, :, :]
print(f"временный массив {tmp.shape}: {tmp.nbytes / 1e6:.0f} МБ (ответ — {r_numpy.nbytes / 1e6:.0f} МБ)")
r_mm, t_mm = measure(numpy_dist_matmul, X, Y)
print(f"через X @ Y.T:  {t_mm*1000:.2f} мс, max ошибка {np.abs(r_mm - r_numpy).max():.1e}")
try:
    from scipy.spatial.distance import cdist
    r_cd, t_cd = measure(cdist, X, Y)
    print(f"scipy cdist:    {t_cd*1000:.2f} мс, совпадает: {np.allclose(r_cd, r_numpy)}")
except ImportError:
    print("scipy не установлен — cdist пропущен")

# Вывод (задача 11). Broadcasting быстрее двойного цикла ≈20–30× и пишется в
# 1 строку, но временный массив (1000, 1000, 3) занимает 24 МБ при ответе 8 МБ
# (при n = m = 10⁴ было бы 2.4 ГБ). Меньше памяти и ещё быстрее — формула через
# X @ Y.T (ошибка ~1e-14) и scipy cdist: компромисс «скорость/память» решают
# они, а не broadcasting.


# ── Задача 12. Скользящее среднее ──────────────────────────
title("Задача 12. Скользящее среднее (k = 10)")
rng = np.random.default_rng(42)
a = rng.normal(size=100_000)
k = 10


def loop_moving_avg(arr, k):
    result = []
    for i in range(k - 1, len(arr)):
        s = 0.0
        for j in range(i - k + 1, i + 1):
            s += arr[j]
        result.append(s / k)
    return result


def numpy_moving_avg(arr, k):
    return np.convolve(arr, np.ones(k) / k, mode="valid")


r_loop,  t_loop  = measure(loop_moving_avg,  a, k)
r_numpy, t_numpy = measure(numpy_moving_avg, a, k)
report(loop_moving_avg, numpy_moving_avg, t_loop, t_numpy, np.allclose(r_loop, r_numpy))

# Вывод (задача 12). np.convolve быстрее в сотни раз (≈650×), хотя делает те же
# n·k операций — сложность у обоих O(n·k). Это не алгоритмическое ускорение, а
# константный фактор: операция в C стоит доли наносекунды, в интерпретаторе —
# около 100 нс; на практике этого достаточно. Читаемость: 1 строка против 7.


# ── Задача 13. Кумулятивная сумма ──────────────────────────
title("Задача 13. Кумулятивная сумма")
rng = np.random.default_rng(42)
a = rng.normal(size=100_000)


def loop_cumsum(arr):
    s = 0.0
    result = []
    for x in arr:
        s += x
        result.append(s)
    return result


def numpy_cumsum(arr):
    return np.cumsum(arr, dtype=np.float64)


r_loop,  t_loop  = measure(loop_cumsum,  a)
r_numpy, t_numpy = measure(numpy_cumsum, a)
report(loop_cumsum, numpy_cumsum, t_loop, t_numpy,
       np.allclose(r_loop, r_numpy, rtol=0, atol=1e-9))

exact = math.fsum(a)  # точная сумма для проверки
print(f"max |цикл - np.cumsum| = {np.abs(np.array(r_loop) - r_numpy).max():.1e}")
print(f"ошибка полной суммы: цикл {abs(r_loop[-1] - exact):.1e}, "
      f"np.cumsum {abs(r_numpy[-1] - exact):.1e}, a.sum() {abs(a.sum() - exact):.1e}")

# Вывод (задача 13). np.cumsum быстрее ≈25×, совпадает с циклом не просто до 1e-9,
# а побитово (разница 0): cumsum складывает так же последовательно, поэтому он
# не точнее цикла (ошибка обоих 6.3e-13). Попарное суммирование NumPy есть только
# в a.sum() — ошибка 5.7e-14, в 10 раз меньше; для полной точности — math.fsum.


# ── Задача 14. Группировка по значению ─────────────────────
title("Задача 14. Частоты значений")
rng = np.random.default_rng(42)
ints = rng.integers(0, 100, size=1_000_000)


def loop_counts(ints):
    counts = {}
    for v in ints:
        counts[v] = counts.get(v, 0) + 1
    return counts


def numpy_counts(ints):
    return np.bincount(ints, minlength=100)


r_loop,  t_loop  = measure(loop_counts,  ints)
r_numpy, t_numpy = measure(numpy_counts, ints)
same = all(r_loop.get(v, 0) == r_numpy[v] for v in range(100))
report(loop_counts, numpy_counts, t_loop, t_numpy, same)

# Вывод (задача 14). bincount быстрее словаря ≈100×, частоты совпадают: значение
# само служит индексом в массиве счётчиков, хешировать ничего не нужно. Плата —
# только неотрицательные целые в узком диапазоне; для строк, отрицательных и
# сильно разреженных значений (id до 10⁹) лучше словарь.


# ── Задача 15. Полный кейс: «сырые» измерения ──────────────
title("Задача 15. Пайплайн обработки измерений")
rng = np.random.default_rng(42)
raw = rng.normal(size=(100_000, 3))
raw[rng.integers(0, 100_000, 100), rng.integers(0, 3, 100)] = 1e6
rows = raw.tolist()  # для циклов — обычные списки Python


# Шаг 1. Выбросы: |x - mean| > 5·std
def loop_step1(rows):
    n, k = len(rows), len(rows[0])
    mean = [0.0] * k
    for r in rows:
        for j in range(k):
            mean[j] += r[j] / n
    var = [0.0] * k
    for r in rows:
        for j in range(k):
            var[j] += (r[j] - mean[j]) ** 2 / n
    std = [math.sqrt(v) for v in var]
    mask = []
    for r in rows:
        mask.append([abs(r[j] - mean[j]) > 5 * std[j] for j in range(k)])
    return mask


def numpy_step1(raw):
    return np.abs(raw - raw.mean(axis=0)) > 5 * raw.std(axis=0)


# Шаг 2. Выбросы → NaN → среднее столбца
def loop_step2(rows, mask):
    n, k = len(rows), len(rows[0])
    s = [0.0] * k
    cnt = [0] * k
    for i in range(n):
        for j in range(k):
            if not mask[i][j]:
                s[j] += rows[i][j]
                cnt[j] += 1
    col_mean = [s[j] / cnt[j] for j in range(k)]
    clean = []
    for i in range(n):
        clean.append([col_mean[j] if mask[i][j] else rows[i][j] for j in range(k)])
    return clean


def numpy_step2(raw, mask):
    r = np.where(mask, np.nan, raw)
    return np.where(np.isnan(r), np.nanmean(r, axis=0), r)


# Шаг 3. Нормировка столбцов
def loop_step3(rows):
    n, k = len(rows), len(rows[0])
    mean = [0.0] * k
    for r in rows:
        for j in range(k):
            mean[j] += r[j] / n
    var = [0.0] * k
    for r in rows:
        for j in range(k):
            var[j] += (r[j] - mean[j]) ** 2 / n
    std = [math.sqrt(v) for v in var]
    return [[(r[j] - mean[j]) / std[j] for j in range(k)] for r in rows]


def numpy_step3(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


# Шаг 4. Среднее и std после нормировки
def loop_step4(rows):
    n, k = len(rows), len(rows[0])
    mean = [0.0] * k
    for r in rows:
        for j in range(k):
            mean[j] += r[j] / n
    var = [0.0] * k
    for r in rows:
        for j in range(k):
            var[j] += (r[j] - mean[j]) ** 2 / n
    return mean, [math.sqrt(v) for v in var]


def numpy_step4(z):
    return z.mean(axis=0), z.std(axis=0)


# Шаг 5. Корреляционная матрица 3×3
def loop_step5(rows):
    n, k = len(rows), len(rows[0])
    mean = [sum(r[j] for r in rows) / n for j in range(k)]
    corr = [[0.0] * k for _ in range(k)]
    for p in range(k):
        for q in range(k):
            spq = spp = sqq = 0.0
            for r in rows:
                dp, dq = r[p] - mean[p], r[q] - mean[q]
                spq += dp * dq
                spp += dp * dp
                sqq += dq * dq
            corr[p][q] = spq / math.sqrt(spp * sqq)
    return corr


def numpy_step5(z):
    return np.corrcoef(z.T)


# Каждый шаг получает результат предыдущего шага своей же версии.
m_l, t1l = measure(loop_step1, rows)
m_n, t1n = measure(numpy_step1, raw)
c_l, t2l = measure(loop_step2, rows, m_l)
c_n, t2n = measure(numpy_step2, raw, m_n)
z_l, t3l = measure(loop_step3, c_l)
z_n, t3n = measure(numpy_step3, c_n)
s_l, t4l = measure(loop_step4, z_l)
s_n, t4n = measure(numpy_step4, z_n)
k_l, t5l = measure(loop_step5, z_l)
k_n, t5n = measure(numpy_step5, z_n)

steps = [
    ("1. выбросы",    t1l, t1n, loop_step1, numpy_step1, np.array_equal(m_l, m_n)),
    ("2. замена",     t2l, t2n, loop_step2, numpy_step2, np.allclose(c_l, c_n)),
    ("3. нормировка", t3l, t3n, loop_step3, numpy_step3, np.allclose(z_l, z_n)),
    ("4. mean, std",  t4l, t4n, loop_step4, numpy_step4,
     np.allclose(s_l[0], s_n[0], atol=1e-12) and np.allclose(s_l[1], s_n[1])),
    ("5. корреляция", t5l, t5n, loop_step5, numpy_step5, np.allclose(k_l, k_n)),
]

print(f"{'шаг':<15}{'цикл, мс':>10}{'NumPy, мс':>11}{'ускорение':>11}{'SLOC цикл/NumPy':>17}  совпадает")
for name, tl, tn, fl, fn, ok in steps:
    print(f"{name:<15}{tl*1000:10.1f}{tn*1000:11.2f}{tl/tn:10.0f}×{sloc(fl):>10} / {sloc(fn):<5} {ok}")
TL = sum(s[1] for s in steps)
TN = sum(s[2] for s in steps)
SL = sum(sloc(s[3]) for s in steps)
SN = sum(sloc(s[4]) for s in steps)
print(f"{'ВСЕГО':<15}{TL*1000:10.1f}{TN*1000:11.2f}{TL/TN:10.0f}×{SL:>10} / {SN:<5}")

print(f"\nнайдено выбросов: {m_n.sum()}")
print("после нормировки mean:", np.round(s_n[0], 12), " std:", np.round(s_n[1], 12))
print("корреляционная матрица:\n", np.round(k_n, 4))

# Вывод (задача 15). Весь пайплайн на NumPy быстрее ≈25× (≈440 мс → ≈17 мс) и
# в 10 раз короче (61 строка против 6); найдены все 100 выбросов, после
# нормировки mean ≈ 0 и std = 1, корреляции независимых датчиков ≈ 0.
# Максимальное ускорение (≈40×) на шаге 5: np.corrcoef сводит 9 проходов цикла
# к одному матричному умножению. Самые нечитаемые места цикла — шаги 1–3:
# повторяющийся ручной подсчёт mean/std, двойные индексы mask[i][j] и счётчик
# cnt; в NumPy это .mean(axis=0), .std(axis=0), np.where и np.nanmean.
