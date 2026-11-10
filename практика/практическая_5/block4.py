# Практическая работа (урок 5), блок 4 «Реальные кейсы», задачи 11–15.
# Запуск: python block4.py
# Сделано по шаблону compare_simple.py: в каждой задаче два способа
# (цикл и NumPy), замер time.perf_counter() 3 раза, берём минимум,
# печатаем время, ускорение и проверку совпадения результатов.

import inspect
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


# ── Сколько строк кода в функции (для оценки читаемости) ───
def sloc(fn):
    lines = inspect.getsource(fn).splitlines()[1:]
    return sum(1 for s in lines if s.strip() and not s.strip().startswith("#"))


def report(t_loop, t_numpy, same):
    print(f"цикл:   {t_loop*1000:10.2f} мс")
    print(f"NumPy:  {t_numpy*1000:10.2f} мс")
    print(f"ускорение: {t_loop / t_numpy:.1f}×")
    print(f"результаты совпадают: {same}")


# ════════════════════════════════════════════════════════════
# Задача 11. Матрица расстояний D[i, j] = ||X[i] - Y[j]||
# ════════════════════════════════════════════════════════════
print("\n=== Задача 11. Расстояния между точками ===")
rng = np.random.default_rng(42)
X = rng.normal(size=(1000, 3))
Y = rng.normal(size=(1000, 3))


def loop_dist(X, Y):
    n, m = X.shape[0], Y.shape[0]
    D = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(3):
                s += (X[i, k] - Y[j, k]) ** 2
            D[i, j] = s ** 0.5
    return D


def numpy_dist(X, Y):
    return np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2))


def numpy_dist_gram(X, Y):
    # ||x - y||² = ||x||² + ||y||² - 2·x·y  →  нет массива (n, m, 3)
    d2 = (X**2).sum(axis=1)[:, None] + (Y**2).sum(axis=1)[None, :] - 2 * X @ Y.T
    return np.sqrt(np.maximum(d2, 0))


r_loop,  t_loop  = measure(loop_dist,  X, Y)
r_numpy, t_numpy = measure(numpy_dist, X, Y)
report(t_loop, t_numpy, np.allclose(r_loop, r_numpy))

tmp = X[:, None, :] - Y[None, :, :]
print(f"временный массив {tmp.shape}: {tmp.nbytes / 1e6:.0f} МБ, ответ {r_numpy.nbytes / 1e6:.0f} МБ")
r_gram, t_gram = measure(numpy_dist_gram, X, Y)
print(f"через X @ Y.T: {t_gram*1000:.2f} мс, max ошибка {np.abs(r_gram - r_numpy).max():.1e}")
try:
    from scipy.spatial.distance import cdist
    r_cd, t_cd = measure(cdist, X, Y)
    print(f"scipy cdist:   {t_cd*1000:.2f} мс, совпадает: {np.allclose(r_cd, r_numpy)}")
except ImportError:
    print("scipy не установлен — cdist пропущен")
print(f"SLOC: цикл {sloc(loop_dist)}, NumPy {sloc(numpy_dist)}")

# Вывод (задача 11). NumPy через broadcasting быстрее цикла примерно в 40 раз
# (≈870 мс → ≈22 мс), а код занимает 1 строку вместо 9. Платим памятью:
# X[:, None, :] - Y[None, :, :] создаёт временный массив (1000, 1000, 3) = 24 МБ,
# хотя ответ весит 8 МБ; при n = m = 10⁴ это было бы уже 2.4 ГБ.
# Меньше памяти берут формула через X @ Y.T (нет массива n×m×3, ≈6 мс, но
# вычитание близких чисел даёт ошибку ~1e-14) и scipy cdist (≈2.5 мс, без
# временных массивов). Компромисс: broadcasting самый понятный, cdist самый
# экономный.


# ════════════════════════════════════════════════════════════
# Задача 12. Скользящее среднее, окно k = 10
# ════════════════════════════════════════════════════════════
print("\n=== Задача 12. Скользящее среднее ===")
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
report(t_loop, t_numpy, np.allclose(r_loop, r_numpy))
print(f"SLOC: цикл {sloc(loop_moving_avg)}, NumPy {sloc(numpy_moving_avg)}")

# Вывод (задача 12). np.convolve быстрее в сотни раз (≈100 мс → ≈0.2 мс),
# хотя делает те же n·k ≈ 10⁶ умножений-сложений: асимптотика у обоих O(n·k).
# Это не алгоритмическое ускорение, а константный фактор: одна операция в C
# стоит доли наносекунды, в интерпретаторе — около 100 нс. На практике этого
# достаточно. Читаемость: 1 строка против 7 с двумя вложенными циклами.


# ════════════════════════════════════════════════════════════
# Задача 13. Кумулятивная сумма
# ════════════════════════════════════════════════════════════
print("\n=== Задача 13. Кумулятивная сумма ===")
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
    return np.cumsum(arr)


r_loop,  t_loop  = measure(loop_cumsum,  a)
r_numpy, t_numpy = measure(numpy_cumsum, a)
report(t_loop, t_numpy, np.allclose(r_loop, r_numpy, rtol=0, atol=1e-9))

import math
exact = math.fsum(a)
print(f"max |цикл - np.cumsum| = {np.abs(np.array(r_loop) - r_numpy).max():.1e}")
print(f"ошибка последнего элемента: цикл {abs(r_loop[-1] - exact):.1e}, "
      f"np.cumsum {abs(r_numpy[-1] - exact):.1e}, a.sum() {abs(a.sum() - exact):.1e}")
print(f"SLOC: цикл {sloc(loop_cumsum)}, NumPy {sloc(numpy_cumsum)}")

# Вывод (задача 13). np.cumsum быстрее примерно в 25 раз, результаты совпадают
# не только до 1e-9, а побитово (разница 0). Подсказка, что np.cumsum точнее
# цикла, не подтвердилась: cumsum обязан выдать все промежуточные суммы и
# складывает так же последовательно, как цикл (ошибка обоих 6.3e-13).
# Попарное суммирование NumPy использует в a.sum() — там ошибка в 10 раз меньше
# (5.7e-14). Для максимальной точности нужен math.fsum.


# ════════════════════════════════════════════════════════════
# Задача 14. Частота каждого значения 0..99
# ════════════════════════════════════════════════════════════
print("\n=== Задача 14. Группировка по значению ===")
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
report(t_loop, t_numpy, same)
print(f"SLOC: цикл {sloc(loop_counts)}, NumPy {sloc(numpy_counts)}")

# Вывод (задача 14). np.bincount быстрее словаря примерно в 100 раз, частоты
# совпадают. Значение само служит индексом в массиве счётчиков, хешировать
# ничего не нужно. Ограничение: только неотрицательные целые, и память —
# max(v) + 1 ячеек. Словарь лучше для строк, отрицательных или сильно
# разреженных значений (id до 10⁹); bincount — для целых в узком диапазоне
# (категории, классы, гистограммы).


# ════════════════════════════════════════════════════════════
# Задача 15. Полный кейс: обработка «сырых» измерений
# ════════════════════════════════════════════════════════════
print("\n=== Задача 15. Пайплайн обработки измерений ===")
rng = np.random.default_rng(42)
raw = rng.normal(size=(100_000, 3))
raw[rng.integers(0, 100_000, 100), rng.integers(0, 3, 100)] = 1e6
n, d = raw.shape


# --- Шаг 1. Найти выбросы: |x - mean| > 5·std ---------------
def loop_step1(raw):
    mask = np.zeros((n, d), dtype=bool)
    for j in range(d):
        s = 0.0
        for i in range(n):
            s += raw[i, j]
        mean = s / n
        s2 = 0.0
        for i in range(n):
            s2 += (raw[i, j] - mean) ** 2
        std = (s2 / n) ** 0.5
        for i in range(n):
            mask[i, j] = abs(raw[i, j] - mean) > 5 * std
    return mask


def numpy_step1(raw):
    return np.abs(raw - raw.mean(axis=0)) > 5 * raw.std(axis=0)


# --- Шаг 2. Выбросы → NaN → среднее столбца -----------------
def loop_step2(raw, mask):
    clean = raw.copy()
    for j in range(d):
        s = 0.0
        cnt = 0
        for i in range(n):
            if not mask[i, j]:
                s += raw[i, j]
                cnt += 1
        col_mean = s / cnt
        for i in range(n):
            if mask[i, j]:
                clean[i, j] = col_mean
    return clean


def numpy_step2(raw, mask):
    r = np.where(mask, np.nan, raw)
    return np.where(np.isnan(r), np.nanmean(r, axis=0), r)


# --- Шаг 3. Нормировка столбцов -----------------------------
def loop_step3(x):
    z = np.empty((n, d))
    for j in range(d):
        s = 0.0
        for i in range(n):
            s += x[i, j]
        mean = s / n
        s2 = 0.0
        for i in range(n):
            s2 += (x[i, j] - mean) ** 2
        std = (s2 / n) ** 0.5
        for i in range(n):
            z[i, j] = (x[i, j] - mean) / std
    return z


def numpy_step3(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


# --- Шаг 4. Среднее и std после нормировки ------------------
def loop_step4(z):
    means, stds = [], []
    for j in range(d):
        s = 0.0
        for i in range(n):
            s += z[i, j]
        mean = s / n
        s2 = 0.0
        for i in range(n):
            s2 += (z[i, j] - mean) ** 2
        means.append(mean)
        stds.append((s2 / n) ** 0.5)
    return means, stds


def numpy_step4(z):
    return z.mean(axis=0), z.std(axis=0)


# --- Шаг 5. Корреляционная матрица 3×3 ----------------------
def loop_step5(z):
    # z уже нормирован (mean = 0), поэтому отклонения от среднего = сами z
    corr = np.zeros((d, d))
    for p in range(d):
        for q in range(d):
            spq = spp = sqq = 0.0
            for i in range(n):
                spq += z[i, p] * z[i, q]
                spp += z[i, p] ** 2
                sqq += z[i, q] ** 2
            corr[p, q] = spq / (spp * sqq) ** 0.5
    return corr


def numpy_step5(z):
    return np.corrcoef(z.T)


# Каждый шаг получает результат предыдущего шага своей же версии.
m_l, t1l = measure(loop_step1, raw)
m_n, t1n = measure(numpy_step1, raw)
c_l, t2l = measure(loop_step2, raw, m_l)
c_n, t2n = measure(numpy_step2, raw, m_n)
z_l, t3l = measure(loop_step3, c_l)
z_n, t3n = measure(numpy_step3, c_n)
s_l, t4l = measure(loop_step4, z_l)
s_n, t4n = measure(numpy_step4, z_n)
k_l, t5l = measure(loop_step5, z_l)
k_n, t5n = measure(numpy_step5, z_n)

steps = [
    ("1. выбросы",      t1l, t1n, loop_step1, numpy_step1, np.array_equal(m_l, m_n)),
    ("2. замена",       t2l, t2n, loop_step2, numpy_step2, np.allclose(c_l, c_n)),
    ("3. нормировка",   t3l, t3n, loop_step3, numpy_step3, np.allclose(z_l, z_n)),
    ("4. mean и std",   t4l, t4n, loop_step4, numpy_step4,
     np.allclose(s_l[0], s_n[0], atol=1e-12) and np.allclose(s_l[1], s_n[1])),
    ("5. корреляция",   t5l, t5n, loop_step5, numpy_step5, np.allclose(k_l, k_n)),
]

print(f"{'шаг':<16}{'цикл, мс':>12}{'NumPy, мс':>12}{'ускорение':>12}{'SLOC':>10}  совпадает")
for name, tl, tn, fl, fn, ok in steps:
    print(f"{name:<16}{tl*1000:12.1f}{tn*1000:12.2f}{tl/tn:11.0f}×"
          f"{sloc(fl):>5} / {sloc(fn):<3} {ok}")
TL = sum(s[1] for s in steps)
TN = sum(s[2] for s in steps)
SL = sum(sloc(s[3]) for s in steps)
SN = sum(sloc(s[4]) for s in steps)
print(f"{'ВСЕГО':<16}{TL*1000:12.1f}{TN*1000:12.2f}{TL/TN:11.0f}×{SL:>5} / {SN:<3}")

print(f"\nнайдено выбросов: {m_n.sum()}")
print("после нормировки mean:", np.round(s_n[0], 12), " std:", np.round(s_n[1], 12))
print("корреляционная матрица:\n", np.round(k_n, 4))

# Вывод (задача 15). Весь пайплайн на NumPy быстрее примерно в 50 раз
# (≈900 мс → ≈17 мс) и короче в 10 раз (61 строка против 6). Все шаги дали
# одинаковый результат: найдены все 100 выбросов, после нормировки mean ≈ 0 и
# std = 1, корреляции между независимыми датчиками близки к 0.
# Максимальное ускорение (≈240×) на шаге 5: цикл проходит 10⁵ строк для каждой
# из 9 пар датчиков, а np.corrcoef сводит всё к одному матричному умножению.
# Минимальное (≈20–30×) — на шагах 2 и 4: в NumPy-версии там несколько отдельных
# проходов (where, isnan, nanmean / mean, std), в цикле работы меньше всего.
# Самые нечитаемые места цикла — шаги 1–3: двойные индексы raw[i, j],
# ручной подсчёт mean и std, отдельный счётчик cnt, чтобы исключить выбросы
# из среднего. Расчёт mean/std в цикле повторён трижды, в NumPy это
# .mean(axis=0) и .std(axis=0).
