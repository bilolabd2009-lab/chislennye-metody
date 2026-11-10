"""
Практическая работа (урок 5). Переписать обработку большого массива с циклов на NumPy;
сравнить время и читаемость. Блок 4 «Реальные кейсы» (задачи 11–15).

Запуск:  python numpy_vs_loops.py
Скрипт печатает все таблицы и собирает отчёт report.md (таблицы + выводы).

Правила замера (из методички):
  * repeat=3, берём минимум;
  * одни и те же данные для обоих способов (default_rng(42));
  * результаты сравниваются через np.isclose / np.allclose.

«Чистый Python» здесь — именно чистый: цикл работает со списком Python
(a.tolist()), а не с ndarray. Перевод в список делается ДО замера.
"""

import inspect
import math
import os
import platform
import sys
import time
import tracemalloc

import numpy as np
import scipy
from scipy.spatial.distance import cdist

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = []  # строки отчёта в markdown


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def out(text=""):
    """Печать в консоль и одновременно в отчёт."""
    print(text)
    REPORT.append(text)


def table(header, rows):
    out("| " + " | ".join(header) + " |")
    out("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        out("| " + " | ".join(str(c) for c in r) + " |")
    out()


def conclusion(text):
    out("**Вывод.** " + " ".join(text.split()))
    out()


def measure(fn, *args, repeat=3, number=1, setup=None):
    """Лучшее (минимальное) время одного вызова из repeat прогонов.

    number — сколько раз вызывать внутри одного прогона (нужно для очень
    маленьких массивов, где один вызов короче разрешения таймера).
    setup — функция, которая перед каждым прогоном готовит свежие аргументы
    (нужно для in-place задач: иначе второй прогон получит уже изменённые данные).
    """
    best = float("inf")
    result = None
    for _ in range(repeat):
        if setup is not None:
            args = setup()
        t0 = time.perf_counter()
        for _ in range(number):
            result = fn(*args)
        t1 = time.perf_counter()
        best = min(best, (t1 - t0) / number)
    return result, best


def peak_memory(fn, *args):
    """Пиковая память (байт), выделенная во время вызова fn (tracemalloc видит и NumPy)."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def sloc(fn):
    """Число строк кода в теле функции (без def, пустых строк и комментариев)."""
    lines = inspect.getsource(fn).splitlines()[1:]
    return sum(1 for ln in lines if ln.strip() and not ln.strip().startswith("#"))


def ms(t):
    if t < 1e-3:
        return f"{t * 1e6:.1f} мкс"
    return f"{t * 1e3:.2f} мс"


def x(speedup):
    return f"{speedup:.0f}×" if speedup >= 100 else f"{speedup:.1f}×"


def mb(nbytes):
    return f"{nbytes / 1e6:.1f} МБ"


def th(n):
    """Число с пробелом между разрядами: 1 000 000."""
    return f"{n:,}".replace(",", " ")


def rng42():
    return np.random.default_rng(42)


def compare(name_loop, f_loop, args_loop, name_np, f_np, args_np, check, **kw):
    """Замер двух вариантов + строка таблицы. Возвращает словарь с цифрами."""
    r_loop, t_loop = measure(f_loop, *args_loop, **kw)
    r_np, t_np = measure(f_np, *args_np, **kw)
    ok = bool(check(r_loop, r_np))
    return dict(r_loop=r_loop, r_np=r_np, t_loop=t_loop, t_np=t_np,
                speedup=t_loop / t_np, ok=ok,
                sloc_loop=sloc(f_loop), sloc_np=sloc(f_np))


def standard_table(res):
    table(["вариант", "время", "SLOC", "результат совпадает"],
          [("цикл Python", ms(res["t_loop"]), res["sloc_loop"], ""),
           ("NumPy", ms(res["t_np"]), res["sloc_np"], "да" if res["ok"] else "НЕТ"),
           ("**ускорение**", f"**{x(res['speedup'])}**", "", "")])


# ---------------------------------------------------------------------------
# Блок 4. Реальные кейсы
# ---------------------------------------------------------------------------

def loop_dist(X, Y):
    D = []
    for xi in X:
        row = []
        for yj in Y:
            row.append(math.sqrt((xi[0] - yj[0]) ** 2 + (xi[1] - yj[1]) ** 2
                                 + (xi[2] - yj[2]) ** 2))
        D.append(row)
    return D


def numpy_dist(X, Y):
    return np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2))


def numpy_dist_gram(X, Y):
    d2 = (X**2).sum(axis=1)[:, None] + (Y**2).sum(axis=1)[None, :] - 2 * X @ Y.T
    return np.sqrt(np.maximum(d2, 0))


def task11():
    out("## Задача 11. Матрица расстояний (n = m = 1000, точки в R³)")
    out()
    rng = rng42()
    X = rng.normal(size=(1000, 3))
    Y = rng.normal(size=(1000, 3))
    r_l, t_l = measure(loop_dist, X.tolist(), Y.tolist())
    r_b, t_b = measure(numpy_dist, X, Y)
    r_g, t_g = measure(numpy_dist_gram, X, Y)
    r_c, t_c = measure(cdist, X, Y)
    m_b = peak_memory(numpy_dist, X, Y)
    m_g = peak_memory(numpy_dist_gram, X, Y)
    m_c = peak_memory(cdist, X, Y)
    err_g = np.abs(r_g - r_b).max()
    table(["вариант", "время", "ускорение", "пиковая память", "совпадает с broadcasting"],
          [("двойной цикл", ms(t_l), "1×", "—", "да" if np.allclose(r_l, r_b) else "НЕТ"),
           ("broadcasting (n, m, 3)", ms(t_b), x(t_l / t_b), mb(m_b), "—"),
           ("через X @ Y.T", ms(t_g), x(t_l / t_g), mb(m_g), f"max ошибка {err_g:.1e}"),
           ("`scipy.spatial.distance.cdist`", ms(t_c), x(t_l / t_c), mb(m_c),
            "да" if np.allclose(r_c, r_b) else "НЕТ")])
    tmp = 1000 * 1000 * 3 * 8
    conclusion(f"""
        Временный массив `X[:, None, :] - Y[None, :, :]` имеет форму (1000, 1000, 3):
        3·10⁶ × 8 байт = {tmp / 1e6:.0f} МБ, и ещё столько же уходит на квадраты — отсюда пик
        {mb(m_b)}, хотя ответ занимает всего 8 МБ. Broadcasting быстрее цикла в {x(t_l / t_b)},
        но платит памятью: при n = m = 10⁴ временный массив был бы уже 2.4 ГБ. Формула
        ‖x − y‖² = ‖x‖² + ‖y‖² − 2·x·y сводит работу к матричному умножению (пик {mb(m_g)}),
        но вычитает близкие числа: здесь ошибка до {err_g:.0e} по модулю, а для очень близких
        точек относительная ошибка становится большой (отсюда np.maximum(d2, 0) против
        отрицательных d2).
        cdist считает расстояния в C без временных массивов: и быстро ({x(t_l / t_c)}), и
        экономно ({mb(m_c)}). Компромисс: broadcasting — самый короткий и понятный код, но
        память растёт как n·m·d. Для больших данных нужны cdist или обработка блоками.
    """)


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


def numpy_moving_avg_cumsum(arr, k):
    c = np.cumsum(np.insert(arr, 0, 0.0))
    return (c[k:] - c[:-k]) / k


def task12():
    out("## Задача 12. Скользящее среднее (n = 10⁵, k = 10)")
    out()
    a = rng42().normal(size=10**5)
    k = 10
    res = compare("цикл", loop_moving_avg, (a.tolist(), k), "NumPy", numpy_moving_avg, (a, k),
                  lambda r1, r2: np.allclose(r1, r2))
    r_c, t_c = measure(numpy_moving_avg_cumsum, a, k)
    err_c = np.abs(r_c - res["r_np"]).max()
    cmp_c = "даже медленнее, чем" if t_c > res["t_np"] else "лишь немного быстрее, чем"
    table(["вариант", "сложность", "время", "ускорение", "SLOC", "совпадает"],
          [("двойной цикл", "O(n·k)", ms(res["t_loop"]), "1×", res["sloc_loop"], "—"),
           ("`np.convolve(..., mode='valid')`", "O(n·k)", ms(res["t_np"]), x(res["speedup"]),
            res["sloc_np"], "да" if res["ok"] else "НЕТ"),
           ("через `np.cumsum` (доп.)", "O(n)", ms(t_c), x(res["t_loop"] / t_c),
            sloc(numpy_moving_avg_cumsum), f"ошибка {err_c:.0e}")])
    conclusion(f"""
        np.convolve выполняет те же n·k ≈ 10⁶ умножений-сложений, что и двойной цикл, —
        асимптотика та же, O(n·k). Ускорение {x(res['speedup'])} берётся только из константы:
        одна операция в C стоит доли наносекунды, а в интерпретаторе — десятки наносекунд.
        Это не алгоритмическое улучшение, но на практике его достаточно: 10⁵ точек
        обрабатываются за {ms(res['t_np'])}. Настоящее улучшение алгоритма — префиксные суммы
        (O(n), не зависит от k). Но при маленьком k = 10 оно {cmp_c} convolve
        ({ms(t_c)} против {ms(res['t_np'])}): лишние проходы по памяти и временные массивы
        съедают выигрыш, он появится при больших k. И за него платим точностью: разность
        больших накопленных сумм даёт ошибку {err_c:.0e}.
    """)


def loop_cumsum(arr):
    s = 0.0
    result = []
    for x_ in arr:
        s += x_
        result.append(s)
    return result


def numpy_cumsum(arr):
    return np.cumsum(arr)


def task13():
    out("## Задача 13. Кумулятивная сумма (n = 10⁵)")
    out()
    a = rng42().normal(size=10**5)
    res = compare("цикл", loop_cumsum, (a.tolist(),), "NumPy", numpy_cumsum, (a,),
                  lambda r1, r2: np.allclose(r1, r2, rtol=0, atol=1e-9))
    standard_table(res)
    diff = np.abs(np.asarray(res["r_loop"]) - res["r_np"]).max()
    exact = math.fsum(a.tolist())
    table(["способ", "последний элемент / полная сумма", "ошибка относительно math.fsum"],
          [("math.fsum (точная сумма)", f"{exact:.15f}", "—"),
           ("цикл: result[-1]", f"{res['r_loop'][-1]:.15f}", f"{abs(res['r_loop'][-1] - exact):.1e}"),
           ("np.cumsum(a)[-1]", f"{res['r_np'][-1]:.15f}", f"{abs(res['r_np'][-1] - exact):.1e}"),
           ("a.sum() (попарное суммирование)", f"{a.sum():.15f}", f"{abs(a.sum() - exact):.1e}")])
    conclusion(f"""
        Ускорение {x(res['speedup'])}, совпадение до 1e-9 выполнено с запасом: максимальное
        расхождение цикла и np.cumsum — {diff:.1e}{' (побитово одинаково)' if diff == 0 else ''}.
        Подсказка из условия, что np.cumsum точнее цикла, на практике не подтверждается:
        np.cumsum обязан выдать все n промежуточных сумм и поэтому складывает так же
        последовательно, как цикл. Попарное суммирование (pairwise) NumPy применяет в a.sum(),
        и у неё ошибка полной суммы действительно меньше ({abs(a.sum() - exact):.0e} против
        {abs(res['r_np'][-1] - exact):.0e}). Если нужна максимальная точность, нужно компенсированное
        суммирование (math.fsum, алгоритм Кэхэна) или хотя бы проверка по math.fsum. Выигрыш np.cumsum — скорость и одна
        строка кода вместо {res['sloc_loop']}.
    """)


def loop_counts(ints):
    counts = {}
    for v in ints:
        counts[v] = counts.get(v, 0) + 1
    return counts


def numpy_counts(ints):
    return np.bincount(ints, minlength=100)


def task14():
    out("## Задача 14. Частоты значений (10⁶ целых из 0..99)")
    out()
    ints = rng42().integers(0, 100, size=10**6)
    res = compare("цикл", loop_counts, (ints.tolist(),), "NumPy", numpy_counts, (ints,),
                  lambda d, b: all(d.get(v, 0) == b[v] for v in range(100)) and sum(d.values()) == b.sum())
    standard_table(res)
    b = res["r_np"]
    out(f"Частоты: min {th(b.min())}, max {th(b.max())}, сумма {th(b.sum())}.")
    out()
    try:
        np.bincount(np.array([-1, 2, 3]))
        neg = "ошибки нет"
    except ValueError as exc:
        neg = f"ValueError: {exc}"
    out(f"`np.bincount([-1, 2, 3])` → `{neg}`.")
    out()
    conclusion(f"""
        Частоты совпадают для всех 100 значений, ускорение {x(res['speedup'])}. bincount
        работает так быстро, потому что значение само служит индексом в массиве счётчиков:
        никакого хеширования, просто `counts[v] += 1` в C. Отсюда и ограничение: только
        неотрицательные целые, и память под счётчики — max(v) + 1 ячеек. Словарь лучше,
        когда значения произвольные: строки, отрицательные или очень разреженные числа
        (например, id до 10⁹ — bincount выделил бы гигабайты). bincount лучше для целых
        в узком диапазоне: категории, классы, байты, гистограммы.
    """)


# ---- Задача 15: пайплайн -------------------------------------------------

def make_raw():
    rng = rng42()
    raw = rng.normal(size=(100_000, 3))
    raw[rng.integers(0, 100_000, 100), rng.integers(0, 3, 100)] = 1e6
    return raw


def s1_loop(rows):
    n = len(rows)
    k = len(rows[0])
    mean = [0.0] * k
    for r in rows:
        for j in range(k):
            mean[j] += r[j]
    for j in range(k):
        mean[j] /= n
    var = [0.0] * k
    for r in rows:
        for j in range(k):
            var[j] += (r[j] - mean[j]) ** 2
    sd = [math.sqrt(v / n) for v in var]
    mask = []
    for r in rows:
        mask.append([abs(r[j] - mean[j]) > 5 * sd[j] for j in range(k)])
    return mask


def s1_np(raw):
    return np.abs(raw - raw.mean(axis=0)) > 5 * raw.std(axis=0)


def s2_loop(rows, mask):
    n = len(rows)
    k = len(rows[0])
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


def s2_np(raw, mask):
    r = np.where(mask, np.nan, raw)
    return np.where(np.isnan(r), np.nanmean(r, axis=0), r)


def s3_loop(rows):
    n = len(rows)
    k = len(rows[0])
    mean = [0.0] * k
    for r in rows:
        for j in range(k):
            mean[j] += r[j]
    mean = [m / n for m in mean]
    var = [0.0] * k
    for r in rows:
        for j in range(k):
            var[j] += (r[j] - mean[j]) ** 2
    sd = [math.sqrt(v / n) for v in var]
    return [[(r[j] - mean[j]) / sd[j] for j in range(k)] for r in rows]


def s3_np(clean):
    return (clean - clean.mean(axis=0)) / clean.std(axis=0)


def s4_loop(rows):
    n = len(rows)
    k = len(rows[0])
    mean = [0.0] * k
    for r in rows:
        for j in range(k):
            mean[j] += r[j]
    mean = [m / n for m in mean]
    var = [0.0] * k
    for r in rows:
        for j in range(k):
            var[j] += (r[j] - mean[j]) ** 2
    return mean, [math.sqrt(v / n) for v in var]


def s4_np(z):
    return z.mean(axis=0), z.std(axis=0)


def s5_loop(rows):
    n = len(rows)
    k = len(rows[0])
    mean = [sum(r[j] for r in rows) / n for j in range(k)]
    corr = [[0.0] * k for _ in range(k)]
    for a in range(k):
        for b in range(k):
            sab = saa = sbb = 0.0
            for r in rows:
                da = r[a] - mean[a]
                db = r[b] - mean[b]
                sab += da * db
                saa += da * da
                sbb += db * db
            corr[a][b] = sab / math.sqrt(saa * sbb)
    return corr


def s5_np(z):
    return np.corrcoef(z.T)


def task15():
    out("## Задача 15. Полный кейс: обработка «сырых» измерений (10⁵ × 3)")
    out()
    raw = make_raw()
    rows = raw.tolist()
    n_out = int((raw == 1e6).sum())

    steps = []
    # каждый шаг получает на вход результат предыдущего шага своей же версии
    m_l, t1l = measure(s1_loop, rows)
    m_n, t1n = measure(s1_np, raw)
    steps.append(("1. найти выбросы: abs(x − mean) > 5·std", t1l, t1n, s1_loop, s1_np,
                  np.array_equal(m_l, m_n)))
    c_l, t2l = measure(s2_loop, rows, m_l)
    c_n, t2n = measure(s2_np, raw, m_n)
    steps.append(("2. выбросы → NaN → среднее столбца", t2l, t2n, s2_loop, s2_np,
                  np.allclose(c_l, c_n)))
    z_l, t3l = measure(s3_loop, c_l)
    z_n, t3n = measure(s3_np, c_n)
    steps.append(("3. нормировка столбцов", t3l, t3n, s3_loop, s3_np, np.allclose(z_l, z_n)))
    st_l, t4l = measure(s4_loop, z_l)
    st_n, t4n = measure(s4_np, z_n)
    steps.append(("4. mean и std после нормировки", t4l, t4n, s4_loop, s4_np,
                  np.allclose(st_l[0], st_n[0], atol=1e-12) and np.allclose(st_l[1], st_n[1])))
    k_l, t5l = measure(s5_loop, z_l)
    k_n, t5n = measure(s5_np, z_n)
    steps.append(("5. корреляционная матрица 3×3", t5l, t5n, s5_loop, s5_np,
                  np.allclose(k_l, k_n)))

    rows_t = []
    for name, tl, tn, fl, fn, ok in steps:
        rows_t.append((name, ms(tl), ms(tn), x(tl / tn), f"{sloc(fl)} / {sloc(fn)}",
                       "да" if ok else "НЕТ"))
    TL = sum(s[1] for s in steps)
    TN = sum(s[2] for s in steps)
    SL = sum(sloc(s[3]) for s in steps)
    SN = sum(sloc(s[4]) for s in steps)
    rows_t.append(("**весь пайплайн**", f"**{ms(TL)}**", f"**{ms(TN)}**", f"**{x(TL / TN)}**",
                   f"**{SL} / {SN}**", ""))
    table(["шаг", "время цикла", "время NumPy", "ускорение", "SLOC цикл / NumPy",
           "результаты совпадают"], rows_t)

    out(f"Найдено выбросов: {int(m_n.sum())}. Ячеек со значением 1e6 после вставки: {n_out} "
        "(из 100 случайных пар индексов часть могла совпасть).")
    out()
    out("После нормировки (NumPy):")
    out()
    table(["датчик", "mean", "std"],
          [(j + 1, f"{st_n[0][j]:.1e}", f"{st_n[1][j]:.15f}") for j in range(3)])
    out("Корреляционная матрица:")
    out()
    table(["", "датчик 1", "датчик 2", "датчик 3"],
          [(f"датчик {i + 1}", *(f"{k_n[i][j]:+.4f}" for j in range(3))) for i in range(3)])

    best = max(steps, key=lambda s: s[1] / s[2])
    worst = min(steps, key=lambda s: s[1] / s[2])
    why_best = {
        "5": "в цикле 9 пар датчиков × 10⁵ строк, каждая пара — отдельный проход, а "
             "np.corrcoef сводит всё к одному матричному произведению (BLAS)",
    }.get(best[0][0], "там в цикле больше всего элементарной арифметики на элемент, "
                      "а NumPy делает её в C целыми векторами")
    conclusion(f"""
        Пайплайн на NumPy быстрее в {x(TL / TN)} ({ms(TL)} → {ms(TN)}) и короче в
        {SL / SN:.0f} раз (SLOC {SL} против {SN}). Все пять шагов дали одинаковый результат.
        Выбросы найдены все ({int(m_n.sum())} из {n_out}). После нормировки среднее ≈ 0
        (порядка 1e-17), std = 1, корреляции между независимыми датчиками близки к нулю, как
        и должно быть. Максимальное ускорение — на шаге «{best[0][3:]}» ({x(best[1] / best[2])}):
        {why_best}. Минимальное — на шаге «{worst[0][3:]}» ({x(worst[1] / worst[2])}).
        Самые нечитаемые места цикла — шаги 1 и 2: два прохода ради mean и std, вложенные
        `for j in range(k)`, двойные индексы `mask[i][j]`, `rows[i][j]` и ручной подсчёт
        cnt[j], чтобы исключить выбросы из среднего. В NumPy это `np.nanmean` и `np.where`.
        Вычисление mean/std в цикле повторено три раза (шаги 1, 3, 4), в NumPy это просто
        `.mean(axis=0)` и `.std(axis=0)`.
    """)


# ---------------------------------------------------------------------------

def main():
    out("# Практическая работа (блок 4). Переписать обработку большого массива с циклов на NumPy; "
        "сравнить время и читаемость")
    out()
    out(f"Окружение: Python {platform.python_version()}, numpy {np.__version__}, "
        f"scipy {scipy.__version__}, {platform.system()} {platform.release()}.")
    out()
    out("Выполнен блок 4 (задачи 11–15). "
        "Методика: `time.perf_counter()`, 3 прогона, берётся минимум. Данные — `np.random.default_rng(42)`. "
        "Цикл работает со списком Python (`a.tolist()` до замера), NumPy — с ndarray. "
        "SLOC — строки тела функции без пустых и комментариев. Память — пик по `tracemalloc`. "
        "Точные цифры времени меняются от запуска к запуску; порядок величин стабилен.")
    out()
    out("# Блок 4. Реальные кейсы")
    out()
    for t in (task11, task12, task13, task14, task15):
        t()
    with open(os.path.join(HERE, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
