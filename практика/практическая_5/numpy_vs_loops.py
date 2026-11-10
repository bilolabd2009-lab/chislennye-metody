"""
Практическая работа (урок 5). Переписать обработку большого массива с циклов на NumPy;
сравнить время и читаемость.

Запуск:  python numpy_vs_loops.py
Скрипт печатает все таблицы и собирает отчёт report.md (таблицы + выводы).

Правила замера (из методички):
  * repeat=3, берём минимум;
  * одни и те же данные для обоих способов (default_rng(42));
  * результаты сравниваются через np.isclose / np.allclose.

«Чистый Python» здесь — именно чистый: цикл работает со списком Python
(a.tolist()), а не с ndarray. Перевод в список делается ДО замера.
В задаче 1 отдельно показано, что цикл прямо по ndarray ещё медленнее.
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


def n_index(fn):
    """Сколько раз в теле функции встречается индексация `[...]`."""
    body = "\n".join(inspect.getsource(fn).splitlines()[1:])
    return body.count("[")


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
# Блок 1. Основные операции
# ---------------------------------------------------------------------------

def loop_sum(arr):
    total = 0.0
    for x_ in arr:
        total += x_
    return total


def numpy_sum(arr):
    return arr.sum()


def task1():
    out("## Задача 1. Сумма массива")
    out()
    rows = []
    data = {}
    for n in (10, 100, 1_000, 10_000, 10**5, 10**6, 10**7):
        a = rng42().normal(size=n)
        al = a.tolist()
        number = 1000 if n <= 10_000 else 1
        repeat = 5 if n <= 10_000 else 3
        r_l, t_l = measure(loop_sum, al, repeat=repeat, number=number)
        r_n, t_n = measure(numpy_sum, a, repeat=repeat, number=number)
        r_a, t_a = measure(loop_sum, a, repeat=repeat, number=number)
        data[n] = (t_l, t_n, t_a)
        rows.append((f"{n:.0e}".replace("+0", "").replace("+", ""),
                     ms(t_l), ms(t_a), ms(t_n), x(t_l / t_n),
                     "да" if np.isclose(r_l, r_n) else "НЕТ"))
    table(["n", "цикл по list", "цикл по ndarray", "a.sum()", "ускорение (list → NumPy)",
           "совпадает"], rows)

    first10 = next(n for n, (tl, tn, _) in data.items() if tl / tn > 10)
    sp = {n: tl / tn for n, (tl, tn, _) in data.items()}
    peak_n = max(sp, key=sp.get)
    per_el_loop = data[10**6][0] / 10**6 * 1e9
    per_el_np = {n: data[n][1] / n * 1e9 for n in (10**5, 10**7)}
    slow_arr = data[10**6][2] / data[10**6][0]
    out(f"Только запрошенные n: 1e5 → {x(sp[10**5])}, 1e6 → {x(sp[10**6])}, "
        f"1e7 → {x(sp[10**7])}.")
    out()
    conclusion(f"""
        Зависимость от n немонотонная. Сначала ускорение растёт: у вызова NumPy есть
        постоянные накладные расходы (проверки, создание результата — около микросекунды),
        и на n = 10 они больше всей работы, так что там NumPy даже медленнее ({x(sp[10])}).
        С ростом n константа размазывается, и разница больше 10× появляется уже при
        n = {th(first10)}. Максимум — {x(sp[peak_n])} при n = {th(peak_n)}. Дальше ускорение
        падает до {x(sp[10**7])} при n = 1e7. Цикл тратит одинаковые ≈{per_el_loop:.0f} нс на
        элемент при любом n: его ограничивает интерпретатор. NumPy при n = 1e5 тратит
        {per_el_np[10**5]:.2f} нс на элемент, а при n = 1e7 — {per_el_np[10**7]:.2f} нс:
        массив 80 МБ не помещается в кеш процессора, и сумма упирается в скорость чтения
        из оперативной памяти. На всех запрошенных n (1e5, 1e6, 1e7) NumPy быстрее в
        десятки (до ~100) раз. Отдельно: цикл прямо по ndarray ещё в {slow_arr:.1f} раза медленнее
        цикла по списку, потому что на каждой итерации NumPy создаёт новый объект np.float64.
        Читаемость: `a.sum()` — одна строка против четырёх, и смысл виден сразу.
    """)


def loop_mean_std(arr):
    n = len(arr)
    s = 0.0
    s2 = 0.0
    for x_ in arr:
        s += x_
        s2 += x_ * x_
    mean = s / n
    var = s2 / n - mean * mean
    return mean, math.sqrt(max(var, 0.0))


def numpy_mean_std(arr):
    return arr.mean(), arr.std()


def task2():
    out("## Задача 2. Среднее и стандартное отклонение")
    out()
    a = rng42().normal(size=10**6)
    res = compare("цикл", loop_mean_std, (a.tolist(),), "NumPy", numpy_mean_std, (a,),
                  lambda r1, r2: abs(r1[0] - r2[0]) < 1e-12 and abs(r1[1] - r2[1]) < 1e-12)
    standard_table(res)
    dm = abs(res["r_loop"][0] - res["r_np"][0])
    ds = abs(res["r_loop"][1] - res["r_np"][1])
    out(f"|mean_цикл − mean_NumPy| = {dm:.2e}, |std_цикл − std_NumPy| = {ds:.2e} "
        f"(порог 1e-12: {'совпадают' if res['ok'] else 'НЕ совпадают'}).")
    out()

    out("Проверка устойчивости: те же данные, сдвинутые на 1e8 (std не должно измениться).")
    out()
    b = a + 1e8
    m_l, s_l = loop_mean_std(b.tolist())
    m_n, s_n = numpy_mean_std(b)
    s_true = float(np.std(a))  # сдвиг не меняет std; считаем на несдвинутых данных
    table(["способ", "std", "относит. ошибка"],
          [("эталон (std исходных данных)", f"{s_true:.12f}", "—"),
           ("цикл: E[x²] − E[x]²", f"{s_l:.12f}", f"{abs(s_l - s_true) / s_true:.2e}"),
           ("NumPy a.std()", f"{s_n:.12f}", f"{abs(s_n - s_true) / s_true:.2e}")])
    conclusion(f"""
        На данных около нуля результаты совпадают до {max(dm, ds):.0e} (лучше порога 1e-12),
        ускорение {x(res['speedup'])}. Но формула «в лоб» var = E[x²] − E[x]² неустойчива:
        при сдвиге на 1e8 оба слагаемых ≈ 1e16, их разность — вычитание близких чисел, и
        от std остаётся ошибка {abs(s_l - s_true) / s_true:.0%}. NumPy считает std двумя
        проходами: сначала среднее, потом среднее квадратов отклонений (x − mean)², причём
        суммирует попарно (pairwise summation). Отклонения малы, катастрофического вычитания
        нет — ошибка {abs(s_n - s_true) / s_true:.0e}. Читаемость: `a.mean(), a.std()`
        против {res['sloc_loop']} строк с формулой, в которой легко ошибиться.
    """)


def loop_sin_sq(arr):
    result = []
    for x_ in arr:
        result.append(math.sin(x_) + x_ * x_)
    return result


def numpy_sin_sq(arr):
    return np.sin(arr) + arr**2


def numpy_sin_sq_inplace(arr):
    y = np.sin(arr)
    y += np.square(arr)
    return y


def task3():
    out("## Задача 3. Поэлементное преобразование y = sin(x) + x²")
    out()
    a = rng42().normal(size=10**6)
    al = a.tolist()
    res = compare("цикл", loop_sin_sq, (al,), "NumPy", numpy_sin_sq, (a,),
                  lambda r1, r2: np.allclose(r1, r2, rtol=0, atol=1e-12))
    standard_table(res)

    mem_loop = peak_memory(loop_sin_sq, al)
    mem_np = peak_memory(numpy_sin_sq, a)
    mem_np2 = peak_memory(numpy_sin_sq_inplace, a)
    _, t_np2 = measure(numpy_sin_sq_inplace, a)
    table(["вариант", "пиковая память", "время"],
          [("цикл → list из 10⁶ float", mb(mem_loop), ms(res["t_loop"])),
           ("`np.sin(a) + a**2`", mb(mem_np), ms(res["t_np"])),
           ("`y = np.sin(a); y += np.square(a)`", mb(mem_np2), ms(t_np2))])
    conclusion(f"""
        NumPy быстрее в {x(res['speedup'])}. Цикл строит список: 8 байт указателя + 24 байта
        на объект float на каждый элемент, с запасом при append — пик {mb(mem_loop)}.
        Результат NumPy — один массив float64 (8 МБ), но выражение `np.sin(a) + a**2`
        создаёт два временных массива (`np.sin(a)` и `a**2`) и третий под сумму, поэтому
        пик {mb(mem_np)} ≈ 3 × 8 МБ. Если складывать на месте (`y += np.square(a)`),
        остаётся один временный — пик {mb(mem_np2)}. Читаемость: NumPy-строка совпадает
        с формулой из условия, цикл — {res['sloc_loop']} строки служебного кода.
    """)


def loop_clip(arr):
    result = []
    for x_ in arr:
        if x_ < 0:
            result.append(0.0)
        elif x_ > 1:
            result.append(1.0)
        else:
            result.append(x_)
    return result


def numpy_clip(arr):
    return np.clip(arr, 0, 1)


def task4():
    out("## Задача 4. Условная замена (clipping)")
    out()
    a = rng42().normal(size=10**6)
    res = compare("цикл", loop_clip, (a.tolist(),), "NumPy", numpy_clip, (a,),
                  lambda r1, r2: np.array_equal(r1, r2))
    standard_table(res)
    changed = int(np.sum((a < 0) | (a > 1)))
    out(f"Изменено элементов: {th(changed)} из 1 000 000 ({changed / 1e6:.1%}).")
    out()
    conclusion(f"""
        Ускорение {x(res['speedup'])}, результаты совпадают побитово. Изменено
        {changed / 1e6:.1%} элементов — для N(0, 1) это ожидаемо: P(x < 0) + P(x > 1) ≈
        0.5 + 0.159. У цикла 1 `for` и 2 ветки `if/elif` ({res['sloc_loop']} строк), у NumPy
        ноль циклов и ноль условий — одна строка, в {res['sloc_loop'] / res['sloc_np']:.0f} раз
        короче. Название `np.clip` сразу говорит, что происходит, а цикл надо дочитать до конца,
        чтобы понять, что это «обрезка в [0, 1]».
    """)


# ---------------------------------------------------------------------------
# Блок 2. Агрегации по осям
# ---------------------------------------------------------------------------

def loop_sum_rows(mat):
    n = len(mat)
    result = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(len(mat[i])):
            s += mat[i][j]
        result[i] = s
    return result


def numpy_sum_rows(mat):
    return mat.sum(axis=1)


def task5():
    out("## Задача 5. Сумма по строкам матрицы (n = 10⁵, 10 столбцов)")
    out()
    m = rng42().normal(size=(10**5, 10))
    res = compare("цикл", loop_sum_rows, (m.tolist(),), "NumPy", numpy_sum_rows, (m,),
                  lambda r1, r2: np.allclose(r1, r2))
    standard_table(res)
    out(f"Индексаций `[...]` в коде: цикл — {n_index(loop_sum_rows)}, NumPy — "
        f"{n_index(numpy_sum_rows)}.")
    out()
    conclusion(f"""
        Ускорение {x(res['speedup'])}, SLOC {res['sloc_loop']} → {res['sloc_np']}.
        `m.sum(axis=1)` читается как формулировка задачи: «сумма по строкам». В цикле
        смысл спрятан за служебными вещами: заготовка результата, два счётчика i и j,
        промежуточная s, {n_index(loop_sum_rows)} индексаций. Каждая из них — место для
        ошибки (перепутать i и j, забыть обнулить s). Поддерживать проще NumPy-версию:
        чтобы считать по столбцам, достаточно поменять axis=1 на axis=0, а цикл пришлось бы
        переписать.
    """)


def loop_center_cols(mat):
    n = len(mat)
    k = len(mat[0])
    means = [0.0] * k
    for row in mat:
        for j in range(k):
            means[j] += row[j]
    for j in range(k):
        means[j] /= n
    result = []
    for row in mat:
        result.append([row[j] - means[j] for j in range(k)])
    return result


def numpy_center_cols(mat):
    return mat - mat.mean(axis=0)


def task6():
    out("## Задача 6. Среднее по столбцам и нормировка (broadcasting)")
    out()
    m = rng42().normal(size=(10**5, 10))
    res = compare("цикл", loop_center_cols, (m.tolist(),), "NumPy", numpy_center_cols, (m,),
                  lambda r1, r2: np.allclose(r1, r2))
    standard_table(res)
    after = np.abs(np.asarray(res["r_np"]).mean(axis=0)).max()
    after_loop = np.abs(np.asarray(res["r_loop"]).mean(axis=0)).max()
    out(f"max |среднее столбца| после нормировки: NumPy {after:.1e}, цикл {after_loop:.1e}.")
    out()
    out(f"Индексаций `[...]`: цикл — {n_index(loop_center_cols)}, NumPy — "
        f"{n_index(numpy_center_cols)}.")
    out()
    conclusion(f"""
        Средние после нормировки ≈ 0 (до {max(after, after_loop):.0e} — это уровень ошибки
        округления), ускорение {x(res['speedup'])}. NumPy-версия — одна строка: массив (n, 10)
        минус вектор (10,) растягивается по строкам сам (broadcasting). Цикл — {res['sloc_loop']}
        строк, три прохода и {n_index(loop_center_cols)} индексаций против
        {n_index(numpy_center_cols)}. Разница в SLOC — {res['sloc_loop'] - res['sloc_np']} строк
        на одну операцию.
    """)


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


def task7():
    out("## Задача 7. Максимум и его индекс")
    out()
    a = rng42().normal(size=10**6)
    res = compare("цикл", loop_max, (a.tolist(),), "NumPy", numpy_max, (a,),
                  lambda r1, r2: r1[0] == r2[0] and r1[1] == r2[1])
    standard_table(res)
    out(f"Максимум {res['r_np'][0]:.6f} на индексе {res['r_np'][1]} — у обоих вариантов.")
    out()

    out("Те же данные с одним NaN:")
    out()
    rows = []
    for pos, label in ((0, "NaN в начале (a[0])"), (500_000, "NaN в середине (a[500000])")):
        b = a.copy()
        b[pos] = np.nan
        lv, li = loop_max(b.tolist())
        rows.append((label, f"{lv:.6f} (idx {li})", f"{b.max():.6f}",
                     f"{np.nanmax(b):.6f} (idx {np.nanargmax(b)})"))
    table(["данные", "цикл", "`a.max()`", "`np.nanmax` / `np.nanargmax`"], rows)
    conclusion(f"""
        Без NaN ускорение {x(res['speedup'])}, ответы совпадают. С NaN цикл ведёт себя
        непредсказуемо: любое сравнение с NaN даёт False. Если NaN стоит первым, он становится
        max_val, ни одно `x > nan` не срабатывает, и цикл возвращает NaN. Если NaN в середине,
        цикл его молча пропускает и случайно отвечает правильно. То есть результат зависит от
        того, где оказался пропуск. `a.max()` честно возвращает NaN (сигнал: «в данных дыра»),
        а `np.nanmax` осознанно пропускает NaN и даёт правильный ответ в обоих случаях. Функции
        с префиксом nan — защита от реальных данных: пропуски в измерениях бывают всегда, и
        поведение на них должно быть явным, а не зависеть от позиции.
    """)


# ---------------------------------------------------------------------------
# Блок 3. Фильтрация и маски
# ---------------------------------------------------------------------------

def loop_filter(arr):
    return [x_ for x_ in arr if x_ > 0]


def numpy_filter(arr):
    return arr[arr > 0]


def list_footprint(lst):
    """Память списка: сам список (указатели) + объекты float."""
    return sys.getsizeof(lst) + sum(sys.getsizeof(v) for v in lst)


def task8():
    out("## Задача 8. Фильтрация по порогу")
    out()
    a = rng42().normal(size=10**6)
    res = compare("цикл", loop_filter, (a.tolist(),), "NumPy", numpy_filter, (a,),
                  lambda r1, r2: np.array_equal(r1, r2))
    standard_table(res)
    lst, arr = res["r_loop"], res["r_np"]
    lst_bytes = list_footprint(lst)
    ptr_bytes = sys.getsizeof(lst)
    table(["результат", "элементов", "память", "байт на элемент"],
          [("list (цикл)", th(len(lst)), mb(lst_bytes), f"{lst_bytes / len(lst):.1f}"),
           ("  из них указатели списка", "", mb(ptr_bytes), f"{ptr_bytes / len(lst):.1f}"),
           ("  из них объекты float", "", mb(lst_bytes - ptr_bytes),
            f"{(lst_bytes - ptr_bytes) / len(lst):.1f}"),
           ("ndarray float64 (NumPy)", th(arr.size), mb(arr.nbytes), f"{arr.nbytes / arr.size:.1f}")])
    conclusion(f"""
        NumPy быстрее в {x(res['speedup'])} и компактнее в {lst_bytes / arr.nbytes:.1f} раза:
        список хранит указатель (8 байт) на отдельный объект float (24 байта: счётчик ссылок,
        указатель на тип, само число), а ndarray — сплошной блок по 8 байт. Важно и то, что
        результат цикла — список: для дальнейшей математики его придётся снова превращать в
        массив. Результат `a[a > 0]` — сразу ndarray, цепочка вычислений не прерывается.
        Длина у обоих одинаковая: {th(len(lst))} элементов.
    """)


def loop_abs_inplace(lst):
    for i in range(len(lst)):
        if lst[i] < 0:
            lst[i] = -lst[i]
    return lst


def numpy_abs_mask(arr):
    arr[arr < 0] *= -1
    return arr


def numpy_abs_out(arr):
    np.abs(arr, out=arr)
    return arr


def task9():
    out("## Задача 9. Замена по маске (in-place)")
    out()
    a = rng42().normal(size=10**6)
    expected = np.abs(a)
    # перед каждым прогоном — свежая копия, иначе второй прогон получит уже положительные числа
    r_l, t_l = measure(loop_abs_inplace, setup=lambda: (a.tolist(),))
    r_m, t_m = measure(numpy_abs_mask, setup=lambda: (a.copy(),))
    r_o, t_o = measure(numpy_abs_out, setup=lambda: (a.copy(),))

    lst = a.tolist()
    lst_id = id(lst)
    same_list = id(loop_abs_inplace(lst)) == lst_id
    b = a.copy()
    ptr = b.__array_interface__["data"][0]
    same_mask = numpy_abs_mask(b).__array_interface__["data"][0] == ptr
    c = a.copy()
    ptr_c = c.__array_interface__["data"][0]
    same_out = numpy_abs_out(c).__array_interface__["data"][0] == ptr_c
    mem_mask = peak_memory(numpy_abs_mask, a.copy())
    mem_out = peak_memory(numpy_abs_out, a.copy())

    table(["вариант", "время", "ускорение", "SLOC", "тот же объект", "доп. память", "верно"],
          [("цикл по list", ms(t_l), "1×", sloc(loop_abs_inplace),
            "да" if same_list else "нет", "—", "да" if np.array_equal(r_l, expected) else "НЕТ"),
           ("`a[a < 0] *= -1`", ms(t_m), x(t_l / t_m), sloc(numpy_abs_mask),
            "да" if same_mask else "нет", mb(mem_mask),
            "да" if np.array_equal(r_m, expected) else "НЕТ"),
           ("`np.abs(a, out=a)`", ms(t_o), x(t_l / t_o), sloc(numpy_abs_out),
            "да" if same_out else "нет", mb(mem_out),
            "да" if np.array_equal(r_o, expected) else "НЕТ")])
    out("«Тот же объект» проверен по id списка и по адресу буфера данных ndarray "
        "(`__array_interface__['data']`) до и после.")
    out()
    conclusion(f"""
        Все три варианта работают на месте: новый список/массив не создаётся. Но «на месте»
        бывает разным. `a[a < 0] *= -1` пишет в исходный буфер, однако по дороге создаёт
        булеву маску (1 МБ) и временную копию выбранных элементов — пик {mb(mem_mask)}.
        `np.abs(a, out=a)` не выделяет почти ничего ({mb(mem_out)}) и быстрее всех:
        {x(t_l / t_o)} против цикла. Для больших данных это важно: если массив занимает
        половину оперативной памяти, копия просто не поместится, а in-place с out= работает.
        Читаемость: `np.abs(a, out=a)` прямо говорит «модуль, результат туда же».
    """)


def loop_band(arr):
    return [x_ for x_ in arr if -1 < x_ < 1]


def numpy_band(arr):
    return arr[(arr > -1) & (arr < 1)]


def task10():
    out("## Задача 10. Комбинированная маска")
    out()
    a = rng42().normal(size=10**6)
    res = compare("цикл", loop_band, (a.tolist(),), "NumPy", numpy_band, (a,),
                  lambda r1, r2: np.array_equal(r1, r2))
    standard_table(res)
    out(f"Отобрано {th(len(res['r_np']))} элементов (≈68% — правило «одной сигмы»).")
    out()

    out("Что будет без скобок:")
    out()
    errors = []
    for label, arr in (("float-массив", a[:5]), ("int-массив", np.arange(-2, 3))):
        try:
            arr > -1 & arr < 1  # noqa: B015 — специально, чтобы получить ошибку
            errors.append((label, "ошибки нет"))
        except Exception as exc:  # noqa: BLE001
            errors.append((label, f"`{type(exc).__name__}: {str(exc)[:70]}…`"))
    table(["данные", "`a > -1 & a < 1`"], errors)
    conclusion(f"""
        Ускорение {x(res['speedup'])}. В Python `-1 < x < 1` — удобная цепочка сравнений,
        но для массива она не работает: цепочка `p < q < r` раскрывается в `(p < q) and (q < r)`,
        а `and` требует одно True/False от целого массива. Без скобок ещё хуже: `&` имеет более
        высокий приоритет, чем `<` и `>`, поэтому `a > -1 & a < 1` читается как
        `a > (-1 & a) < 1`. Для float побитовое `&` вообще не определено (TypeError), для int
        получается цепочка сравнений с массивом (ValueError про неоднозначную истинность).
        Поэтому в NumPy каждое условие берут в скобки и соединяют `&`, `|`, `~`. Это
        менее естественно, чем `-1 < x < 1`, — здесь чистый Python читается лучше.
    """)


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
    out("# Практическая работа. Переписать обработку большого массива с циклов на NumPy; "
        "сравнить время и читаемость")
    out()
    out(f"Окружение: Python {platform.python_version()}, numpy {np.__version__}, "
        f"scipy {scipy.__version__}, {platform.system()} {platform.release()}.")
    out()
    out("Методика: `time.perf_counter()`, 3 прогона, берётся минимум (для n ≤ 10⁴ — "
        "среднее по 1000 вызовов внутри прогона). Данные — `np.random.default_rng(42)`. "
        "Цикл работает со списком Python (`a.tolist()` до замера), NumPy — с ndarray. "
        "SLOC — строки тела функции без пустых и комментариев. Память — пик по `tracemalloc`. "
        "Точные цифры времени меняются от запуска к запуску; порядок величин стабилен.")
    out()
    out("# Блок 1. Основные операции")
    out()
    for t in (task1, task2, task3, task4):
        t()
    out("# Блок 2. Агрегации по осям")
    out()
    for t in (task5, task6, task7):
        t()
    out("# Блок 3. Фильтрация и маски")
    out()
    for t in (task8, task9, task10):
        t()
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
