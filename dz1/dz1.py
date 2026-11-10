"""
Домашнее задание 1. Устойчивость алгоритма и корректная постановка эксперимента.

Запуск:  python dz1.py
Скрипт печатает все таблицы, сохраняет графики в папку plots/
и собирает отчёт report.md (таблицы + выводы).

Эталон везде — mpmath с 50 значащими цифрами.
"""

import math
import os
import platform
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

mp.mp.dps = 50
HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")
os.makedirs(PLOTS, exist_ok=True)

REPORT = []  # строки отчёта в markdown


def out(text=""):
    """Печать в консоль и одновременно в отчёт."""
    print(text)
    REPORT.append(text)


def table(header, rows):
    """Markdown-таблица."""
    out("| " + " | ".join(header) + " |")
    out("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        out("| " + " | ".join(str(c) for c in r) + " |")
    out()


def rel_err(value, exact):
    """Относительная погрешность float-значения относительно mpmath-эталона."""
    exact = mp.mpf(exact)
    if exact == 0:
        return float(abs(mp.mpf(value)))
    return float(abs((mp.mpf(value) - exact) / exact))


def e(x):
    return f"{x:.2e}"


def save(fig, name):
    path = os.path.join(PLOTS, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    out(f"![{name}](plots/{name})")
    out()


def conclusion(text):
    out("**Вывод.** " + text.strip())
    out()


# ---------------------------------------------------------------------------
# Блок 1. Обусловленность задачи
# ---------------------------------------------------------------------------

def task1():
    out("## Задача 1. Число обусловленности вручную")
    out()
    out("cond = |x · f'(x) / f(x)|")
    out()
    x1, x2, x3, x4 = mp.mpf("1e-8"), mp.mpf("1.001"), mp.mpf("1.000001"), mp.pi / 2
    rows = [
        ("√x", "x = 1e-8", "x·(1/(2√x))/√x = 1/2", float(mp.mpf(1) / 2)),
        ("ln x", "x = 1.001", "x·(1/x)/ln x = 1/ln x", float(1 / mp.log(x2))),
        ("x − 1", "x = 1.000001", "x/(x − 1)", float(x3 / (x3 - 1))),
        ("sin x", "x = π/2", "x·cos x / sin x = 0", float(abs(x4 * mp.cos(x4) / mp.sin(x4)))),
    ]
    table(["f(x)", "точка", "формула cond", "cond"],
          [(f, p, frm, "0" if c < 1e-30 else f"{c:.4g}") for f, p, frm, c in rows])
    conclusion(
        "По возрастанию: sin x в π/2 (cond = 0) < √x (0.5) < ln x в 1.001 (≈1000) < "
        "x − 1 в 1.000001 (≈10⁶). Самая «опасная» — x − 1 около единицы: это вычитание "
        "близких чисел, относительная ошибка входа усиливается в миллион раз. ln x около 1 "
        "опасен по той же причине (ln x ≈ x − 1). У √x и sin x в π/2 ошибка входа не "
        "растёт (у синуса в максимуме даже гасится), это хорошо обусловленные задачи."
    )


def task2():
    out("## Задача 2. Вычитание близких чисел")
    out()
    out("f(a, b) = a − b. cond по a: |a · ∂f/∂a / f| = |a / (a − b)|, "
        "по b: |b / (a − b)|. При a ≈ b знаменатель → 0, cond → ∞.")
    out()
    a, b, da = 1 + 1e-8, 1.0, 1e-9
    f0 = a - b
    f1 = (a + da) - b
    cond_a = abs(a / (a - b))
    rel_in = da / a
    rel_out_pred = cond_a * rel_in
    rel_out_fact = abs(f1 - f0) / abs(f0)
    # то же точно, без округлений
    A, B, DA = mp.mpf(1) + mp.mpf("1e-8"), mp.mpf(1), mp.mpf("1e-9")
    rel_out_exact = float(DA / (A - B))
    table(["величина", "значение"], [
        ("a − b (float)", f"{f0:.17g}"),
        ("Δf = f(a+δa,b) − f(a,b) (float)", f"{f1 - f0:.6e}"),
        ("Δf / δa (float)", f"{(f1 - f0) / da:.10f}"),
        ("cond по a = a/(a−b)", e(cond_a)),
        ("относит. изменение входа δa/a", e(rel_in)),
        ("предсказание: cond · δa/a", e(rel_out_pred)),
        ("факт (float): |Δf|/|f|", e(rel_out_fact)),
        ("факт (mpmath, точно)", e(rel_out_exact)),
        ("ошибка самого a − b в float (отн.)", e(rel_err(f0, A - B))),
    ])
    conclusion(
        "Изменение входа на 10⁻⁹ (относительно) дало изменение результата на 10⁻¹ — ровно "
        "в cond ≈ 10⁸ раз, предсказание совпадает с фактом. Δf/δa ≈ 1, т.е. абсолютная "
        "чувствительность нормальная, а плохой делает задачу деление на маленький результат "
        "a − b. Заодно видно, что уже округление a = 1 + 10⁻⁸ до double (ошибка ~10⁻¹⁶) "
        "даёт ошибку разности ~10⁻⁸ — тот же множитель 10⁸."
    )


def task3():
    out("## Задача 3. Кратный корень")
    out()
    out("x² − 2x + (1 − ε) = 0, точные корни 1 ± √ε.")
    out()
    rows = []
    prev = None
    for k in (2, 4, 6):
        eps = 10.0 ** -k
        a, b, c = 1.0, -2.0, 1.0 - eps
        d = math.sqrt(b * b - 4 * a * c)
        x1 = (-b + d) / (2 * a)
        shift = abs(x1 - 1)
        ratio = "—" if prev is None else f"{prev / shift:.4g}"
        rows.append((f"1e-{k}", f"{x1:.12f}", e(shift), f"{shift / eps:.4g}", ratio))
        prev = shift
    table(["ε", "корень x₁", "сдвиг |x₁ − 1|", "сдвиг / ε", "отношение сдвигов (пред./тек.)"], rows)
    conclusion(
        "Сдвиг корня равен √ε: при уменьшении возмущения в 100 раз корень сдвигается "
        "всего в 10 раз меньше. Отношение «сдвиг / ε» = 1/√ε растёт неограниченно "
        "(10, 100, 1000), т.е. у двукратного корня число обусловленности бесконечно. "
        "Для корня кратности m сдвиг ~ ε^(1/m): чем выше кратность, тем хуже обусловленность. "
        "Это свойство задачи, а не алгоритма — точный корень сдвигается так же."
    )


def task4():
    out("## Задача 4. Обусловленность vs устойчивость")
    out()
    out("Пример: f(x) = sinh x при малых x. cond = |x · cosh x / sinh x| ≈ 1 при x → 0 "
        "— задача обусловлена отлично.")
    out()
    out("- неустойчивый алгоритм: (eˣ − e⁻ˣ)/2 — вычитание двух близких к 1 чисел;")
    out("- устойчивый: (expm1(x) − expm1(−x))/2 — expm1 точно считает eˣ − 1, "
        "а слагаемые разного знака, сокращения нет.")
    out()
    xs = [10.0 ** -k for k in range(1, 16, 2)]
    rows, eb, eg = [], [], []
    for x in xs:
        bad = (math.exp(x) - math.exp(-x)) / 2
        good = (math.expm1(x) - math.expm1(-x)) / 2
        ref = mp.sinh(mp.mpf(x))
        cond = float(abs(mp.mpf(x) * mp.cosh(x) / mp.sinh(x)))
        rb, rg = rel_err(bad, ref), rel_err(good, ref)
        eb.append(rb); eg.append(rg)
        rows.append((e(x), f"{cond:.6f}", e(rb), e(rg)))
    table(["x", "cond задачи", "ошибка (eˣ − e⁻ˣ)/2", "ошибка через expm1"], rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(xs, [max(v, 1e-18) for v in eb], "o-", label="(e^x − e^−x)/2 — неустойчиво")
    ax.loglog(xs, [max(v, 1e-18) for v in eg], "s-", label="через expm1 — устойчиво")
    ax.set_xlabel("x"); ax.set_ylabel("относительная ошибка"); ax.grid(True, which="both", alpha=.3)
    ax.set_title("Задача 4: sinh x при малых x"); ax.legend()
    save(fig, "task04_sinh.png")
    conclusion(
        "**Проблема задачи:** её нет — cond ≈ 1 на всём диапазоне, малая ошибка входа даёт "
        "такую же малую ошибку выхода. **Проблема алгоритма:** формула через eˣ и e⁻ˣ "
        "вычитает два числа ≈ 1, каждое уже округлено с ошибкой ~10⁻¹⁶, и после сокращения "
        "относительная ошибка становится ~10⁻¹⁶/x (при x = 10⁻¹⁵ — уже ~5%). "
        "Вариант через expm1 даёт машинную точность на всех x."
    )


# ---------------------------------------------------------------------------
# Блок 2. Устойчивость алгоритма
# ---------------------------------------------------------------------------

def task5():
    out("## Задача 5. Квадратное уравнение")
    out()
    out("a = 1, c = 1, меньший по модулю корень x₂.")
    out()
    rows = []
    for b in (1e4, 1e8, 1e12):
        a, c = 1.0, 1.0
        d = math.sqrt(b * b - 4 * a * c)
        bad = (-b + d) / (2 * a)
        good = 2 * c / (-b - d)
        B = mp.mpf(b)
        ref = (-B + mp.sqrt(B * B - 4)) / 2
        rows.append((e(b), f"{bad:.6e}", f"{good:.6e}", mp.nstr(ref, 12),
                     e(rel_err(bad, ref)), e(rel_err(good, ref))))
    table(["b", "x₂ плохо", "x₂ хорошо", "эталон", "ошибка плохого", "ошибка хорошего"], rows)
    conclusion(
        "Ошибка плохой формулы растёт примерно как ε_mach · b²: при b = 10⁴ уже ~10⁻⁹ "
        "(потеряно 7 знаков из 16), при b = 10⁸ — 25% (√(b² − 4) отличается от b лишь в "
        "последнем бите), при b = 10¹² b² − 4 == b² в double, −b + b = 0 и ошибка 100%. "
        "Катастрофическая ошибка начинается, когда 4ac/b² сравнимо с ε_mach, т.е. при "
        "b ≳ 2/√ε_mach ≈ 10⁸. Хорошая формула не вычитает близкие числа и даёт машинную "
        "точность при всех b."
    )


def task6():
    out("## Задача 6. (1 − cos x) / x² при малых x")
    out()
    xs = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    rows = []
    for x in xs:
        bad = (1 - math.cos(x)) / x ** 2
        good = 2 * math.sin(x / 2) ** 2 / x ** 2
        X = mp.mpf(x)
        ref = (1 - mp.cos(X)) / X ** 2
        rows.append((e(x), f"{bad:.10g}", f"{good:.10g}", mp.nstr(ref, 12),
                     e(rel_err(bad, ref)), e(rel_err(good, ref))))
    table(["x", "плохо", "хорошо", "эталон", "ошибка плохого", "ошибка хорошего"], rows)

    # плотная сетка: где ошибка плохого впервые > 1 %
    grid = np.logspace(-1, -10, 400)
    eb, eg = [], []
    first = None
    for x in grid:
        x = float(x)
        X = mp.mpf(x)
        ref = (1 - mp.cos(X)) / X ** 2
        rb = rel_err((1 - math.cos(x)) / x ** 2, ref)
        eb.append(rb)
        eg.append(rel_err(2 * math.sin(x / 2) ** 2 / x ** 2, ref))
        if first is None and rb > 0.01:
            first = x
    out(f"На сетке из 400 точек ошибка плохого способа впервые превышает 1% при x ≈ {first:.2e} "
        f"(оценка: ε_mach / x² = 0.01 → x ≈ √(ε_mach/0.01) ≈ {math.sqrt(2.2e-16 / 0.01):.1e}).")
    out()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(grid, np.maximum(eb, 1e-18), ".", ms=3, label="1 − cos x (в лоб)")
    ax.loglog(grid, np.maximum(eg, 1e-18), ".", ms=3, label="2 sin²(x/2)")
    ax.axhline(0.01, color="gray", ls="--", lw=1, label="1 %")
    ax.set_xlabel("x"); ax.set_ylabel("относительная ошибка"); ax.grid(True, which="both", alpha=.3)
    ax.invert_xaxis(); ax.set_title("Задача 6: (1 − cos x)/x²"); ax.legend()
    save(fig, "task06_one_minus_cos.png")
    conclusion(
        f"Ошибка плохого способа растёт как ε_mach/x², превышает 1% около x ≈ {first:.0e}, "
        "а при x ≤ 10⁻⁸ cos x округляется ровно в 1 и результат 0 (ошибка 100%). "
        "Через 2·sin²(x/2) ошибка остаётся ~10⁻¹⁶. Правило: если в формуле из функции, "
        "близкой к константе, вычитается эта константа (1 − cos x, cos a − cos b, "
        "sin a − sin b, 1 − cos² …), надо искать тождество, которое превращает разность "
        "в произведение."
    )


def sqrt_diff_bad(x):
    return math.sqrt(x + 1) - math.sqrt(x)


def sqrt_diff_good(x):
    return 1 / (math.sqrt(x + 1) + math.sqrt(x))


def sqrt_diff_ref(x):
    X = mp.mpf(x)
    return mp.sqrt(X + 1) - mp.sqrt(X)


def task7():
    out("## Задача 7. √(x+1) − √x при больших x")
    out()
    xs = [1e4, 1e8, 1e12, 1e14, 1e16]
    rows, eb = [], []
    for x in xs:
        bad, good, ref = sqrt_diff_bad(x), sqrt_diff_good(x), sqrt_diff_ref(x)
        rb = rel_err(bad, ref)
        eb.append(rb)
        rows.append((e(x), f"{bad:.10e}", f"{good:.10e}", mp.nstr(ref, 12),
                     e(rb), e(rel_err(good, ref))))
    table(["x", "плохо", "хорошо", "эталон", "ошибка плохого", "ошибка хорошего"], rows)
    k = np.polyfit(np.log10(xs[:4]), np.log10(eb[:4]), 1)[0]
    out(f"Наклон log(ошибки) от log(x) по точкам 1e4…1e14: **{k:.2f}**.")
    out()
    conclusion(
        f"Ошибка плохого способа растёт примерно как x¹ (оценённый показатель {k:.2f}): "
        "√(x+1) и √x совпадают в первых ~log₁₀(x)/2 знаках, и каждое удвоение порядка x "
        "съедает ещё порядок точности. При x = 10¹⁶ x + 1 == x в double, ответ 0 и ошибка 100%. "
        "Домножение на сопряжённое убирает вычитание, ошибка остаётся ~10⁻¹⁶ при любом x."
    )


def task8():
    out("## Задача 8. ln(1 + x) при малых x")
    out()
    rows = []
    for x in (1e-2, 1e-8, 1e-12, 1e-16):
        bad, good = math.log(1 + x), math.log1p(x)
        ref = mp.log(1 + mp.mpf(x))
        rows.append((e(x), f"{bad:.12e}", f"{good:.12e}", mp.nstr(ref, 12),
                     e(rel_err(bad, ref)), e(rel_err(good, ref))))
    table(["x", "log(1+x)", "log1p(x)", "эталон", "ошибка log", "ошибка log1p"], rows)
    conclusion(
        "Проблема не в логарифме, а в сложении 1 + x: при x = 10⁻¹⁶ сумма округляется до 1 "
        "и ln даёт 0, при x = 10⁻¹² теряется ~4 знака из 16. В документации Python "
        "про math.log1p сказано: «calculated in a way which is accurate for x near zero» — "
        "функция получает x напрямую и не формирует 1 + x (вызывается log1p из C-библиотеки "
        "libm: ряд при малых x или поправка к ln(1 + x) на ошибку округления суммы), поэтому "
        "малая часть не теряется. Отсюда и «странные» функции "
        "стандартной библиотеки: log1p, expm1 (eˣ − 1), hypot (√(x²+y²) без переполнения "
        "x²) — это готовые устойчивые алгоритмы для типичных опасных формул."
    )


def task9():
    out("## Задача 9. Накопление ошибки в сумме Σ 1/k")
    out()
    out(f"Python {platform.python_version()}. Начиная с 3.12 встроенная `sum()` для float "
        "использует компенсированное суммирование (Неймайер), поэтому сравниваем её "
        "отдельно от наивного цикла.")
    out()
    ns = [10 ** 5, 10 ** 6, 10 ** 7]
    rows, en = [], []
    for n in ns:
        arr = [1.0 / k for k in range(1, n + 1)]
        s = 0.0
        for v in arr:
            s += v
        s_sum, s_fsum = sum(arr), math.fsum(arr)
        ref = mp.harmonic(n)
        r_naive = rel_err(s, ref)
        en.append(r_naive)
        rows.append((f"1e{int(math.log10(n))}", mp.nstr(ref, 17), e(r_naive),
                     e(rel_err(s_sum, ref)), e(rel_err(s_fsum, ref))))
    table(["n", "эталон Sₙ", "наивный цикл", "sum(arr)", "math.fsum(arr)"], rows)
    k = np.polyfit(np.log10(ns), np.log10(en), 1)[0]
    # экстраполяция до порога 1e-10
    c = en[-1] / ns[-1] ** k
    n_lim = (1e-10 / c) ** (1 / k) if k > 0 else float("inf")
    out(f"Наклон ошибки наивного цикла: ≈ n^{k:.2f}. Экстраполяция до ошибки 10⁻¹⁰: "
        f"n ≈ {n_lim:.1e}.")
    out()
    # float32 — чтобы рост был виден на тех же n
    rows32, e32 = [], []
    for n in ns:
        arr = (1.0 / np.arange(1, n + 1, dtype=np.float64)).astype(np.float32)
        s = np.float32(0)
        # наивный цикл в float32 через cumsum (последовательное сложение слева направо)
        s = np.cumsum(arr, dtype=np.float32)[-1]
        r = rel_err(float(s), mp.harmonic(n))
        e32.append(r)
        rows32.append((f"1e{int(math.log10(n))}", f"{float(s):.8g}", e(r)))
    out("Для наглядности тот же наивный цикл в float32:")
    out()
    table(["n", "сумма float32", "ошибка"], rows32)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(ns, en, "o-", label="наивный цикл, float64")
    ax.loglog(ns, e32, "s-", label="наивный цикл, float32")
    ax.axhline(1e-10, color="gray", ls="--", lw=1, label="порог 1e-10")
    ax.set_xlabel("n"); ax.set_ylabel("относительная ошибка"); ax.grid(True, which="both", alpha=.3)
    ax.set_title("Задача 9: гармоническая сумма"); ax.legend()
    save(fig, "task09_harmonic.png")
    conclusion(
        f"У наивного цикла в float64 ошибка растёт примерно как n^{k:.1f} и на n ≤ 10⁷ "
        f"остаётся ~10⁻¹⁴ — порог 10⁻¹⁰ при этих n не достигается (по экстраполяции — "
        f"около n ~ {n_lim:.0e}, т.е. практически только для огромных массивов). "
        "Однако ошибка растёт систематически: каждое маленькое 1/k прибавляется к большой "
        "сумме и теряет младшие биты. В float32 тот же цикл уже при n = 10⁵ ошибается в "
        "~4-м знаке, а дальше сумма вообще перестаёт расти. sum() в Python 3.12+ и fsum "
        "дают точность ~10⁻¹⁶ — наивный цикл для больших массивов использовать не стоит."
    )


# ---------------------------------------------------------------------------
# Блок 3. Корректная постановка эксперимента
# ---------------------------------------------------------------------------

def f6_bad(x, dtype=np.float64):
    x = dtype(x)
    return (dtype(1) - np.cos(x)) / (x * x)


def f6_good(x, dtype=np.float64):
    x = dtype(x)
    s = np.sin(x / dtype(2))
    return dtype(2) * s * s / (x * x)


def f6_ref(x):
    X = mp.mpf(float(x))
    return (1 - mp.cos(X)) / X ** 2


def task10():
    out("## Задача 10. Разделение источников ошибки")
    out()
    out("Функция из задачи 6: (1 − cos x)/x².")
    out()
    out("**(а) меняем x, алгоритм плохой, float64:**")
    out()
    ra = []
    for x in (1e-1, 1e-3, 1e-5, 1e-7):
        ra.append((e(x), e(rel_err(float(f6_bad(x)), f6_ref(x)))))
    table(["x", "ошибка"], ra)

    x0 = 1e-2
    # эталон для float32 считаем от того же x, округлённого до float32
    x32 = float(np.float32(x0))
    out(f"**(б) x = {x0}, алгоритм плохой, меняем тип:**")
    out()
    e64 = rel_err(float(f6_bad(x0, np.float64)), f6_ref(x0))
    e32 = rel_err(float(f6_bad(x0, np.float32)), f6_ref(x32))
    table(["тип", "ошибка"], [("float64", e(e64)), ("float32", e(e32))])

    out(f"**(в) x = {x0}, float64, меняем алгоритм:**")
    out()
    eg = rel_err(float(f6_good(x0)), f6_ref(x0))
    table(["алгоритм", "ошибка"], [("плохой", e(e64)), ("хороший", e(eg))])

    eg7 = rel_err(float(f6_good(1e-7)), f6_ref(1e-7))
    eb7 = rel_err(float(f6_bad(1e-7)), f6_ref(1e-7))
    orders = lambda a, b: f"{abs(math.log10(max(a, 1e-18)) - math.log10(max(b, 1e-18))):.1f}"
    ea = [rel_err(float(f6_bad(x)), f6_ref(x)) for x in (1e-1, 1e-7)]
    out("**Сводка:**")
    out()
    table(["фактор", "что меняли", "ошибка: было → стало", "порядков"], [
        ("сокращение (x)", "x: 1e-1 → 1e-7", f"{e(ea[0])} → {e(ea[1])}", orders(*ea)),
        ("машинная точность", "float64 → float32 (x=1e-2)", f"{e(e64)} → {e(e32)}", orders(e64, e32)),
        ("устойчивость алгоритма", "плохой → хороший (x=1e-2)", f"{e(e64)} → {e(eg)}", orders(e64, eg)),
        ("устойчивость алгоритма", "плохой → хороший (x=1e-7)", f"{e(eb7)} → {e(eg7)}", orders(eb7, eg7)),
    ])
    conclusion(
        "(а) При уменьшении x на 6 порядков ошибка плохого алгоритма выросла на ~11 порядков "
        "(закон ε/x²) — доминирует сокращение при вычитании 1 − cos x. (б) При фиксированном "
        "x = 0.01 переход на float32 добавил ~9 порядков: механизм тот же, но ε ≈ 6·10⁻⁸ "
        "вместо 1.1·10⁻¹⁶ — здесь доминирует машинная точность. (в) Смена алгоритма при "
        "x = 0.01 убирает ~3 порядка, а при x = 10⁻⁷ — ~13: чем сильнее сокращение, тем "
        "важнее устойчивость. Сама задача хорошо обусловлена (cond ≈ x²/6 → 0), т.е. вся эта "
        "ошибка — вклад алгоритма и точности типа, а не задачи."
    )


def task11():
    out("## Задача 11. Эталон обязателен")
    out()
    out("Два разных алгоритма для (1 − cos x)/x² при x = 10⁻⁸:")
    out()
    out("- A: `(1 - math.cos(x)) / x**2` (скалярно, модуль math);")
    out("- B: `np.float64(1) - np.cos(x)` и деление на `x*x` (numpy).")
    out()
    x = 1e-8
    a = (1 - math.cos(x)) / x ** 2
    b = float((np.float64(1) - np.cos(np.float64(x))) / (np.float64(x) * np.float64(x)))
    ref = f6_ref(x)
    table(["", "значение"], [
        ("A", repr(a)), ("B", repr(b)), ("A == B", a == b),
        ("эталон mpmath", mp.nstr(ref, 15)),
        ("ошибка A", e(rel_err(a, ref))), ("ошибка B", e(rel_err(b, ref))),
    ])
    out("Ещё пример: сумма 0.1 десять раз — циклом и через functools.reduce:")
    out()
    import functools
    s1 = 0.0
    for _ in range(10):
        s1 += 0.1
    s2 = functools.reduce(lambda p, q: p + q, [0.1] * 10)
    table(["", "значение"], [
        ("цикл", repr(s1)), ("reduce", repr(s2)), ("цикл == reduce", s1 == s2),
        ("цикл == 1.0", s1 == 1.0), ("math.fsum", repr(math.fsum([0.1] * 10))),
    ])
    conclusion(
        "Оба алгоритма дают 0.0 и совпадают через ==, но правильный ответ 0.5 — ошибка 100%. "
        "Совпадение двух float-результатов говорит лишь о том, что они повторяют одну и ту же "
        "ошибку (одинаковое округление cos x до 1, одинаковый порядок сложений). Проверять "
        "можно только относительно независимого эталона с заведомо большей точностью "
        "(mpmath, decimal, аналитический ответ)."
    )


def task12():
    out("## Задача 12. Диапазон, а не одна точка")
    out()
    xs = np.logspace(2, 16, 29)
    eb = [rel_err(sqrt_diff_bad(float(x)), sqrt_diff_ref(float(x))) for x in xs]
    eg = [rel_err(sqrt_diff_good(float(x)), sqrt_diff_ref(float(x))) for x in xs]
    table(["x", "ошибка плохого", "ошибка хорошего", "ε_mach · x"],
          [(e(x), e(b), e(g), e(2.22e-16 * x)) for x, b, g in zip(xs, eb, eg)])
    mask = (xs <= 1e15) & (np.array(eb) > 0)
    k, c = np.polyfit(np.log10(xs[mask]), np.log10(np.array(eb)[mask]), 1)
    out(f"Наклон по точкам 1e2…1e15 (МНК в log-log): **{k:.2f}**, "
        f"коэффициент 10^{c:.2f} ≈ {10 ** c:.1e} (у ε_mach — 2.2e-16).")
    out()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(xs, np.maximum(eb, 1e-18), "o-", ms=4, label="√(x+1) − √x")
    ax.loglog(xs, np.maximum(eg, 1e-18), "s-", ms=4, label="1/(√(x+1) + √x)")
    ax.loglog(xs, 2.22e-16 * xs, "--", color="gray", label="ε_mach · x")
    ax.loglog(xs, 10 ** c * xs ** k, ":", color="black", label=f"МНК: наклон {k:.2f}")
    ax.set_xlabel("x"); ax.set_ylabel("относительная ошибка"); ax.grid(True, which="both", alpha=.3)
    ax.set_title("Задача 12: ошибка на диапазоне 1e2…1e16"); ax.legend(fontsize=8)
    save(fig, "task12_sqrt_range.png")
    conclusion(
        f"Наклон ≈ {k:.2f} совпадает с предсказанием «ошибка ≈ ε_mach · x» (наклон 1); "
        "отдельные точки скачут на порядок вверх-вниз (от того, как удачно округлились √(x+1) "
        "и √x), а у x ≈ 10¹⁶ ошибка насыщается на 100%. По одной точке нельзя было бы "
        "отличить «повезло с округлением» от закономерности и нельзя оценить скорость роста; "
        "диапазон на логарифмической сетке показывает закон, границу применимости и "
        "место, где метод ломается полностью."
    )


def task13():
    out("## Задача 13. Воспроизводимость")
    out()
    out(f"Python {platform.python_version()}, numpy {np.__version__}, mpmath {mp.__version__}, "
        f"ОС {platform.system()} {platform.release()}.")
    out()

    def experiment(seed):
        random.seed(seed)
        a = [random.uniform(-1, 1) * 10 ** random.randint(-8, 8) for _ in range(10 ** 5)]
        rng = np.random.default_rng(seed)
        b = (rng.uniform(-1, 1, 10 ** 5) * 10.0 ** rng.integers(-8, 9, 10 ** 5)).tolist()
        res = {}
        for name, arr in (("random", a), ("numpy", b)):
            ref = mp.fsum([mp.mpf(v) for v in arr])
            s_naive = 0.0
            for v in arr:
                s_naive += v
            res[name] = {
                "наивный цикл": s_naive,
                "sum": sum(arr),
                "math.fsum": math.fsum(arr),
                "numpy.sum": float(np.sum(np.array(arr))),
                "_ref": ref,
            }
        return res

    r1, r2, r3 = experiment(42), experiment(42), experiment(43)
    rows = []
    for gen in ("random", "numpy"):
        for m in ("наивный цикл", "sum", "math.fsum", "numpy.sum"):
            rows.append((gen, m,
                         r1[gen][m] == r2[gen][m],
                         e(rel_err(r1[gen][m], r1[gen]["_ref"])),
                         e(rel_err(r3[gen][m], r3[gen]["_ref"]))))
    table(["генератор", "способ", "seed 42: запуск 1 == запуск 2",
           "ошибка (seed 42)", "ошибка (seed 43)"], rows)
    out("Числа: значения вида U(−1,1)·10^k, k ∈ [−8, 8], 10⁵ штук, эталон — mpmath.fsum.")
    out()
    conclusion(
        "При одном seed результаты всех способов совпадают бит в бит, т.е. эксперимент "
        "воспроизводим. При seed = 43 конкретные ошибки другие, но картина та же: fsum и "
        "sum (Python ≥ 3.12) точны до последнего бита, numpy.sum (попарное суммирование) "
        "на 1–2 порядка точнее наивного цикла, наивный цикл — худший. Значит разница между "
        "способами систематическая (следствие алгоритма), а не случайная. Без фиксации "
        "seed, версии Python (в 3.11 sum ещё наивная!) и версии numpy (меняются алгоритмы "
        "суммирования и SIMD) тот же код может дать другие числа и другой вывод."
    )


def task14():
    out("## Задача 14. Границы представимости")
    out()
    vals = {"1e300": 1e300, "1e-300": 1e-300, "0": 0.0, "1e-320 (денорм.)": 1e-320,
            "inf": float("inf"), "nan": float("nan")}

    def naive(arr):
        s = 0.0
        for v in arr:
            s += v
        return s

    def fs(arr):
        try:
            return repr(math.fsum(arr))
        except Exception as ex:  # noqa: BLE001
            return f"{type(ex).__name__}: {ex}"

    cases = [
        ("[1e300]×1000", [1e300] * 1000),
        ("[1e300, 1, −1e300]", [1e300, 1.0, -1e300]),
        ("[1e-300]×10", [1e-300] * 10),
        ("[1e-320, 1e-320]", [1e-320, 1e-320]),
        ("[1e-320, 1.0, −1.0]", [1e-320, 1.0, -1.0]),
        ("все виды без inf/nan", [1e300, 1e-300, 0.0, 1e-320, -1e300]),
        ("+ inf", [1e300, 1e-300, 0.0, 1e-320, float("inf")]),
        ("inf и −inf", [float("inf"), 1.0, float("-inf")]),
        ("+ nan (в начале)", [float("nan"), 1e300, 1.0]),
        ("+ nan (в конце)", [1e300, 1.0, float("nan")]),
        ("1e308 + 1e308 − 1e308", [1e308, 1e308, -1e308]),
    ]
    rows = [(name, repr(naive(arr)), fs(arr)) for name, arr in cases]
    table(["массив", "наивная сумма", "math.fsum"], rows)
    out(f"Наименьшее нормализованное: {sys.float_info.min:.3e}; 1e-320 — денормализованное, "
        f"1e-320 == 0: {1e-320 == 0}, 1e-320/2 = {1e-320 / 2!r}.")
    out()
    conclusion(
        "NaN «отравляет» оба способа: наивная сумма становится nan сразу при встрече и уже не "
        "восстанавливается; fsum тоже возвращает nan. inf ведёт себя как обычная бесконечность "
        "(сумма = inf), но inf + (−inf) у наивного способа даёт nan, а fsum выбрасывает "
        "ValueError, т.е. явно сообщает о проблеме. Переполнение промежуточной суммы: "
        "в 1e308 + 1e308 − 1e308 наивный цикл получает inf и уже не может вернуться, хотя "
        "ответ 1e308 представим; fsum выдаёт OverflowError «intermediate overflow». "
        "На [1e300]×1000 переполнения нет, но наивная сумма теряет младшие знаки, fsum точна. С [1e300, 1, −1e300] и [1e-320, 1, −1] наивная "
        "сумма теряет малое слагаемое (0.0), fsum возвращает его точно. Денормализованные числа складываются корректно, "
        "но с потерей точности. В реальном коде: заранее проверять math.isfinite / np.isnan "
        "и решать, что делать с такими значениями; масштабировать данные, чтобы не "
        "переполняться; для точных сумм использовать fsum и ловить его исключения."
    )


def task15():
    out("## Задача 15. Полный корректный эксперимент")
    out()
    out("**Цель.** Исследовать устойчивость вычисления f(x) = ln(1 + x) при малых x "
        "(функция из задачи 8) и определить, где наивный алгоритм ещё пригоден.")
    out()
    out("**Постановка.** Плохой алгоритм: `math.log(1 + x)`. Хороший: `math.log1p(x)`. "
        "Задача хорошо обусловлена: cond = x / ((1 + x)·ln(1 + x)) → 1 при x → 0. "
        "Вход — 12 точек логарифмической сетки от 10⁻¹ до 10⁻¹⁷ (шаг 1 порядок до 10⁻⁶, "
        "дальше 2 порядка, и 10⁻¹⁷ как точка за границей ε_mach), float64.")
    out()
    out("**Эталон.** `mpmath.log1p` с 50 значащими цифрами, вычисляемый от того же "
        "double-значения x (т.е. эталон точно соответствует входу, ошибки входа нет).")
    out()
    xs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-17]
    rows, eb, eg = [], [], []
    for x in xs:
        bad, good = math.log(1 + x), math.log1p(x)
        ref = mp.log1p(mp.mpf(x))
        rb, rg = rel_err(bad, ref), rel_err(good, ref)
        eb.append(rb); eg.append(rg)
        rows.append((e(x), f"{bad:.15e}", f"{good:.15e}", mp.nstr(ref, 16), e(rb), e(rg)))
    out("**Результаты.**")
    out()
    table(["x", "плохой", "хороший", "эталон", "ошибка плохого", "ошибка хорошего"], rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(xs, np.maximum(eb, 1e-18), "o-", label="log(1 + x)")
    ax.loglog(xs, np.maximum(eg, 1e-18), "s-", label="log1p(x)")
    ax.loglog(xs, [1.1e-16 / x for x in xs], "--", color="gray", label="ε_mach / x")
    ax.set_xlabel("x"); ax.set_ylabel("относительная ошибка"); ax.grid(True, which="both", alpha=.3)
    ax.set_ylim(1e-18, 10); ax.invert_xaxis(); ax.set_title("Задача 15: ln(1 + x)"); ax.legend()
    save(fig, "task15_log1p.png")
    good_until = [x for x, r in zip(xs, eb) if r < 1e-6]
    out(f"Плохой алгоритм даёт ошибку < 10⁻⁶ при x ≥ {min(good_until):.0e}; "
        f"максимальная ошибка хорошего на всём диапазоне: {max(eg):.1e}.")
    out()
    conclusion(
        "Неустойчивость не в логарифме, а в операции 1 + x: она округляет x до шага сетки "
        "double около 1 (≈1.1·10⁻¹⁶), относительная ошибка этого округления ~ε_mach/x, и "
        "логарифм её сохраняет. Поэтому ошибка плохого алгоритма идёт вдоль прямой ε_mach/x: "
        "10⁻¹⁵ при x = 0.1, 10⁻¹⁰ при x = 10⁻⁶, 10⁻⁴ при x = 10⁻¹², 100% при x ≤ 10⁻¹⁶. "
        "Пригоден он, если нужная точность δ больше ε_mach/x, т.е. при x ≳ ε_mach/δ "
        "(для 6 верных знаков — при x ≳ 10⁻¹⁰). log1p получает x целиком и даёт машинную "
        "точность (~10⁻¹⁶) на всём диапазоне — это и есть достижимая точность, так как "
        "задача обусловлена с cond ≈ 1."
    )


def main():
    out("# ДЗ 1. Устойчивость алгоритма и корректная постановка эксперимента")
    out()
    out(f"Окружение: Python {platform.python_version()}, numpy {np.__version__}, "
        f"mpmath {mp.__version__} (эталон: {mp.mp.dps} значащих цифр), "
        f"ε_mach(float64) = {sys.float_info.epsilon:.3e}.")
    out()
    out("# Блок 1. Обусловленность задачи")
    out()
    for t in (task1, task2, task3, task4):
        t()
    out("# Блок 2. Устойчивость алгоритма")
    out()
    for t in (task5, task6, task7, task8, task9):
        t()
    out("# Блок 3. Корректная постановка эксперимента")
    out()
    for t in (task10, task11, task12, task13, task14, task15):
        t()
    with open(os.path.join(HERE, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
