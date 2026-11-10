import numpy as np

# 1
a = np.arange(24).reshape(6, 4)
print(a)
print("shape:", a.shape, "ndim:", a.ndim, "size:", a.size, "dtype:", a.dtype)

# 2
b = a.reshape(4, 6)
c = b.reshape(2, 3, 4)
d = c.reshape(24,)          # или c.ravel()
print("равны:", np.array_equal(d, np.arange(24)), np.array_equal(d.reshape(6, 4), a))

# 3
col = a[:, 2]
print("столбец:", col, "| view?", np.shares_memory(col, a), "| base is a:", col.base is a.base or col.base is a)
col[0] = 999
print("a[0, 2] после изменения среза:", a[0, 2])   # 999 -> это view
col[0] = 2                                        # вернём как было

# 4
mask = a > 10
filtered = a[mask]
print("элементы > 10:", filtered)
print("среднее:", filtered.mean())

# 5
centered = a - a.mean(axis=1, keepdims=True)
print(centered)
print("форма:", centered.shape, "== a.shape:", centered.shape == a.shape)
print("суммы по строкам:", centered.sum(axis=1), "~0:", np.allclose(centered.sum(axis=1), 0))
