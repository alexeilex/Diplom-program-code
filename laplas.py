import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


# ==============================================================================
# Сборка матрицы для задачи Δp = 0 в криволинейной области
# ==============================================================================
def build_system_curved(Nx, Ny, Lx, Ly, f_bottom, f_top,
                        P_left_func=None, P_right_func=None,
                        P_in=None, P_out=None):
    """
    Строит систему A p = b для активных ячеек криволинейной области.

    Параметры:
    ----------
    Nx, Ny : int
        количество ячеек по x и y.
    Lx, Ly : float
        размеры расчётной области (прямоугольная сетка).
    f_bottom, f_top : callable
        функции одной переменной x, задающие нижнюю и верхнюю границы области.
    P_left_func, P_right_func : callable (или None)
        функции давления на левой (x=0) и правой (x=Lx) границах.
        Если не заданы, используются постоянные значения P_in и P_out.
    P_in, P_out : float
        постоянные давления на левой и правой границах (используются,
        если P_left_func / P_right_func не переданы).
    """
    # Если функции давления не заданы, создаём константные
    if P_left_func is None:
        P_left_func = lambda y: P_in
    if P_right_func is None:
        P_right_func = lambda y: P_out

    hx = Lx / Nx
    hy = Ly / Ny
    ax = 1.0 / hx**2
    ay = 1.0 / hy**2

    # Центры ячеек
    x_centers = (np.arange(Nx) + 0.5) * hx
    y_centers = (np.arange(Ny) + 0.5) * hy

    # Маска активных ячеек (центр строго внутри области)
    inside = np.zeros((Nx, Ny), dtype=bool)
    for i in range(Nx):
        for j in range(Ny):
            yb = f_bottom(x_centers[i])
            yt = f_top(x_centers[i])
            if yb < y_centers[j] < yt:
                inside[i, j] = True

    # Нумерация только активных ячеек
    idx_map = -np.ones((Nx, Ny), dtype=int)
    active = np.argwhere(inside)
    for k, (i, j) in enumerate(active):
        idx_map[i, j] = k
    N_active = len(active)

    A = sp.lil_matrix((N_active, N_active), dtype=float)
    b = np.zeros(N_active, dtype=float)

    for (i, j) in active:
        k = idx_map[i, j]
        diag = -2.0 * ax - 2.0 * ay

        # --- горизонтальные соседи (ось x) ---
        # левый сосед
        if i - 1 >= 0:
            if inside[i-1, j]:
                A[k, idx_map[i-1, j]] += ax
            else:
                diag += ax          # твёрдая стенка (Нейман)
        else:                       # открытая левая граница
            yj = y_centers[j]
            p_val = P_left_func(yj)
            diag -= ax
            b[k] -= 2.0 * ax * p_val

        # правый сосед
        if i + 1 < Nx:
            if inside[i+1, j]:
                A[k, idx_map[i+1, j]] += ax
            else:
                diag += ax
        else:                       # открытая правая граница
            yj = y_centers[j]
            p_val = P_right_func(yj)
            diag -= ax
            b[k] -= 2.0 * ax * p_val

        # --- вертикальные соседи (ось y) ---
        # нижний сосед
        if j - 1 >= 0:
            if inside[i, j-1]:
                A[k, idx_map[i, j-1]] += ay
            else:
                diag += ay          # твёрдая стенка (Нейман)
        else:
            diag += ay

        # верхний сосед
        if j + 1 < Ny:
            if inside[i, j+1]:
                A[k, idx_map[i, j+1]] += ay
            else:
                diag += ay
        else:
            diag += ay

        A[k, k] = diag

    return A.tocsr(), b, hx, hy, inside, x_centers, y_centers


# ==============================================================================
# Решение задачи
# ==============================================================================
def solve_curved(Nx, Ny, Lx, Ly, f_bottom, f_top,
                 P_left_func=None, P_right_func=None,
                 P_in=None, P_out=None):
    """
    Решает уравнение Лапласа в криволинейной области.
    Возвращает:
        A, b, p_active, p_full, inside, xc, yc
    p_full – (Nx, Ny) массив, неактивные ячейки заполнены NaN.
    """
    A, b, hx, hy, inside, xc, yc = build_system_curved(
        Nx, Ny, Lx, Ly, f_bottom, f_top,
        P_left_func, P_right_func, P_in, P_out
    )
    p_active = spla.spsolve(A, b)

    # Восстановление полного поля
    p_full = np.full((Nx, Ny), np.nan)
    for k, (i, j) in enumerate(np.argwhere(inside)):
        p_full[i, j] = p_active[k]

    return A, b, p_active, p_full, inside, xc, yc


# ==============================================================================
# Вычисление ошибок (только для активных ячеек)
# ==============================================================================
def error_norms(p_num, p_ex):
    """
    Сравнение численного и точного решений.
    Возвращает (максимальная ошибка, L2-ошибка, относительная L2-ошибка).
    """
    mask = ~np.isnan(p_num)
    err = p_num[mask] - p_ex[mask]
    if len(err) == 0:
        return 0.0, 0.0, 0.0
    max_err = np.max(np.abs(err))
    l2_err = np.sqrt(np.mean(err**2))
    rel_l2 = l2_err / max(1e-14, np.sqrt(np.mean(p_ex[mask]**2)))
    return max_err, l2_err, rel_l2


# ==============================================================================
# Визуализация
# ==============================================================================
def plot_pressure(p_full, Lx, Ly, f_bottom, f_top, title="Давление"):
    """
    Рисует поле давления p_full (неактивные ячейки = NaN).
    """
    Nx, Ny = p_full.shape
    xc = (np.arange(Nx) + 0.5) * (Lx / Nx)
    x_edges = np.linspace(0, Lx, Nx + 1)
    y_edges = np.linspace(0, Ly, Ny + 1)

    plt.figure(figsize=(8, 4))
    plt.pcolormesh(x_edges, y_edges, p_full.T, shading='flat', cmap='viridis')
    plt.colorbar(label='p')
    plt.plot(xc, f_bottom(xc), 'k', lw=2, label='bottom')
    plt.plot(xc, f_top(xc), 'k', lw=2, label='top')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(title, dpi=150, bbox_inches='tight')
    print("Plot saved as " + title)


# ==============================================================================
# Тесты
# ==============================================================================
def horizontal_test():
    # -------------------------------
    # Тест 1: Прямоугольная область
    # -------------------------------
    print("=== Тест 1: Прямые горизонтальные стенки ===")
    Lx = 3.0
    Ly = 2.0
    P_in = 1.0
    P_out = 2.0
    f_bottom_rect = lambda x: np.zeros_like(x)
    f_top_rect    = lambda x: np.full_like(x, Ly)

    for Nx, Ny in [(3, 2), (30, 20), (300, 200)]:
        A, b, p_act, p_full, inside, xc, yc = solve_curved(
            Nx, Ny, Lx, Ly, f_bottom_rect, f_top_rect,
            P_in=P_in, P_out=P_out
        )
        # Точное решение (линейное по x)
        Xc, Yc = np.meshgrid(xc, yc, indexing='ij')
        p_ex = P_in + (P_out - P_in) * (Xc / Lx)

        max_err, l2_err, rel_l2 = error_norms(p_full, p_ex)
        print(f"Сетка {Nx}x{Ny}: активных {np.sum(inside)}, "
              f"max err = {max_err:.2e}, L2 err = {l2_err:.2e}")

    # Визуализация последней сетки
    plot_pressure(p_full, Lx, Ly, f_bottom_rect, f_top_rect,
                  title="Прямоугольная_область.png")
def paralelogram_test():
    # -------------------------------
    # Тест 2: Параллелограмм
    # -------------------------------
    print("\n=== Тест 2: Параллелограмм (наклонные стенки) ===")
    Lx = 3.0
    Ly = 5.0               # чтобы вместить наклон

    # Граничные кривые
    f_bottom_par = lambda x: (4/3) * x
    f_top_par    = lambda x: (4/3) * x + 1.0

    # Точное линейное решение: p(x,y) = (3x + 4y)/25
    p_exact_func = lambda x, y: (3*x + 4*y) / 25.0

    # Граничные давления на левой и правой границах
    P_left_func  = lambda y: p_exact_func(0, y)    # 4y/25
    P_right_func = lambda y: p_exact_func(Lx, y)   # (9+4y)/25

    for Nx, Ny in [(30, 50), (60, 100), (120, 200)]:
        A, b, p_act, p_full, inside, xc, yc = solve_curved(
            Nx, Ny, Lx, Ly, f_bottom_par, f_top_par,
            P_left_func=P_left_func,
            P_right_func=P_right_func
        )
        # Точное поле на центрах ячеек
        Xc, Yc = np.meshgrid(xc, yc, indexing='ij')
        p_ex = p_exact_func(Xc, Yc)

        max_err, l2_err, rel_l2 = error_norms(p_full, p_ex)
        print(f"Сетка {Nx}x{Ny}: активных {np.sum(inside)}, "
              f"max err = {max_err:.2e}, L2 err = {l2_err:.2e}")


if __name__ == "__main__":
    paralelogram_test()
    print("\n=== Тест 3: Синусоидальный канал ===")

    wm = 1e-3        # механическая ширина
    delta = 0.8
    Lw = 1.25e-3
    Lx = 10e-3       # длина канала

    # Сдвигаем геометрию так, чтобы она находилась полностью в [0, Ly]
    Ly = 3.2e-3      # чуть больше максимальной ширины
    shift = Ly / 2   # 0.6e-3

    f_bottom = lambda x: shift - 0.5 * wm * (1 + delta * np.sin(2 * np.pi * x / Lw))
    f_top    = lambda x: shift + 0.5 * wm * (1 + delta * np.sin(2 * np.pi * x / Lw))

    # Граничные давления
    P_in = 1.0
    P_out = 2.0

    # Размер сетки (возьмём умеренный)
    Nx = 100
    Ny = 20

    A, b, p_act, p_full, inside, xc, yc = solve_curved(
        Nx, Ny, Lx, Ly, f_bottom, f_top,
        P_in=P_in, P_out=P_out
    )

    print(f"Активных ячеек: {np.sum(inside)} из {Nx*Ny}")

    # Визуализация
    plot_pressure(p_full, Lx, Ly, f_bottom, f_top, title="синусоида.png")