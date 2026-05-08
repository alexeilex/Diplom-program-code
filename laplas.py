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

def hyperbolic_channel_test():
    """
    Тест: гиперболический канал, ограниченный линиями тока z^2 (y = C/(2x)).
    Аналитическое решение: p(x,y) = x^2 - y^2, на стенках ∂p/∂n = 0.
    """
    print("\n=== Тест: гиперболический канал (аналитическое решение z²) ===")

    # Константы линий тока
    C_bottom = 1.0
    C_top = 2.0

    # Интервал по x: [x_left, x_right]
    x_left = 0.8
    x_right = 2.0
    Lx = x_right - x_left

    # Определим вертикальные границы, чтобы кривые поместились в [0, Ly]
    y_min = C_bottom / (2.0 * x_right)   # минимальная y (на правом краю)
    y_max = C_top / (2.0 * x_left)       # максимальная y (на левом краю)
    Ly = y_max * 1.05                    # небольшой запас сверху

    # Функции границ (принимают локальный x ∈ [0, Lx])
    def f_bottom(x_loc):
        x_orig = x_loc + x_left
        return C_bottom / (2.0 * x_orig)

    def f_top(x_loc):
        x_orig = x_loc + x_left
        return C_top / (2.0 * x_orig)

    # Аналитическое решение (в физических координатах)
    def p_exact(x_orig, y):
        return x_orig**2 - y**2

    # Граничные давления на левом и правом срезах (вертикальные линии)
    def P_left(y):
        return p_exact(x_left, y)

    def P_right(y):
        return p_exact(x_right, y)

    # Тестовые сетки
    for Nx, Ny in [(40, 30), (80, 60), (160, 120)]:
        A, b, p_act, p_full, inside, xc, yc = solve_curved(
            Nx, Ny, Lx, Ly,
            f_bottom, f_top,
            P_left_func=P_left,
            P_right_func=P_right
        )

        # Координаты центров ячеек в физической системе
        Xc_phys = xc + x_left          # (Nx,)
        Yc_phys = yc                   # (Ny,) – уже в физической системе
        Xg, Yg = np.meshgrid(Xc_phys, Yc_phys, indexing='ij')
        p_ex = p_exact(Xg, Yg)

        max_err, l2_err, rel_l2 = error_norms(p_full, p_ex)
        print(f"Сетка {Nx}x{Ny}: активных {np.sum(inside)}, "
              f"max err = {max_err:.2e}, L2 err = {l2_err:.2e}")

    # Визуализация для средней сетки
    Nx, Ny = 80, 60
    A, b, p_act, p_full, inside, xc, yc = solve_curved(
        Nx, Ny, Lx, Ly, f_bottom, f_top,
        P_left_func=P_left, P_right_func=P_right
    )

#    import matplotlib.pyplot as plt
    x_edges = np.linspace(0, Lx, Nx+1) + x_left
    y_edges = np.linspace(0, Ly, Ny+1)
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(x_edges, y_edges, p_full.T, shading='flat', cmap='viridis')
    plt.colorbar(label='p')
    x_plot = np.linspace(0, Lx, 200) + x_left
    plt.plot(x_plot, f_bottom(x_plot - x_left), 'k', lw=2, label='bottom')
    plt.plot(x_plot, f_top(x_plot - x_left), 'k', lw=2, label='top')
    plt.title('Гиперболический канал (z²)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.savefig('hyperbolic_channel.png', dpi=150)
    print("Plot saved as hyperbolic_channel.png")

def converging_diverging_channel_test():
    """
    Тест: канал с верхней прямой y = a*x + offset_up
          и нижней прямой y = -a*x + offset_bottom (offset_bottom = offset_up - 1).
    Аналитическое решение: p(x,y) = ln( (x - x0)^2 + (y - y0)^2 ),
    где (x0, y0) – точка пересечения граничных прямых (особенность вне области).
    """
    print("\n=== Тест: сходящийся/расходящийся канал (аналитика ln(r)) ===")
    a = 1.0                # наклон верхней стенки
    offset_up = 2.0        # верхняя прямая: y = x + 2
    offset_bottom = offset_up - 1.0   # нижняя: y = -x + 1 (сдвиг на 1 вниз при x=0)

    # Точка пересечения
    x0 = (offset_bottom - offset_up) / (2 * a)   # для a=1, offset_up=2 → -0.5
    y0 = a * x0 + offset_up                      # y0 = 1.5

    Lx = 2.0               # длина канала (точка пересечения x0 < 0)
    # Подбираем Ly так, чтобы границы поместились
    y_max = a * Lx + offset_up           # верх при x=Lx
    y_min = -a * Lx + offset_bottom      # низ при x=Lx (может быть и ниже)
    Ly = max(y_max, offset_up) + 0.2     # запас, чтобы y_min>=0? Проверим: нужно y_min >=0. offset_bottom=1, при Lx=2, y_min = -2+1 = -1, отрицательное. Значит, сдвигаем всё вверх.
    # Проще: сдвинем геометрию, чтобы нижняя граница была >= 0. Добавим сдвиг shift.
    shift = max(0.0, -y_min) + 0.1
    y0_shifted = y0 + shift
    # Переопределим граничные функции с учётом сдвига
    def f_bottom(x):
        return -a * x + offset_bottom + shift
    def f_top(x):
        return a * x + offset_up + shift
    # Обновлённый Ly
    Ly = a * Lx + offset_up + shift + 0.2

    # Граничные давления на левой и правой вертикальных границах
    def P_left(y):
        # p_exact(0, y) = ln( (0 - x0)^2 + ((y - shift) - y0)^2 )
        dy = (y - shift) - y0
        return np.log(x0**2 + dy**2)   # x0 отрицательное, но в квадрате

    def P_right(y):
        # p_exact(Lx, y) = ln( (Lx - x0)^2 + ((y - shift) - y0)^2 )
        dx = Lx - x0
        dy = (y - shift) - y0
        return np.log(dx**2 + dy**2)

    # Точное решение на всей сетке (для проверки)
    def p_exact_func(x_mesh, y_mesh):
        # x_mesh, y_mesh – координаты центров ячеек в исходной системе (без shift)
        # Учтём, что y в расчётной области = y_phys + shift? 
        # У нас f_bottom и f_top принимают x физический и возвращают y с учётом shift.
        # В сетке координаты yc заданы в [0, Ly], где Ly включает shift.
        # Физическая координата y_phys = yc, потому что мы сдвинули стенки.
        # Значит, в формуле p_exact нужно использовать yc (не вычитая shift), а точка y0 должна быть в той же сдвинутой системе.
        # Удобнее: перенесём точку пересечения в сдвинутую систему.
        y0_eff = y0 + shift
        return np.log((x_mesh - x0)**2 + (y_mesh - y0_eff)**2)

    # Тестовые сетки
    for Nx, Ny in [(40, 60), (80, 120), (160, 240)]:
        A, b, p_act, p_full, inside, xc, yc = solve_curved(
            Nx, Ny, Lx, Ly, f_bottom, f_top,
            P_left_func=P_left,
            P_right_func=P_right
        )
        Xg, Yg = np.meshgrid(xc, yc, indexing='ij')
        p_ex = p_exact_func(Xg, Yg)

        max_err, l2_err, rel_l2 = error_norms(p_full, p_ex)
        print(f"Сетка {Nx}x{Ny}: активных {np.sum(inside)}, "
              f"max err = {max_err:.2e}, L2 err = {l2_err:.2e}")

    # Визуализация для средней сетки
    Nx, Ny = 80, 120
    A, b, p_act, p_full, inside, xc, yc = solve_curved(
        Nx, Ny, Lx, Ly, f_bottom, f_top,
        P_left_func=P_left,
        P_right_func=P_right
    )
    x_edges = np.linspace(0, Lx, Nx+1)
    y_edges = np.linspace(0, Ly, Ny+1)
    plt.figure(figsize=(8, 5))
    plt.pcolormesh(x_edges, y_edges, p_full.T, shading='flat', cmap='viridis')
    plt.colorbar(label='p')
    x_plot = np.linspace(0, Lx, 200)
    plt.plot(x_plot, f_top(x_plot), 'k', lw=2, label='top')
    plt.plot(x_plot, f_bottom(x_plot), 'k', lw=2, label='bottom')
    plt.title('Канал с наклонными непараллельными стенками (ln(r))')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.savefig('converging_channel.png', dpi=150)
    print("Plot saved as converging_channel.png")
def sin_test():
    print("\n=== Тест 3: Синусоидальный канал ===")

    wm = 1e-3        # механическая ширина
    delta = 0.8
    Lw = 1.25e-3
    Lx = 1.25e-3      # длина канала

    # Сдвигаем геометрию так, чтобы она находилась полностью в [0, Ly]
    Ly = 3.2e-3      # чуть больше максимальной ширины
    shift = Ly / 2   # 0.6e-3

    f_bottom = lambda x: shift - 0.5 * wm * (1 + delta * np.sin(2 * np.pi * x / Lw))
    f_top    = lambda x: shift + 0.5 * wm * (1 + delta * np.sin(2 * np.pi * x / Lw))

    # Граничные давления
    P_in = 15.7
    P_out = 14.4

    # Размер сетки (возьмём умеренный)
    Nx = 100
    Ny = 40

    A, b, p_act, p_full, inside, xc, yc = solve_curved(
        Nx, Ny, Lx, Ly, f_bottom, f_top,
        P_in=P_in, P_out=P_out
    )

    print(f"Активных ячеек: {np.sum(inside)} из {Nx*Ny}")

    # Визуализация
    plot_pressure(p_full, Lx, Ly, f_bottom, f_top, title="синусоида.png")
if __name__ == "__main__":
    hyperbolic_channel_test()
    #paralelogram_test()
    converging_diverging_channel_test()
