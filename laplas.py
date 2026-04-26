import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Построение системы для криволинейной области
# ----------------------------------------------------------------------
def build_system_curved(Nx, Ny, Lx, Ly, P_in, P_out, f_bottom, f_top):
    """
    Строит A p = b для активных ячеек криволинейной области.
    f_bottom(x) и f_top(x) — функции, задающие нижнюю и верхнюю границы.
    """
    hx = Lx / Nx
    hy = Ly / Ny
    ax = 1.0 / hx**2
    ay = 1.0 / hy**2

    # Координаты центров ячеек
    x_centers = (np.arange(Nx) + 0.5) * hx
    y_centers = (np.arange(Ny) + 0.5) * hy

    # Маска активных ячеек
    inside = np.zeros((Nx, Ny), dtype=bool)
    for i in range(Nx):
        for j in range(Ny):
            y_bottom = f_bottom(x_centers[i])
            y_top = f_top(x_centers[i])
            if y_bottom < y_centers[j] < y_top:
                inside[i, j] = True

    # Перенумерация активных ячеек
    idx_map = -np.ones((Nx, Ny), dtype=int)
    active = np.argwhere(inside)  # массив пар (i, j)
    for k, (i, j) in enumerate(active):
        idx_map[i, j] = k
    N = len(active)

    # Разреженная матрица и правая часть
    A = sp.lil_matrix((N, N), dtype=float)
    b = np.zeros(N, dtype=float)

    for (i, j) in active:
        k = idx_map[i, j]
        diag = -2.0 * ax - 2.0 * ay

        # Левый сосед
        if i - 1 >= 0:
            if inside[i-1, j]:
                A[k, idx_map[i-1, j]] += ax
            else:
                diag += ax          # Нейман (отражение)
        else:                       # открытая левая граница
            diag -= ax
            b[k] -= 2.0 * ax * P_in

        # Правый сосед
        if i + 1 < Nx:
            if inside[i+1, j]:
                A[k, idx_map[i+1, j]] += ax
            else:
                diag += ax
        else:                       # открытая правая граница
            diag -= ax
            b[k] -= 2.0 * ax * P_out

        # Нижний сосед
        if j - 1 >= 0:
            if inside[i, j-1]:
                A[k, idx_map[i, j-1]] += ay
            else:
                diag += ay          # Нейман
        else:
            diag += ay              # на всякий случай, если вдруг дошли до y=0

        # Верхний сосед
        if j + 1 < Ny:
            if inside[i, j+1]:
                A[k, idx_map[i, j+1]] += ay
            else:
                diag += ay
        else:
            diag += ay

        A[k, k] = diag

    return A.tocsr(), b, hx, hy, inside, x_centers, y_centers


def solve_curved(Nx, Ny, Lx, Ly, P_in, P_out, f_bottom, f_top):
    A, b, hx, hy, inside, xc, yc = build_system_curved(
        Nx, Ny, Lx, Ly, P_in, P_out, f_bottom, f_top
    )
    p_active = spla.spsolve(A, b)

    # Восстановление на полной сетке (NaN для неактивных)
    p_full = np.full((Nx, Ny), np.nan)
    for k, (i, j) in enumerate(np.argwhere(inside)):
        p_full[i, j] = p_active[k]

    return A, b, p_active, p_full, inside, xc, yc


# ----------------------------------------------------------------------
# Точные решения (линейная функция по x) для прямоугольной области
# ----------------------------------------------------------------------
def discrete_exact_solution(Nx, Ny, Lx, Ly, P_in, P_out):
    """Точное дискретное решение для полной прямоугольной сетки."""
    hx = Lx / Nx
    x = (np.arange(Nx) + 0.5) * hx
    p_x = P_in + (P_out - P_in) * (x / Lx)
    p = np.repeat(p_x[:, None], Ny, axis=1)
    return p


def continuous_exact_solution(Nx, Ny, Lx, Ly, P_in, P_out):
    """Непрерывное точное решение (линейная функция)."""
    hx = Lx / Nx
    x = (np.arange(Nx) + 0.5) * hx
    p_x = P_in + (P_out - P_in) * (x / Lx)
    p = np.repeat(p_x[:, None], Ny, axis=1)
    return p


# ----------------------------------------------------------------------
# Вычисление ошибок
# ----------------------------------------------------------------------
def error_norms(p_num, p_ex):
    """Считает ошибки только по конечным значениям (не NaN)."""
    mask = ~np.isnan(p_num)
    err = p_num[mask] - p_ex[mask]
    max_err = np.max(np.abs(err)) if len(err) > 0 else 0.0
    l2_err = np.sqrt(np.mean(err**2)) if len(err) > 0 else 0.0
    rel_l2_err = l2_err / max(1e-14, np.sqrt(np.mean(p_ex[mask]**2)))
    return max_err, l2_err, rel_l2_err


# ----------------------------------------------------------------------
# Запуск одного расчёта
# ----------------------------------------------------------------------
def run_case(Nx, Ny, Lx, Ly, P_in, P_out, f_bottom=None, f_top=None, print_matrix=False):
    """
    Расчёт давления в криволинейной области.
    Если f_bottom и f_top не заданы, используются прямые стенки y=0 и y=Ly.
    """
    if f_bottom is None:
        f_bottom = lambda x: 0.0
    if f_top is None:
        f_top = lambda x: Ly

    A, b, p_active, p_full, inside, xc, yc = solve_curved(
        Nx, Ny, Lx, Ly, P_in, P_out, f_bottom, f_top
    )

    # Точное решение для полной сетки (линейное по x)
    p_ex = continuous_exact_solution(Nx, Ny, Lx, Ly, P_in, P_out)

    # Сравнение только в активных ячейках
    max_err, l2_err, rel_l2_err = error_norms(p_full, p_ex)

    print(f"\nСетка {Nx}x{Ny}")
    print(f"Активных ячеек: {np.sum(inside)} из {Nx*Ny}")
    print(f"hx = {Lx/Nx:.6e}, hy = {Ly/Ny:.6e}")
    print(f"max error   = {max_err:.6e}")
    print(f"L2 error    = {l2_err:.6e}")
    print(f"rel L2      = {rel_l2_err:.6e}")

    if print_matrix:
        np.set_printoptions(precision=3, suppress=True)
        print("\nA =")
        print(A.toarray())
        print("\nb =")
        print(b)
        print("\np_active =")
        print(p_active)
        print("\np_full =")
        print(p_full)
        print("\np_ex =")
        print(p_ex)

    return A, b, p_full, p_ex

def plot_pressure(p_full, Lx, Ly, f_bottom, f_top, title="Давление в канале"):
    """
    Визуализация поля давления p_full.
    p_full: (Nx, Ny) массив, NaN для неактивных ячеек.
    f_bottom, f_top: функции границ.
    """
    Nx, Ny = p_full.shape
    xc = (np.arange(Nx) + 0.5) * (Lx / Nx)
    yc = (np.arange(Ny) + 0.5) * (Ly / Ny)
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
    plt.show()
    plt.savefig('plot.png', dpi=150, bbox_inches='tight')
    print("Plot saved as plot.png")
# ----------------------------------------------------------------------
# Основной блок
# ----------------------------------------------------------------------
if __name__ == "__main__":
    Lx = 3.0
    Ly = 2.0
    P_in = 1.0
    P_out = 2.0

    # 1. Проверка на прямых границах (должна совпадать со старым кодом)
 #   print("=== Прямые границы (f_bottom=0, f_top=Ly) ===")
  #  for Nx, Ny in [(3, 2), (30, 20), (300, 200)]:
   #     run_case(Nx, Ny, Lx, Ly, P_in, P_out)

    # 2. Пример с криволинейными границами
    print("\n=== Криволинейные границы(параллелограмм) ===")
    f_bottom = lambda x: 4.0/3.0*x
    f_top = lambda x: 4.0/3.0 * x+1.0
    for Nx, Ny in [(3, 2), (30, 20), (300, 200)]:
        A, b, p_full, p_ex = run_case(Nx, Ny, Lx, Ly, P_in, P_out, f_bottom, f_top, print_matrix=False)
# Визуализация
plot_pressure(p_full, Lx, Ly, f_bottom, f_top, title="Давление в криволинейном канале")