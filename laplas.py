import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_solutions_3d(p_num, p_ex, Lx, Ly, scale_error=1.0):
    # --- ЖЁСТКО приводим к numpy ---
    if hasattr(p_num, "toarray"):
        p_num = p_num.toarray()
    p_num = np.asarray(p_num, dtype=float)

    if hasattr(p_ex, "toarray"):
        p_ex = p_ex.toarray()
    p_ex = np.asarray(p_ex, dtype=float)

    # Проверка формы
    if p_num.shape != p_ex.shape:
        raise ValueError(f"Shape mismatch: {p_num.shape} vs {p_ex.shape}")

    Nx, Ny = p_num.shape

    hx = Lx / Nx
    hy = Ly / Ny

    x = (np.arange(Nx) + 0.5) * hx
    y = (np.arange(Ny) + 0.5) * hy

    X, Y = np.meshgrid(x, y, indexing='ij')

    error = p_num - p_ex

    fig = plt.figure(figsize=(18, 5))

    # Численное
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, p_num)
    ax1.set_title("Numerical")

    # Точное
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, p_ex)
    ax2.set_title("Exact")

    # Ошибка
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, scale_error * error)
    ax3.set_title(f"Error x{scale_error}")

    plt.tight_layout()
    plt.show()
def idx(i, j, Nx):
    """
    Нумерация неизвестных p[i,j] в один индекс:
    k = i + j*Nx
    i = 0..Nx-1, j = 0..Ny-1
    """
    return i + j * Nx


def build_system(Nx, Ny, Lx, Ly, P_in, P_out):
    """
    Строит систему A p = b для задачи

        Δ p = 0

    на прямоугольнике [0, Lx] x [0, Ly] со следующими условиями:
        p(0, y)   = P_in
        p(Lx, y)  = P_out
        ∂p/∂y = 0 на y=0 и y=Ly

    Неизвестные p[i,j] расположены в центрах ячеек.
    Схема ровно соответствует твоей записи:
        ( (p_{i+1,j}-p_{i,j})/hx - (p_{i,j}-p_{i-1,j})/hx ) / hx
      + ( (p_{i,j+1}-p_{i,j})/hy - (p_{i,j}-p_{i,j-1})/hy ) / hy
    """
    hx = Lx / Nx
    hy = Ly / Ny

    ax = 1.0 / hx**2
    ay = 1.0 / hy**2

    N = Nx * Ny
    A = sp.lil_matrix((N, N), dtype=float)
    b = np.zeros(N, dtype=float)

    for j in range(Ny):
        for i in range(Nx):
            k = idx(i, j, Nx)

            # -------------------
            # x-направление
            # -------------------
            if Nx == 1:#тут поменять граничные условия потом
                # Если одна колонка, то слева и справа одновременно границы
                # Δ_x p = (P_out - 2 p + P_in)/hx^2
                A[k, k] += -2.0 * ax
                b[k] += -ax * (P_in + P_out)

            elif i == 0:
                # Левая граница:
                # (p_{1,j} - 2 p_{0,j} + P_in) / hx^2
                A[k, k] += -2.0 * ax+ 1.0*ax/2
                A[k, idx(i + 1, j, Nx)] += ax
                b[k] += -ax * P_in/2

            elif i == Nx - 1:
                # Правая граница:
                # (P_out - 2 p_{Nx-1,j} + p_{Nx-2,j}) / hx^2
                A[k, k] += -2.0 * ax+1.0*ax/2
                A[k, idx(i - 1, j, Nx)] += ax
                b[k] += -ax * P_out/2

            else:
                # Внутренняя точка
                A[k, k] += -2.0 * ax
                A[k, idx(i - 1, j, Nx)] += ax
                A[k, idx(i + 1, j, Nx)] += ax

            # -------------------
            # y-направление
            # -------------------
            if Ny == 1:
                # Одна строка по y: вклад по y отсутствует
                pass

            elif j == 0:
                # Нижняя граница: ∂p/∂y = 0
                # (p_{i,1} - p_{i,0}) / hy^2
                A[k, k] += -1.0 * ay
                A[k, idx(i, j + 1, Nx)] += ay

            elif j == Ny - 1:
                # Верхняя граница: ∂p/∂y = 0
                # (p_{i,Ny-2} - p_{i,Ny-1}) / hy^2
                A[k, k] += -1.0 * ay
                A[k, idx(i, j - 1, Nx)] += ay

            else:
                # Внутренняя точка
                A[k, k] += -2.0 * ay
                A[k, idx(i, j - 1, Nx)] += ay
                A[k, idx(i, j + 1, Nx)] += ay

    return A.tocsr(), b, hx, hy


def discrete_exact_solution(Nx, Ny, P_in, P_out):
    """
    Точное решение именно ДИСКРЕТНОЙ задачи для этой постановки.

    Для каждой строки по y решение одинаковое:
        p_i = ((Nx - i) * P_in + (i + 1) * P_out) / (Nx + 1)

    Это удовлетворяет A p = b для собранной выше схемы.
    """
    x_vals = np.array([
        ((Nx - i) * P_in + (i + 1) * P_out) / (Nx + 1)
        for i in range(Nx)
    ], dtype=float)
    p = np.repeat(x_vals[:, None], Ny, axis=1)
    return p


def continuous_exact_solution(Nx, Ny, Lx, Ly, P_in, P_out):
    """
    Точное решение непрерывной задачи:
        p(x, y) = P_in + (P_out - P_in) * x / Lx
    на центрах ячеек.
    """
    hx = Lx / Nx
    x = (np.arange(Nx) + 0.5) * hx
    p = P_in + (P_out - P_in) * (x[:, None] / Lx)
    p = np.repeat(p, Ny, axis=1)
    return p


def residual_norm(A, p):
    """
    Норма невязки ||A p - b||_∞ и ||A p - b||_2,
    если b уже известен снаружи.
    """
    raise NotImplementedError("Use residual_norm_with_rhs(A, p, b)")


def residual_norm_with_rhs(A, p, b):
    r = A @ p.reshape(-1, order="F") - b
    return np.max(np.abs(r)), np.sqrt(np.mean(r**2))


def error_norms(p_num, p_ex):
    err = p_num - p_ex
    max_err = np.max(np.abs(err))
    l2_err = np.sqrt(np.mean(err**2))
    rel_l2_err = l2_err / max(1e-14, np.sqrt(np.mean(p_ex**2)))
    return max_err, l2_err, rel_l2_err


def run_case(Nx, Ny, Lx, Ly, P_in, P_out, print_matrix=False):
    A, b, hx, hy = build_system(Nx, Ny, Lx, Ly, P_in, P_out)

    # Дискретно-точное решение для этой постановки
    p_num = discrete_exact_solution(Nx, Ny, P_in, P_out)

    # Непрерывное точное решение (линейная функция)
    p_ex = continuous_exact_solution(Nx, Ny, Lx, Ly, P_in, P_out)

    # Невязка именно для собранной матрицы
    res_inf, res_rms = residual_norm_with_rhs(A, p_num, b)

    # Ошибка относительно непрерывного решения
    max_err, l2_err, rel_l2_err = error_norms(p_num, p_ex)

    print(f"\nСетка {Nx}x{Ny}")
    print(f"hx = {hx:.6e}, hy = {hy:.6e}")
    #print(f"residual inf-norm = {res_inf:.6e}")
    #print(f"residual rms      = {res_rms:.6e}")
    print(f"max error         = {max_err:.6e}")
    #print(f"L2 error          = {l2_err:.6e}")
    #print(f"relative L2       = {rel_l2_err:.6e}")

    if print_matrix:
        np.set_printoptions(precision=3, suppress=True)
        print("\nA =")
        print(A.toarray())
        print("\nb =")
        print(b)
        print("\nDiscrete exact solution p_num =")
        print(p_num)
        print("\nContinuous exact solution p_ex =")
        print(p_ex)

    return A, b, p_num, p_ex


if __name__ == "__main__":
    Lx = 3.0
    Ly = 2.0
    P_in = 1.0
    P_out = 2.0

    # Маленькая сетка: здесь удобно сверять матрицу вручную
    A, b,p_num, p_ex = run_case(30, 20, Lx, Ly, P_in, P_out,print_matrix=False)
    plot_solutions_3d(p_num, p_ex, Lx, Ly, scale_error=1)
    # Проверка на другой ориентации
    run_case(3, 2, Lx, Ly, P_in, P_out, print_matrix=True)

    # Более крупные сетки
    for Nx, Ny in [(30, 20), (300, 200)]:
        run_case(Nx, Ny, Lx, Ly, P_in, P_out, print_matrix=False)