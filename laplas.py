import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def idx(i, j, Nx):
    return i + j * Nx


def build_system(Nx, Ny, Lx, Ly, P_in, P_out):
    """
    Δp = 0 на прямоугольнике.
    По x: Dirichlet через ghost cells второго порядка:
        p_{-1,j} = 2 P_in - p_{0,j}
        p_{Nx,j} = 2 P_out - p_{Nx-1,j}
    По y: Neumann 0 через отражение:
        p_{i,-1} = p_{i,0}
        p_{i,Ny} = p_{i,Ny-1}
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

            diag = -2.0 * ax - 2.0 * ay

            # x-направление
            if Nx == 1:
                # обе границы одновременно
                diag -= 2.0 * ax
                b[k] -= 2.0 * ax * (P_in + P_out)
            else:
                if i == 0:
                    diag -= ax
                    A[k, idx(i + 1, j, Nx)] += ax
                    b[k] -= 2.0 * ax * P_in
                elif i == Nx - 1:
                    diag -= ax
                    A[k, idx(i - 1, j, Nx)] += ax
                    b[k] -= 2.0 * ax * P_out
                else:
                    A[k, idx(i - 1, j, Nx)] += ax
                    A[k, idx(i + 1, j, Nx)] += ax

            # y-направление
            if Ny == 1:
                # обе Neumann-границы одновременно => вклад по y = 0
                diag += 2.0 * ay
            else:
                if j == 0:
                    diag += ay
                    A[k, idx(i, j + 1, Nx)] += ay
                elif j == Ny - 1:
                    diag += ay
                    A[k, idx(i, j - 1, Nx)] += ay
                else:
                    A[k, idx(i, j - 1, Nx)] += ay
                    A[k, idx(i, j + 1, Nx)] += ay

            A[k, k] += diag

    return A.tocsr(), b, hx, hy


def solve_system(Nx, Ny, Lx, Ly, P_in, P_out):
    A, b, hx, hy = build_system(Nx, Ny, Lx, Ly, P_in, P_out)
    p_vec = spla.spsolve(A, b)
    p = p_vec.reshape((Nx, Ny), order="F")
    return A, b, p, hx, hy


def discrete_exact_solution(Nx, Ny, Lx, Ly, P_in, P_out):
    """
    Для этой схемы точное дискретное решение совпадает с линейной функцией
    в центрах ячеек.
    """
    hx = Lx / Nx
    x = (np.arange(Nx) + 0.5) * hx
    p_x = P_in + (P_out - P_in) * (x / Lx)
    p = np.repeat(p_x[:, None], Ny, axis=1)
    return p


def continuous_exact_solution(Nx, Ny, Lx, Ly, P_in, P_out):
    hx = Lx / Nx
    x = (np.arange(Nx) + 0.5) * hx
    p_x = P_in + (P_out - P_in) * (x / Lx)
    p = np.repeat(p_x[:, None], Ny, axis=1)
    return p


def error_norms(p_num, p_ex):
    err = p_num - p_ex
    max_err = np.max(np.abs(err))
    l2_err = np.sqrt(np.mean(err**2))
    rel_l2_err = l2_err / max(1e-14, np.sqrt(np.mean(p_ex**2)))
    return max_err, l2_err, rel_l2_err


def run_case(Nx, Ny, Lx, Ly, P_in, P_out, print_matrix=False):
    A, b, p_num, hx, hy = solve_system(Nx, Ny, Lx, Ly, P_in, P_out)
    p_ex = discrete_exact_solution(Nx, Ny, Lx, Ly, P_in, P_out)

    max_err, l2_err, rel_l2_err = error_norms(p_num, p_ex)

    print(f"\nСетка {Nx}x{Ny}")
    print(f"hx = {hx:.6e}, hy = {hy:.6e}")
    print(f"max error   = {max_err:.6e}")
    print(f"L2 error    = {l2_err:.6e}")
    print(f"rel L2      = {rel_l2_err:.6e}")

    if print_matrix:
        np.set_printoptions(precision=3, suppress=True)
        print("\nA =")
        print(A.toarray())
        print("\nb =")
        print(b)
        print("\np_num =")
        print(p_num)
        print("\np_ex =")
        print(p_ex)

    return A, b, p_num, p_ex


if __name__ == "__main__":
    Lx = 3.0
    Ly = 2.0
    P_in = 1.0
    P_out = 2.0
    
for Nx, Ny in [(3, 2), (30, 20), (300, 200)]:
    run_case(Nx, Ny, Lx, Ly, P_in, P_out, print_matrix=False)