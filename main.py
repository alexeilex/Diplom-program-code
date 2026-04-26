import numpy as np
import numpy as np
import matplotlib.pyplot as plt



# ------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------

def norm2(a):
    return np.sqrt(np.sum(a * a))


def cg_solve(A, b, x0=None, tol=1e-8, max_iter=5000):
    """
    Решает A x = b методом сопряженных градиентов.
    A(x) -- функция, которая возвращает A*x.
    """
    x = np.zeros_like(b) if x0 is None else x0.copy()

    r = b - A(x)
    b_norm = norm2(b)
    r_norm = norm2(r)

    if b_norm == 0.0 or r_norm <= tol * max(1.0, b_norm):
        return x

    p = r.copy()
    rs_old = np.sum(r * r)

    for _ in range(max_iter):
        Ap = A(p)
        denom = np.sum(p * Ap)

        if abs(denom) < 1e-30:
            break

        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = np.sum(r * r)
        if np.sqrt(rs_new) <= tol * max(1.0, b_norm):
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x
import numpy as np
import matplotlib.pyplot as plt


def poiseuille_analytic_profile(y, Lx, Ly, P_in, P_out, nu):
    """
    Аналитический профиль Пуазёйля для канала:
        u(y) = (dp/dx)/(2*nu) * y * (Ly - y)
    где dp/dx = (P_out - P_in)/Lx.
    Эквивалентно:
        u(y) = ((P_in - P_out)/(2*nu*Lx)) * y * (Ly - y)
    """
    fx = (P_in - P_out) / Lx
    return (fx / (2.0 * nu)) * y * (Ly - y)


def compare_with_poiseuille(u_full, hx, hy, Lx, Ly, P_in, P_out, nu, make_plot=True):
    """
    Сравнивает численный профиль u(y) с аналитическим решением Пуазёйля.
    Берём усреднение по x по внутренним значениям, потому что u в MAC-сетке
    зависит от x-индекса на вертикальных гранях.
    """
    Ny = u_full.shape[1]

    # y-координаты для u: это центры по y
    y_u = (np.arange(Ny) + 0.5) * hy

    # численный профиль: усредняем по x, исключая граничные нули
    u_num = np.mean(u_full[1:-1, :], axis=0)

    # аналитика
    u_exact = poiseuille_analytic_profile(y_u, Lx, Ly, P_in, P_out, nu)

    # ошибки
    abs_err = np.abs(u_num - u_exact)
    max_err = np.max(abs_err)
    l2_err = np.sqrt(np.mean((u_num - u_exact) ** 2))
    rel_l2_err = l2_err / max(1e-14, np.sqrt(np.mean(u_exact ** 2)))

    print("\nПроверка профиля Пуазёйля")
    print(f"max error     = {max_err:.6e}")
    print(f"L2 error      = {l2_err:.6e}")
    print(f"relative L2   = {rel_l2_err:.6e}")

    if make_plot:
        plt.figure(figsize=(7, 4))
        plt.plot(y_u, u_num, label="численный профиль u(y)")
        plt.plot(y_u, u_exact, "--", label="аналитика Пуазёйля")
        plt.xlabel("y")
        plt.ylabel("u")
        plt.title("Сравнение профиля скорости")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return y_u, u_num, u_exact, abs_err

# ------------------------------------------------------------
# Дискретные операторы на MAC-сетке
# ------------------------------------------------------------

def neg_laplacian_dirichlet(phi, hx, hy):
    """
    Возвращает -Δ_h phi для массива с нулевыми Дирихле на границах.
    phi может быть:
      - u_int формы (Nx-1, Ny)
      - v_int формы (Nx, Ny-1)
    """
    P = np.pad(phi, ((1, 1), (1, 1)), mode="constant", constant_values=0.0)

    lap = (
        (P[2:, 1:-1] - 2.0 * P[1:-1, 1:-1] + P[:-2, 1:-1]) / hx**2
        + (P[1:-1, 2:] - 2.0 * P[1:-1, 1:-1] + P[1:-1, :-2]) / hy**2
    )
    return -lap


def grad_p_to_u(p, hx):
    """
    ∂p/∂x на вертикальных гранях.
    p shape: (Nx, Ny)
    return shape: (Nx-1, Ny)
    """
    return (p[1:, :] - p[:-1, :]) / hx


def grad_p_to_v(p, hy):
    """
    ∂p/∂y на горизонтальных гранях.
    p shape: (Nx, Ny)
    return shape: (Nx, Ny-1)
    """
    return (p[:, 1:] - p[:, :-1]) / hy


def embed_u(u_int):
    """
    Вкладывает внутренние u в полный массив shape (Nx+1, Ny).
    Границы i=0 и i=Nx равны 0.
    """
    Nx_minus_1, Ny = u_int.shape
    u = np.zeros((Nx_minus_1 + 2, Ny))   # <-- было +1, надо +2
    u[1:-1, :] = u_int
    return u

def embed_v(v_int):
    """
    Вкладывает внутренние v в полный массив shape (Nx, Ny+1).
    Границы j=0 и j=Ny равны 0.
    """
    Nx, Ny_minus_1 = v_int.shape
    v = np.zeros((Nx, Ny_minus_1 + 2))   # <-- было +1, надо +2
    v[:, 1:-1] = v_int
    return v


def divergence_mac(u, v, hx, hy):
    """
    Дивергенция в центрах ячеек.
    u shape: (Nx+1, Ny)
    v shape: (Nx, Ny+1)
    return shape: (Nx, Ny)
    """
    return (u[1:, :] - u[:-1, :]) / hx + (v[:, 1:] - v[:, :-1]) / hy


# ------------------------------------------------------------
# Решатель Стокса методом Узавы
# ------------------------------------------------------------

def solve_stokes_uzawa(
    Nx, Ny, Lx, Ly, nu,
    fx_u, fy_v,
    omega=0.5,
    uzawa_tol=1e-8,
    uzawa_max_iter=200,
    cg_tol=1e-10,
    cg_max_iter=5000,
    verbose=True,
):
    """
    Стационарный Стокс на MAC-сетке:
        -nu * Δu + ∇p = f
        div u = 0

    Неизвестные:
        u_int shape (Nx-1, Ny)
        v_int shape (Nx, Ny-1)
        p     shape (Nx, Ny)

    fx_u -- правая часть для u в узлах u_int
    fy_v -- правая часть для v в узлах v_int
    """

    hx = Lx / Nx
    hy = Ly / Ny

    # Проверка размеров правых частей
    assert fx_u.shape == (Nx - 1, Ny), f"fx_u must have shape {(Nx-1, Ny)}"
    assert fy_v.shape == (Nx, Ny - 1), f"fy_v must have shape {(Nx, Ny-1)}"

    p = np.zeros((Nx, Ny))
    u_int = np.zeros((Nx - 1, Ny))
    v_int = np.zeros((Nx, Ny - 1))

    def Au(x):
        return nu * neg_laplacian_dirichlet(x, hx, hy)

    def Av(x):
        return nu * neg_laplacian_dirichlet(x, hx, hy)

    for k in range(uzawa_max_iter):
        # 1) скорость по текущему давлению
        rhs_u = fx_u - grad_p_to_u(p, hx)
        rhs_v = fy_v - grad_p_to_v(p, hy)

        u_new = cg_solve(Au, rhs_u, x0=u_int, tol=cg_tol, max_iter=cg_max_iter)
        v_new = cg_solve(Av, rhs_v, x0=v_int, tol=cg_tol, max_iter=cg_max_iter)

        # 2) собираем полные поля и считаем невязку несжимаемости
        u_full = embed_u(u_new)
        v_full = embed_v(v_new)
        div = divergence_mac(u_full, v_full, hx, hy)

        # 3) обновление давления
        p_new = p + omega * div
        p_new -= np.mean(p_new)  # калибровка давления

        # 4) критерий остановки
        du = norm2(u_new - u_int) / max(1.0, norm2(u_new))
        dv = norm2(v_new - v_int) / max(1.0, norm2(v_new))
        dp = norm2(p_new - p) / max(1.0, norm2(p_new))
        dd = norm2(div) / np.sqrt(Nx * Ny)

        if verbose:
            print(f"iter {k+1:4d}: du={du:.3e}, dv={dv:.3e}, dp={dp:.3e}, div={dd:.3e}")

        u_int, v_int, p = u_new, v_new, p_new

        if max(du, dv, dp, dd) < uzawa_tol:
            if verbose:
                print(f"Uzawa converged in {k+1} iterations")
            break

    u_full = embed_u(u_int)
    v_full = embed_v(v_int)
    return u_full, v_full, p, hx, hy


# ------------------------------------------------------------
# Пример запуска
# ------------------------------------------------------------
if __name__ == "__main__":
    Lx = 4.0
    Ly = 1.0
    Nx = 256
    Ny = 64
    nu = 1.0

    P_in = 1.0
    P_out = 0.0

    fx_u = ((P_in - P_out) / Lx) * np.ones((Nx - 1, Ny))
    fy_v = np.zeros((Nx, Ny - 1))

    u, v, p, hx, hy = solve_stokes_uzawa(
        Nx, Ny, Lx, Ly, nu,
        fx_u, fy_v,
        omega=0.5,
        uzawa_tol=1e-8,
        uzawa_max_iter=100,
        cg_tol=1e-10,
        cg_max_iter=4000,
        verbose=True,
    )

    print("done")
    print("u shape:", u.shape)
    print("v shape:", v.shape)
    print("p shape:", p.shape)

    # Сравнение с Пуазёйлем
    compare_with_poiseuille(u, hx, hy, Lx, Ly, P_in, P_out, nu, make_plot=True)

