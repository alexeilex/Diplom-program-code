import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from laplas import solve_curved
# --- Чтение эталонных данных (уже разрез по Y = -4e-4) ---
data = np.loadtxt('output_Y_-4.000000e-04.txt', delimiter=',', skiprows=1)
x_cfd = data[:, 0]      # X
z_cfd = data[:, 1]      # Z (вертикальная координата)
p_cfd = data[:, 2]      # Давление

# Нам нужен одномерный профиль p(x). Усредним по Z (хотя в этом слое Z почти постоянен)
# Просто возьмём все точки, отсортируем по X и усредним,
# но поскольку структура данных регулярна, можно взять уникальные X и соответствующие P
unique_x = np.unique(x_cfd)
p_x_cfd = np.array([np.mean(p_cfd[x_cfd == x]) for x in unique_x])

# Параметры канала из статьи (форма "а", файл 39_div.plt)
wm = 1e-3                # механическая ширина
delta = 0.8              # безразмерная амплитуда
Lw = 1.25e-2           # период шероховатости
Lx = unique_x[-1] - unique_x[0]   # длина области по X

# Граничные давления из эталона
Pin_cfd = p_x_cfd[0]
Pout_cfd = p_x_cfd[-1]

print(f"Lx = {Lx:.6f} м")
print(f"Pin = {Pin_cfd:.3f} Па, Pout = {Pout_cfd:.3f} Па")

# --- Ваши функции (должны быть определены выше) ---
# ... (вставьте сюда solve_curved, build_system_curved и т.д.)
# Предположим, что они уже импортированы или определены

# Геометрия для вашего решателя (со сдвигом, чтобы все y>0)
# В эталоне z от -0.00075 до +0.00075, центр 0. Сдвинем на 0.001, чтобы нижняя граница стала >0

Ly = 3.2e-3      # чуть больше максимальной ширины
Nx = 100                 # число ячеек по x
Ny = 40                  # по y

shift = Ly / 2
f_bottom = lambda x: shift - 0.5 * wm * (1 + delta * np.sin(2 * np.pi * x / Lw))
f_top    = lambda x: shift + 0.5 * wm * (1 + delta * np.sin(2 * np.pi * x / Lw))
# Запуск вашего решателя с эталонными граничными давлениями
A, b, p_act, p_full, inside, xc, yc = solve_curved(
    Nx, Ny, Lx, Ly, f_bottom, f_top,
    P_in=Pin_cfd, P_out=Pout_cfd
)

# Профиль давления по центру канала (середина по y)
j_mid = Ny // 2
p_laplace = p_full[:, j_mid]

# Интерполируем эталонные данные на центры xc
p_cfd_interp = np.interp(xc, unique_x, p_x_cfd)

# --- График сравнения ---
plt.figure(figsize=(8, 5))
plt.plot(xc * 1e3, p_laplace, 'r-', linewidth=2, label='Идеальная жидкость (Лаплас)')
plt.plot(unique_x * 1e3, p_x_cfd, 'k--', linewidth=2, label='Вязкое течение (CFD)')
plt.xlabel('x, мм')
plt.ylabel('Давление, Па')
plt.title('Сравнение профилей давления в канале формы (а)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sravnenie.png", dpi=150, bbox_inches='tight')
print("Plot saved as sravnenie.png")
# Ошибка
error = p_laplace - p_cfd_interp
max_err = np.max(np.abs(error))
rmse = np.sqrt(np.mean(error**2))
print(f"Максимальная разница: {max_err:.3f} Па")
print(f"Среднеквадратичная ошибка: {rmse:.3f} Па")
print(f"Ошибка в процентах относительно перепада: {max_err/(Pin_cfd-Pout_cfd) *100:.6f}%")