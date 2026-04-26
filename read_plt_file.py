import numpy as np
import matplotlib.pyplot as plt
import sys

def read_tecplot_data(filename):
    """
    Читает файл Tecplot с переменными и зоной.
    Возвращает:
        var_names : list - имена переменных
        dims : tuple - (i, j, k)
        data : np.ndarray - массив формы (n_points, n_vars)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Поиск строки с Variables
    var_line = None
    zone_line = None
    data_start = None
    for idx, line in enumerate(lines):
        if line.strip().startswith('Variables'):
            var_line = line.strip()
        elif line.strip().startswith('Zone'):
            zone_line = line.strip()
            data_start = idx + 1
            break

    if var_line is None or zone_line is None:
        raise ValueError("Не найден заголовок Variables или Zone")

    # Парсим имена переменных
    # Пример: "Variables=X,Y,Z,P,U,V,W,Div"
    var_part = var_line.split('=')[1]
    var_names = [v.strip() for v in var_part.split(',')]

    # Парсим размеры зоны
    # Пример: "Zone  i = 195 j =   5 k =  35"
    import re
    match = re.search(r'i\s*=\s*(\d+)\s+j\s*=\s*(\d+)\s+k\s*=\s*(\d+)', zone_line)
    if not match:
        raise ValueError("Не удалось распарсить Zone")
    i, j, k = map(int, match.groups())
    dims = (i, j, k)
    n_points = i * j * k

    # Чтение данных
    data_lines = lines[data_start:data_start + n_points]
    if len(data_lines) < n_points:
        raise ValueError(f"Недостаточно данных: ожидается {n_points} строк, найдено {len(data_lines)}")

    data = []
    for line in data_lines:
        # Удаляем возможные комментарии в конце строки, если есть
        line = line.split('#')[0].strip()
        if not line:
            continue
        values = list(map(float, line.split()))
        if len(values) != len(var_names):
            raise ValueError(f"Несоответствие числа значений: {len(values)} != {len(var_names)}")
        data.append(values)

    return var_names, dims, np.array(data)

def extract_for_y(data, var_names, target_y, atol=1e-10):
    """
    Извлекает строки X, Z, P для точек, у которых Y близок к target_y.
    """
    idx_X = var_names.index('X')
    idx_Y = var_names.index('Y')
    idx_Z = var_names.index('Z')
    idx_P = var_names.index('P')

    mask = np.isclose(data[:, idx_Y], target_y, atol=atol)
    selected = data[mask]
    if len(selected) == 0:
        print(f"Внимание: не найдено точек с Y = {target_y} (допуск {atol})")
        return None
    # Возвращаем массив [X, Z, P]
    return selected[:, [idx_X, idx_Z, idx_P]]

def write_output(output_filename, extracted):
    """Записывает извлечённые данные в файл (X, Z, P) с заголовком."""
    with open(output_filename, 'w') as f:
        f.write("X, Z, P\n")
        for row in extracted:
            f.write(f"{row[0]:.8e}, {row[1]:.8e}, {row[2]:.8e}\n")
    print(f"Данные записаны в {output_filename}")

def plot_distribution(x_vals, z_vals, p_vals):
    """
    Строит scatter-график: X vs Z, цвет = P.
    """
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(x_vals, z_vals, c=p_vals, cmap='jet', s=20, edgecolor='none')
    plt.colorbar(sc, label='Давление P')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Распределение давления в плоскости Y = const')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')  # если сетка равномерная
    plt.tight_layout()
    plt.savefig("distribution.png", dpi=150, bbox_inches='tight')
    print("Plot saved as distribution.png")
def main():
    input_file = '39_div.plt'

    # Чтение данных
    print("Чтение файла...")
    var_names, dims, data = read_tecplot_data(input_file)
    print(f"Переменные: {var_names}")
    print(f"Размеры зоны: i={dims[0]}, j={dims[1]}, k={dims[2]}")
    print(f"Всего точек: {data.shape[0]}")

    # Определяем уникальные значения Y
    idx_Y = var_names.index('Y')
    unique_y = np.unique(data[:, idx_Y])
    print(f"Доступные значения Y: {unique_y}")

    # Запрашиваем целевое Y (можно ввести вручную)
    try:
        target_y = float(input(f"Введите значение Y из списка выше: "))
    except ValueError:
        print("Некорректный ввод")
        sys.exit(1)

    if target_y not in unique_y:
        # Поиск ближайшего с допуском
        closest = unique_y[np.argmin(np.abs(unique_y - target_y))]
        if np.abs(closest - target_y) < 1e-8:
            target_y = closest
        else:
            print(f"Внимание: Y={target_y} отсутствует. Ближайшее: {closest}")
            target_y = closest

    # Извлечение данных
    extracted = extract_for_y(data, var_names, target_y)
    if extracted is None:
        sys.exit(1)

    # Запись в файл
    output_filename = f"output_Y_{target_y:.6e}.txt"
    write_output(output_filename, extracted)

    # Построение графика
    x_vals = extracted[:, 0]
    z_vals = extracted[:, 1]
    p_vals = extracted[:, 2]
    plot_distribution(x_vals, z_vals, p_vals)

if __name__ == "__main__":
    main()