import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pyts.image import RecurrencePlot, MarkovTransitionField
import warnings
warnings.filterwarnings("ignore")


def normalize_data(values):
    """Простая нормализация данных."""
    return (values - values.mean()) / values.std()


def load_data_csvs_from_dirs_generator(path):
    """
    :param path: path from dirs with csvs
    :return: list - csv values
    """
    for file in sorted(Path(path).glob("*.csv")):
        df = pd.read_csv(file)
        df.dropna(inplace=True)
        values = df.iloc[:, 1].values
        yield normalize_data(values)

def load_data(path):
    data = []
    for file in sorted(Path(path).glob("*.csv")):
        df = pd.read_csv(file)

        df.dropna(inplace=True)

        values = df.iloc[:, 1].values
        data.append(normalize_data(values))
    return data


def downsample_data(data, n=5):
    truncated_length = len(data) // n * n
    truncated_data = data[:truncated_length]

    downsampled = np.array(truncated_data).reshape(-1, n).mean(axis=1)
    return downsampled

def create_image(graph, chunk_len=1600, down_sampled_window_size=5):
    # rp = RecurrencePlot()
    rp = MarkovTransitionField()
    # rp = RecurrencePlot(threshold='point', percentage=20)

    images = []
    graph_len = len(graph)
    chunk_count = max(1,graph_len // chunk_len)
    for i in range(chunk_count):
        chunk = graph[i*chunk_len:(i+1)*chunk_len]
        chunk = downsample_data(chunk,down_sampled_window_size)
        chunk_downsampled_len = int(chunk_len/down_sampled_window_size)
        #chunk_downsampled_len = len(chunk)

        img = rp.fit_transform([chunk])[0]
        current_height = img.shape[0]

        if current_height < chunk_downsampled_len:
            padding_size = chunk_downsampled_len - current_height
            zeros_to_add = np.zeros((padding_size, img.shape[1]))
            img = np.vstack((img, zeros_to_add))

        current_width = img.shape[1]
        if current_width < chunk_downsampled_len:
            pad_left = (chunk_downsampled_len - current_width) // 2
            pad_right = chunk_downsampled_len - current_width - pad_left
            img = np.pad(img, ((0, 0), (pad_left, pad_right)), mode='constant')

        images.append(img)

    return images

def duble_list_from_len_n(imgs_data, n):
    i = 0
    imgs = imgs_data
    while len(imgs) < n:
        imgs.append(imgs[i])
        i += 1
    return imgs


def create_concatenated_graph_image(input_path, output_path, result_filename, line_limit = 16):
    """
    :param input_path: Путь папки пациента
    :param output_path:  Путь вывода
    :param result_filename: Название файла вывода
    :return:
    """
    bmp_dir = Path(input_path) / 'bpm'
    bpm_generation = load_data_csvs_from_dirs_generator(bmp_dir)
    uterus_dir = Path(input_path) / 'uterus'
    uterus_generation = load_data_csvs_from_dirs_generator(uterus_dir)
    images = [[],[]]
    minimal_size_of_data = 100
    # Цикл обхода генератора пока какой-то из них не закончится
    try:
        while True:
            big_0 = True
            big_array = next(bpm_generation)
            small_array = next(uterus_generation)

            if len(big_array) < minimal_size_of_data or len(small_array) < minimal_size_of_data:
                continue

            if len(small_array) > len(big_array):
                k = big_array
                big_array = small_array
                small_array = big_array
                big_0 = False

            # Определяем новую длину (берём большую длину):
            new_length = len(big_array)

            # Используем интерполяцию, увеличивая короткий массив до длины большого массива:
            interpolated_small_array = np.interp(
                np.linspace(0, len(small_array) - 1, new_length),
                range(len(small_array)),
                small_array
            )

            if big_0:
                bpm = big_array
                uterus = interpolated_small_array
            else:
                bpm = interpolated_small_array
                uterus = big_array

            images[0] += create_image(bpm)
            images[1] += create_image(uterus)

            if len(images[0]) >= 16 or len(images[1]) >= 16:
                break

    except StopIteration:
        pass

    if not len(images[0]) or not len(images[1]):
        print(f"!!! Для пациента {input_path} нет входных данных!")

        return
    #Манипуляция с размерами
    images[0] = duble_list_from_len_n(images[0], line_limit)
    images[1] = duble_list_from_len_n(images[1], line_limit)
    images[0] = images[0][:line_limit]
    images[1] = images[1][:line_limit]

    images[0] = np.hstack(images[0])
    images[1] = np.hstack(images[1])

    concat_graphs = np.vstack(images)

    #img = Image.fromarray(concat_graphs, mode='L')
    #save_path = os.path.join('output', 'my_image.png')
    #img.save(save_path)

    plt.figure(figsize=(4, 4))
    plt.imshow(concat_graphs, cmap="gray")
    plt.axis('off')           # Скрываем оси и границы
    plt.gca().set_axis_off()  # Удаляем линии осей
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)          # Устанавливаем отступы на 0
    plt.savefig(
        os.path.join(output_path, result_filename + '.png'),
        bbox_inches='tight',
        pad_inches=0,
        dpi=300
    )
    #plt.show()

def process_all_data(input_path, output_path):
    class_types = ['hypoxia', 'regular']
    for class_type in class_types:
        current_dir = Path(input_path) / class_type

        out_path = Path(output_path) / class_type

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        inner_floders = next(os.walk(current_dir))[1]

        for folder_name in inner_floders:
            inp_path = Path(current_dir) / folder_name

            result_filename = folder_name

            create_concatenated_graph_image(inp_path, out_path, result_filename)

input_dir = 'data'

base_data_path = Path(input_dir) / 'hypoxia'
base_data_path = Path(base_data_path) / '50'
#data_path = Path(data_path) / 'bpm'

#Пример использования функции преобразования для 1 объекта
#create_concatenated_graph_image(base_data_path, 'output/hypoxia', 'hypoxia_50')

#Для всех объектов
process_all_data('data', 'output')