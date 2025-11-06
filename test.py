import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
import seaborn as sns
import gc  # Импортируем модуль очистки памяти

def normalize_data(values):
    """Простая нормализация данных."""
    return (values - values.mean()) / values.std()

def load_data(path):
    df = pd.read_csv(path)
    values = df.iloc[:, 1].values
    return normalize_data(values)

input_dir = 'data'


base_data_path = Path(input_dir) / 'hypoxia'
base_data_path = Path(base_data_path) / '2'
#data_path = Path(data_path) / 'bpm'

graphs_images = []
n = 5

for graph in ['bpm', 'uterus']:
    images = []
    data_path = Path(base_data_path) / graph

    for file in sorted(Path(data_path).glob("*.csv")):
        data  = load_data(file)

        truncated_length = len(data) // n * n
        truncated_data = data[:truncated_length]

        # Reshape и усреднение
        downsampled = truncated_data.reshape(-1, n).mean(axis=1)

        rp = RecurrencePlot(threshold='point', percentage=20)
        img = rp.fit_transform([downsampled ])[0]

        current_height = img.shape[0]
        required_height = 2000
        if current_height < required_height:
            padding_size = required_height - current_height
            zeros_to_add = np.zeros((padding_size, img.shape[1]))  # создаем массив с нулями нужного размера
            img = np.vstack((img, zeros_to_add))  # добавляем строки снизу

        current_width = img.shape[1]
        required_width = 2000
        if current_width < required_width:
            pad_left = (required_width - current_width) // 2
            pad_right = required_width - current_width - pad_left
            img = np.pad(img, ((0, 0), (pad_left, pad_right)), mode='constant')

        images.append(img)
    h_concat_bpm = np.hstack(images)
    graphs_images.append(h_concat_bpm)

c_concat_bpm = np.vstack(graphs_images)
plt.figure(figsize=(8, 8))
plt.imshow(c_concat_bpm, cmap="gray")
plt.show()