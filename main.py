#!/usr/bin/env python
from cmath import pi
import os
import operator
import random

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TASK_DIR = os.path.join(BASE_DIR, 'task/')
SA_FNAME = os.path.join(TASK_DIR, 'Sa4.tx')

REPORT_DIR = os.path.join(BASE_DIR, 'report/')
IMG0_FNAME = os.path.join(REPORT_DIR, '0.png')
IMG1_FNAME = os.path.join(REPORT_DIR, '1.png')


def parse_file(f):
    first, *rest = f.readlines()

    dt = float(first.split(':')[1])
    ks = list(range(512))
    xt = list(map(float, rest))

    return dt, ks, xt


def save_plot(xs, ys, img_fname):
    plt.plot(xs, ys)
    plt.savefig(img_fname)
    plt.clf()


EDGES = (-3.0, 3.0)
N = 512


def generate_random_signal():
    signal = [
        random.uniform(*EDGES) for _ in range(N)
    ]
    return signal


def process_file(f):
    # Содержимое файла с вариантом
    dt, ks, xt = parse_file(f)

    # Сохраняем исходный сигнал
    save_plot(ks, xt, IMG0_FNAME)

    # Дискретное преобразование Фурье (512 комплексных чисел)
    complexs = np.fft.fft(xt)

    # Модули комплексных чисел в том же порядке
    moduls = list(map(sp.Abs, complexs))

    # Сохраняем первый график
    save_plot(ks, moduls, IMG1_FNAME)

    # Находим пики
    x_peaks, _ = find_peaks(moduls)
    y_peaks = [moduls[x] for x in x_peaks]
    peaks = list(zip(x_peaks, y_peaks))
    print('Пики : {}'.format(peaks))

    # Дельта кси
    d_ksi = (2*pi)/(dt*N)

    # Просто частоты
    ksi_40 = d_ksi*40
    ksi_62 = d_ksi*62
    print(
        'Частота первой гармоники: {}\nЧастота второй гармоники: {}'.format(
            ksi_40, ksi_62
        )
    )

    # Круговые частоты
    c_ksi_40 = ksi_40*(2*pi)
    c_ksi_62 = ksi_62*(2*pi)
    print(
        'Круговая частота первой гармоники: {}\nКруговая частота второй гармоники: {}'.format(
            c_ksi_40, c_ksi_62
        )
    )

    # Генерируем слечайный сигнал (шум)
    noise = generate_random_signal()

    # Исходный сигнал с шумом
    noisy_signal = list(map(operator.add, xt, noise))


with open(SA_FNAME, 'r') as f:
    process_file(f)
