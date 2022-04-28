#!/usr/bin/env python
from cmath import pi, cos
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
IMG1_FNAME = os.path.join(REPORT_DIR, '1.png')
IMG2_FNAME = os.path.join(REPORT_DIR, '2.png')
IMG3_FNAME = os.path.join(REPORT_DIR, '3.png')
IMG4_FNAME = os.path.join(REPORT_DIR, '4.png')
IMG5_FNAME = os.path.join(REPORT_DIR, '5.png')
IMG6_FNAME = os.path.join(REPORT_DIR, '6.png')
IMG7_FNAME = os.path.join(REPORT_DIR, '7.png')


def parse_file(f):
    first, *rest = f.readlines()

    dt = float(first.split(':')[1])
    xt = list(map(float, rest))

    return dt, xt


def save_plot(xs, ys, img_fname):
    plt.plot(xs, ys)
    plt.savefig(img_fname)
    plt.clf()


def generate_random_signal(amplitude, duration):
    signal = [
        random.uniform(
            -amplitude, amplitude
        ) for _ in range(duration)
    ]
    return signal


# Амплитуда
A = 0.5
# Длительность сигнала
N = 512


def compute_energy(xt):
    energy = sum((x*x for x in xt))
    return energy


def xt_signal(amplitude, freq1, freq2, dt, k):
    if freq1 >= freq2:
        raise ValueError('Error: freq1 >= freq2')
    
    result = amplitude*cos(freq1*k*dt) + 2*amplitude*cos(freq2*k*dt)
    return result


def make_discrete_xt(
    amplitude, freq1, freq2, dt, ks_count
):
    discrete_xt = [
        xt_signal(
            amplitude, freq1, freq2, dt, k
        ) for k in range(ks_count)
    ]
    return discrete_xt


def process_file(f):
    # Содержимое файла с вариантом
    dt, xt = parse_file(f)

    # Отсчёты
    ks = list(range(N))

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
    w40 = d_ksi*40
    w62 = d_ksi*62
    print(
        'Частота первой гармоники: {}\nЧастота второй гармоники: {}'.format(
            w40, w62
        )
    )

    # Круговые частоты
    c_w40 = w40*(2*pi)
    c_w62 = w62*(2*pi)
    print(
        'Круговая частота первой гармоники: {}\nКруговая частота второй гармоники: {}'.format(
            c_w40, c_w62
        )
    )

    # Картинка 2
    noise1 = generate_random_signal(A + 19.5, N)
    noisy_signal1 = list(map(operator.add, xt, noise1))
    complexs_noisy_signal1 = np.fft.fft(noisy_signal1)
    moduls_noisy_signal1 = list(map(sp.Abs, complexs_noisy_signal1))
    save_plot(ks, moduls_noisy_signal1, IMG2_FNAME)

    # Генерируем случайный сигнал (шум)
    noise2 = generate_random_signal(A, N)

    # Исходный сигнал с шумом
    noisy_signal2 = list(map(operator.add, xt, noise2))

    # Дискретное преобразование фурье зашумлённого сигнала
    complexs_noisy_signal2 = np.fft.fft(noisy_signal2)

    # Модули комплексных чисел в том же порядке
    # (сигнал с шумом)
    moduls_noisy_signal2 = list(map(sp.Abs, complexs_noisy_signal2))

    # Картинка 3
    save_plot(ks, moduls_noisy_signal2, IMG3_FNAME)

    energy_base_signal = compute_energy(xt)
    energy_noise2 = compute_energy(noise2)
    print(
        'Энергия исходного сигнала: {}\nЭнергия шума: {}'.format(
            energy_base_signal, energy_noise2
        )
    )

    # Картинка 4
    discrete_xt1 = make_discrete_xt(
        amplitude=0.7,
        freq1=100,
        freq2=245,
        dt=dt,
        ks_count=N,
    )
    base_with_discrete1 = list(map(operator.add, xt, discrete_xt1))
    complexs_base_with_discrete_fft1 = np.fft.fft(base_with_discrete1)
    moduls_base_with_discrete1 = list(map(sp.Abs, complexs_base_with_discrete_fft1))
    save_plot(ks, moduls_base_with_discrete1, IMG4_FNAME)

    # Картинка 5
    discrete_xt2 = make_discrete_xt(
        amplitude=0.7,
        freq1=100,
        freq2=415,
        dt=dt,
        ks_count=N,
    )
    base_with_discrete2 = list(map(operator.add, xt, discrete_xt2))
    complexs_base_with_discrete_fft2 = np.fft.fft(base_with_discrete2)
    moduls_base_with_discrete2 = list(map(sp.Abs, complexs_base_with_discrete_fft2))
    save_plot(ks, moduls_base_with_discrete2, IMG5_FNAME)

    # Картинка 6
    amplitude = 0.7
    freq1 = 100
    freq2 = 124

    discrete_xt3 = make_discrete_xt(
        amplitude=amplitude,
        freq1=freq1,
        freq2=freq2,
        dt=dt,
        ks_count=N,
    )
    base_with_discrete3 = list(map(operator.add, xt, discrete_xt3))
    complexs_base_with_discrete_fft3 = np.fft.fft(base_with_discrete3)
    moduls_base_with_discrete3 = list(map(sp.Abs, complexs_base_with_discrete_fft3))
    save_plot(ks, moduls_base_with_discrete3, IMG6_FNAME)
    
    # Берём чётные отсчёты
    _k = 0
    ks_even = []
    while _k < N//2:
        ks_even.append(_k)
        _k += 2
    dt_x2 = 2*dt

    xt_evens = [xt[k] for k in ks_even]

    short_discrete_xt3 = make_discrete_xt(
        amplitude=amplitude,
        freq1=freq1,
        freq2=freq2,
        dt=dt_x2,
        ks_count=N//2,
    )
    short_base_with_discrete3 = list(map(operator.add, xt_evens, short_discrete_xt3))
    short_complexs_base_with_discrete_fft3 = np.fft.fft(short_base_with_discrete3)
    short_moduls_base_with_discrete3 = list(map(sp.Abs, short_complexs_base_with_discrete_fft3))
    save_plot(ks_even, short_moduls_base_with_discrete3, IMG7_FNAME)


with open(SA_FNAME, 'r') as f:
    process_file(f)
