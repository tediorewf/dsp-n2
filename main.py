#!/usr/bin/env python
import os

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

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


with open(SA_FNAME, 'r') as f:
    process_file(f)
