#!/usr/bin/env python3

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from tests.test_policies import PROBLEMS, test_domain

optimal = [4, 8, 8, 5, 9, 10, 10, 8, 9, 8]

dt = time.strftime("%Y-%m-%d %H:%M:%S")
lengths = []
for seed in range(3):
    single = test_domain("satellite", debug=True, problems=10, seed=seed)
    for i, single_length in enumerate(single):
        print(i, single_length, optimal[i])
    lengths.append(single)
    print("-" * 10)
mean = np.mean(lengths, axis=0)
print(mean)
std = np.std(lengths, axis=0)
print(std)
plt.plot(mean, label="predict")
plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.5, label="predict")
plt.plot(optimal, label="optimal", color="red")
plt.savefig(f"satellite_{dt}.png")
