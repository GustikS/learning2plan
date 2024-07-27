#!/usr/bin/env python3

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from tests.test_policies import PROBLEMS, test_domain

problems = [f"0_{i:02}" for i in range(11, 16)]
optimal = [17, 11, 8, 18, 13]
lama = [21, 12, 10, 18, 13]

# optimal = [4, 8, 8, 5, 9, 10, 10, 8, 9, 8]

dt = time.strftime("%Y-%m-%d %H:%M:%S")
lengths = []
for seed in range(3):
    
    single = test_domain("satellite", debug=True, problems=problems, seed=seed)
    for i, single_length in enumerate(single):
        print(i, single_length, optimal[i])
    lengths.append(single)
    print("-" * 10)
mean = np.mean(lengths, axis=0)
mean_str = [f"{f:.2f}" for f in mean]
print(mean_str)
std = np.std(lengths, axis=0)
std_str = [f"{f:.2f}" for f in std]
print(std_str)

plt.plot(mean, label="predict")
plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.5, label="predict")
plt.plot(optimal, label="optimal", color="red")
plt.plot(lama, label="lama", color="green")
plt.savefig(f"satellite_{dt}.png")
