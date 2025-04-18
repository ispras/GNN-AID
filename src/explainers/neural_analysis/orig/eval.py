import copy
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def stats(d):
    m = defaultdict(list)

    for i in range(len(d[0])):
        m[d[0][i]].append(d[1][i])

    keys = list(m.keys())
    keys.sort()

    means = []
    stds = []

    for k in keys:
        means.append(np.mean(m[k]))
        stds.append(np.std(m[k]) ** 2)

    means = np.array(means)
    stds = np.array(stds)

    return keys, means, stds


def plot_fidelity(*datapoints, bin_size=0.05, lim=0.5):
    import seaborn as sns
    datapoints = copy.deepcopy(datapoints)
    n = len(datapoints)

    plt.xlim([0, lim])

    fg = ['red', 'blue', 'green', 'orange', 'red', 'blue', 'green', 'orange', 'red', 'blue', 'green', 'orange']
    bg = ['magenta', 'cyan', 'lime', 'wheat', 'magenta', 'cyan', 'lime', 'wheat', 'magenta', 'cyan', 'lime', 'wheat']

    for i in range(n):
        for j in range(len(datapoints[i][0])):
            datapoints[i][0][j] = math.floor(datapoints[i][0][j] / bin_size) * bin_size

        keys, means, stds = stats(datapoints[i])

        plt.plot(keys, means, color=fg[i], label=str(i + 1))
        plt.fill_between(keys, means - stds ** 2, means + stds ** 2, alpha=0.4, color=bg[i])

    plt.xlabel('Sparsity')
    plt.ylabel('Fidelity')

    plt.legend()
