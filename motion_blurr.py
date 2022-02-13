import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from camera import Camera

from unibe import *


def omega(t, v, b):
    return (v / b) / (1 + (v / b * t) ** 2) / np.pi * 180


def get_possible_detector_time(t):
    print(f"{t=}")
    det_time = 0.220 / 1000
    print(f"{det_time}")
    t = np.ceil(t / det_time) * det_time
    t = np.max([t, det_time])
    print(f"final: {t=}")
    return t


if __name__ == "__main__":
    CoCa = Camera()
    theta = np.sqrt(CoCa.Omega) / np.pi * 180
    v2 = 10  # km/s
    v1 = 80  # km/s
    b = 1000  # km
    t_limit = 700
    t2 = np.linspace(-t_limit, t_limit, 1000)  # s
    t1 = t2 * v2 / v1
    o2 = omega(t2, v2, b)
    o1 = omega(t1, v1, b)
    fig, axes = plt.subplots(nrows=1, figsize=(8, 3))
    axes = [axes]
    # axes[0].plot(t2, 1000 * theta / o2, color=BLUE, label=r"$v=10$ km/s")
    # axes[0].plot(t1, 1000 * theta / o1, color=RED, label=r"$v=80$ km/s")
    # axes[0].axhline(0.22, color=ORANGE, label="minimum")
    # axes[0].set_ylabel(r"$t_{exp}$ [ms]")
    # axes[0].set_xlabel(r"$t$ [s]")
    alpha_1 = np.zeros(t1.shape)
    alpha_2 = np.zeros(t1.shape)
    for i in range(len(t1) // 2 + 1, len(t1), 1):
        alpha_1[i] = alpha_1[i - 1] + (t1[i] - t1[i - 1]) * o1[i]
        alpha_2[i] = alpha_2[i - 1] + (t2[i] - t2[i - 1]) * o2[i]
    for i in range(len(t1) // 2 - 1, -1, -1):
        alpha_1[i] = alpha_1[i + 1] + (t1[i] - t1[i + 1]) * o1[i]
        alpha_2[i] = alpha_2[i + 1] + (t2[i] - t2[i + 1]) * o2[i]
    axes[0].semilogy(alpha_2, 1000 * theta / o2, color=RED, ls="--", label=r"$v=10$ km/s")
    axes[0].semilogy(alpha_1, 1000 * theta / o1, color=RED, ls=":", label=r"$v=80$ km/s")

    t_labels = np.array([0, 30, 100, 300])
    tol2 = np.abs(t2[1] - t2[0]) / 2
    tol1 = np.abs(t1[1] - t1[0]) / 2
    for t_label in t_labels:
        ind1 = np.where(np.abs(t_label - t1) <= tol1)
        ind2 = np.where(np.abs(t_label - t2) <= tol2)
        if alpha_2[ind2] != 0:
            axes[0].text(alpha_2[ind2] - 2, 1000 * theta / o2[ind2] + 1, "+" + str(t_label),
                         horizontalalignment='right',
                         verticalalignment='top')
        else:
            axes[0].text(alpha_2[ind2] + 1, 1000 * theta / o2[ind2] + 1, t_label, horizontalalignment='right',
                         verticalalignment='top')
        if -alpha_2[ind2] != 0:
            axes[0].text(-alpha_2[ind2] + 2, 1000 * theta / o2[ind2] + 1, -t_label, horizontalalignment='left',
                         verticalalignment='top')
        if t_label in [0, 30]:
            if alpha_1[ind1] != 0:
                axes[0].text(alpha_1[ind1] - 1, 1000 * theta / o1[ind1] + 0.5, "+" + str(t_label),
                             horizontalalignment='right',
                             verticalalignment='top')
            else:
                axes[0].text(alpha_1[ind1] + 1, 1000 * theta / o1[ind1] + 0.1, t_label, horizontalalignment='right',
                             verticalalignment='top')
            if -alpha_1[ind1] != 0:
                axes[0].text(-alpha_1[ind1] + 1, 1000 * theta / o1[ind1] + 0.5, -t_label, horizontalalignment='left',
                             verticalalignment='top')
        axes[0].scatter(alpha_2[ind2], 1000 * theta / o2[ind2], color=BLACK)
        axes[0].scatter(-alpha_2[ind2], 1000 * theta / o2[ind2], color=BLACK)
        axes[0].scatter(alpha_1[ind1], 1000 * theta / o1[ind1], color=BLACK)
        axes[0].scatter(-alpha_1[ind1], 1000 * theta / o1[ind1], color=BLACK)

    axes[0].axhline(0.22, color=BLACK, ls="--", label="minimum")
    axes[0].set_ylabel(r"$t_{exp}$ [ms]")
    axes[0].set_xlabel(r"$\alpha$ [°]")
    axes[0].legend()
    data = {"alpha": alpha_2,
            "texp10": 1000 * theta / o2,
            "texp80": 1000 * theta / o1
            }
    df = pd.DataFrame(data=data)
    df.to_csv("data/texp.csv", index=0)
    plt.tight_layout()
    plt.savefig("texp.pdf")
    plt.show()
