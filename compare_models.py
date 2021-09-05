import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.integrate import nquad
from hapke import hapke, hapke_ice, disk_int_hapke
import pandas as pd
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import curve_fit
from unibe import *
import os


def phase_clement(phase_angle, material, w=649):
    path = f"data/clement/phase{phase_angle}/"
    files = os.listdir(path)
    i_f = []
    for filename in files:
        df = pd.read_csv(path + filename)
        if material == "ice":
            if "br" not in filename:
                continue
        elif material == "rock":
            if "ur" not in filename:
                continue
        i_f.append(float(df.i_f[df.wavelengths == w]))
    return i_f

def wave_clement(phase_angle, ax, w=649):
    path = f"data/clement/phase{phase_angle}/"
    files = os.listdir(path)
    for filename in files:
        df = pd.read_csv(path + filename)
        if material == "ice":
            if "br" not in filename:
                continue
            ax.errorbar(df.wavelengths, df.i_f, df["std"], capsize=3, capthick=0.4, color=RED,
                        ecolor=RED,
                        elinewidth=0.4,
                        fmt='.')
        elif material == "rock":
            if "ur" not in filename:
                continue
            ax.errorbar(df.wavelengths, df.i_f, df["std"], capsize=3, capthick=0.4, color=BLACK,
                        ecolor=BLACK,
                        elinewidth=0.4,
                        fmt='.')


if __name__ == "__main__":

    fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))
    phase_angles = np.linspace(0.01, 92, 100)
    i = phase_angles
    e = np.zeros(phase_angles.shape)

    phase11_rock = phase_clement(11, "rock")
    phase11_ice = phase_clement(11, "ice")
    phase51_rock = phase_clement(51, "rock")
    phase51_ice = phase_clement(51, "ice")

    ax[0].scatter([11] * len(phase11_rock), phase11_rock, color=BLACK, label=r"rock $\alpha=11$°")
    ax[0].scatter([51] * len(phase51_rock), phase51_rock, color=BLACK, label=r"rock $\alpha=51$°")
    ax[0].scatter([11] * len(phase11_ice), phase11_ice, color=RED, label=r"ice $\alpha=11$°")
    ax[0].scatter([51] * len(phase51_ice), phase51_ice, color=RED, label=r"ice $\alpha=51$°")
    r = hapke(phase_angles, 649)
    r_ice = hapke_ice(phase_angles)
    r_di = disk_int_hapke(phase_angles, 649)
    ax[0].plot(phase_angles, r, color=BLACK, ls="--", label="hapke rock")
    ax[0].plot(phase_angles, r_ice, color=RED, ls="--", label="hapke ice?")
    ax[0].plot(phase_angles, r_di, ls="-.", color=BLACK, label="disk int hapke")
    ax[1].plot(phase_angles, r, color=BLACK, ls="--", label="hapke rock")
    ax[1].plot(phase_angles, r_ice, color=RED, ls="--", label="hapke ice?")
    ax[1].plot(phase_angles, r_di, ls="-.", color=BLACK, label="disk int hapke")

    for material in ["ice", "rock"]:
        for phase_angle in [51, 58, 89, 92, "92b"]:
            filename = f"data/deshapriya/67p_{material}_alpha_{phase_angle}.csv"
            df = pd.read_csv(filename, names=["wavelength", "r"])
            print(df.wavelength)
            df = df[(df.wavelength < 650) & (df.wavelength > 648)]
            if phase_angle == "92b": phase_angle = 92
            if material == "ice":
                c = RED
                ax[1].scatter([phase_angle] * len(df.r), df.r, marker="o", color=c)

            else:
                c = BLACK
                ax[1].scatter([phase_angle] * len(df.r), df.r, marker="o", color=c)

    a = mlines.Line2D([], [], color=BLACK, marker='o', ls='', label='rock')
    b = mlines.Line2D([], [], color=RED, marker='o', ls='', label='ice')
    c = mlines.Line2D([], [], color=BLACK, ls='--', label='hapke rock')
    d = mlines.Line2D([], [], color=RED, ls='--', label='hapke ice')
    e = mlines.Line2D([], [], color=BLACK, ls='-.', label='disk int hapke')
    ax[0].legend(handles=[a, b, c, d, e])
    ax[1].legend(handles=[a, b, c, d, e])
    ax[0].set_xlabel("phase angle [°]")
    ax[1].set_xlabel("phase angle [°]")
    ax[0].set_ylabel("I/F")
    ax[0].set_title("deshapryia 649 nm")
    ax[1].set_title("clement 649 nm")
    plt.tight_layout()
    plt.savefig("plots/hapke_deshapryia_clement_phase.png")
    plt.show()

    fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))
    phase_angles = np.array([51])
    phase_angle = 51
    wave_clement(phase_angle, ax[0])

    wavelengths = np.linspace(200, 1100)
    rock = hapke(phase_angles, wavelengths).T

    ice = np.ones(wavelengths.shape) * hapke_ice(phase_angles)
    r = disk_int_hapke(phase_angles, wavelengths).T
    ax[0].plot(wavelengths, rock, ls="--", color=BLACK, label="rock clement")
    ax[0].plot(wavelengths, ice.T, ls="--", color=RED, label="ice clement")
    ax[0].plot(wavelengths, r, ls="-.", color=BLACK, label="disk int hapke")
    ax[1].plot(wavelengths, rock, ls="--", color=BLACK, label="rock clement")
    ax[1].plot(wavelengths, ice.T, ls="--", color=RED, label="ice clement")
    ax[1].plot(wavelengths, r, ls="-.", color=BLACK, label="disk int hapke")

    for material in ["ice", "rock"]:
        for phase_angle in [51]:
            filename = f"data/deshapriya/67p_{material}_alpha_{phase_angle}.csv"
            df = pd.read_csv(filename, names=["wavelength", "r"])
            if phase_angle == "92b": phase_angle = 92
            if material == "ice":
                c = RED
                ax[1].scatter(df.wavelength, df.r, marker="o", color=c)

            else:
                c = BLACK
                ax[1].scatter(df.wavelength, df.r, marker="o", color=c)
    ax[0].set_xlabel("wavelength [nm]")
    ax[0].set_ylabel("I/F")
    ax[0].set_ylim(0, 0.15)
    ax[0].set_title("clement (51°)")
    ax[1].set_xlabel("wavelength [nm]")
    ax[1].set_ylabel("I/F")
    ax[1].set_ylim(0, 0.15)
    ax[1].set_title("deshapryia (51°)")

    a = mlines.Line2D([], [], color=BLACK, marker='o', ls='', label='rock')
    b = mlines.Line2D([], [], color=RED, marker='o', ls='', label='ice')
    c = mlines.Line2D([], [], color=BLACK, ls='--', label='hapke rock')
    d = mlines.Line2D([], [], color=RED, ls='--', label='hapke ice')
    e = mlines.Line2D([], [], color=BLACK, ls='-.', label='disk int hapke')
    ax[0].legend(handles=[a, b, c, d, e])
    ax[1].legend(handles=[a, b, c, d, e])
    plt.savefig("plots/hapke_deshapriya_clement.png")
    plt.show()
