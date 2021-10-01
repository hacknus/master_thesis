import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from comet import ref_rock, ref_ice
from scipy.integrate import nquad
from hapke import hapke, hapke_ice, disk_int_hapke
from hapke_antoine import hapke_ref
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
        if "br" in filename:
            ax.errorbar(df.wavelengths, df.i_f, df["std"], capsize=3, capthick=0.4, color=RED,
                        ecolor=RED,
                        elinewidth=0.4,
                        fmt='.')
        if "ur" in filename:
            ax.errorbar(df.wavelengths, df.i_f, df["std"], capsize=3, capthick=0.4, color=BLACK,
                        ecolor=BLACK,
                        elinewidth=0.4,
                        fmt='.')


def antoine_hapke(phase):
    i = phase
    e = np.zeros(phase.shape)
    w = 0.047
    slope = 15
    g = -0.335
    bs = 2.38
    hs = 0.06
    w = 0.047
    return hapke_ref(i, e, phase=phase, w=w, slope=slope, hs=hs, bs=bs,
                     spsf_par=[g], spsf_type=10)


def main():
    fig, ax = plt.subplots(ncols=2, nrows=2, sharey=True, figsize=(10, 6))
    phase_angles = np.linspace(0.01, 92, 100)
    i = phase_angles
    e = np.zeros(phase_angles.shape)

    phase11_rock = phase_clement(11, "rock")
    phase11_ice = phase_clement(11, "ice")
    phase51_rock = phase_clement(51, "rock")
    phase51_ice = phase_clement(51, "ice")

    s = 15

    ax[0][0].scatter([11] * len(phase11_rock), phase11_rock, marker="x", s=s, color=BLACK, label=r"rock $\alpha=11$°")
    ax[0][0].scatter([51] * len(phase51_rock), phase51_rock, marker="x", s=s, color=BLACK, label=r"rock $\alpha=51$°")
    ax[0][0].scatter([11] * len(phase11_ice), phase11_ice, marker="x", s=s, color=RED, label=r"ice $\alpha=11$°")
    ax[0][0].scatter([51] * len(phase51_ice), phase51_ice, marker="x", s=s, color=RED, label=r"ice $\alpha=51$°")
    r = ref_rock(649, phase_angles)
    r_ice = ref_ice(649, phase_angles)

    ax[0][0].plot(phase_angles, r, color=BLACK, ls="--", label="hapke rock")
    ax[1][0].plot(phase_angles, r, color=BLACK, ls="--", label="hapke rock")

    ax[0][0].plot(phase_angles, r_ice, color=RED, ls="--", label="hapke ice")
    ax[1][0].plot(phase_angles, r_ice, color=RED, ls="--", label="hapke ice")

    for material in ["ice", "rock"]:
        for phase_angle in [51, 58, 89, 92, "92b"]:
            filename = f"data/deshapriya/67p_{material}_alpha_{phase_angle}.csv"
            df = pd.read_csv(filename, names=["wavelength", "r"])
            print(df.wavelength)
            df = df[(df.wavelength < 650) & (df.wavelength > 648)]
            if phase_angle == "92b": phase_angle = 92
            if material == "ice":
                c = RED
                ax[0][0].scatter([phase_angle] * len(df.r), df.r, marker="o", s=s, color=c)

            else:
                c = BLACK
                ax[0][0].scatter([phase_angle] * len(df.r), df.r, marker="o", s=s, color=c)

    for material in ["ice", "rock"]:
        for phase_angle in [51, 58, 89, 92, "92b"]:
            filename = f"data/deshapriya/67p_{material}_alpha_{phase_angle}.csv"
            df = pd.read_csv(filename, names=["wavelength", "r"])
            print(df.wavelength)
            df = df[(df.wavelength < 950) & (df.wavelength > 930)]
            if phase_angle == "92b": phase_angle = 92
            if material == "ice":
                c = RED
                ax[1][0].scatter([phase_angle] * len(df.r), df.r, marker="o", s=s, color=c)

            else:
                c = BLACK
                ax[1][0].scatter([phase_angle] * len(df.r), df.r, marker="o", s=s, color=c)
    phase11_rock = phase_clement(11, "rock", 932)
    phase11_ice = phase_clement(11, "ice", 932)
    phase51_rock = phase_clement(51, "rock", 932)
    phase51_ice = phase_clement(51, "ice", 932)

    ax[1][0].scatter([11] * len(phase11_rock), phase11_rock, marker="x", s=s, color=BLACK, label=r"rock $\alpha=11$°")
    ax[1][0].scatter([51] * len(phase51_rock), phase51_rock, marker="x", s=s, color=BLACK, label=r"rock $\alpha=51$°")
    ax[1][0].scatter([11] * len(phase11_ice), phase11_ice, marker="x", s=s, color=RED, label=r"ice $\alpha=11$°")
    ax[1][0].scatter([51] * len(phase51_ice), phase51_ice, marker="x", s=s, color=RED, label=r"ice $\alpha=51$°")

    a = mlines.Line2D([], [], color=BLACK, marker='o', ls='', label='deshapryia rock')
    b = mlines.Line2D([], [], color=RED, marker='o', ls='', label='deshapryia ice')
    c = mlines.Line2D([], [], color=BLACK, marker='x', ls='', label='fornasier rock')
    d = mlines.Line2D([], [], color=RED, marker='x', ls='', label='fornasier ice')
    e = mlines.Line2D([], [], color=BLACK, ls='--', label='rock')
    f = mlines.Line2D([], [], color=RED, ls='--', label='ice')
    ax[0][0].legend(handles=[a, b, c, d, e, f])
    ax[0][0].set_xlabel("phase angle [°]")
    ax[1][0].set_xlabel("phase angle [°]")
    ax[0][0].set_ylabel("I/F")
    ax[1][0].set_ylabel("I/F")
    ax[0][0].set_title(r"$\lambda$ = 649 nm")
    ax[1][0].set_title(r"$\lambda$ = 932 nm")

    wave_clement(11, ax[0][1])
    wave_clement(51, ax[1][1])

    wavelengths = np.linspace(200, 1100)
    phase_angles = np.array([11])
    rock = ref_rock(wavelengths, phase_angles).T
    ice = ref_ice(wavelengths, phase_angles).T

    ax[0][1].plot(wavelengths, rock, ls="--", color=BLACK, label="hapke")
    ax[0][1].plot(wavelengths, ice, ls="--", color=RED, label="hapke ice")

    phase_angles = np.array([51])
    rock = ref_rock(wavelengths, phase_angles).T
    ice = ref_ice(wavelengths, phase_angles).T
    ax[1][1].plot(wavelengths, rock, ls="--", color=BLACK, label="hapke")
    ax[1][1].plot(wavelengths, ice, ls="--", color=RED, label="hapke ice")

    for material in ["ice", "rock"]:
        for phase_angle in [51]:
            filename = f"data/deshapriya/67p_{material}_alpha_{phase_angle}.csv"
            df = pd.read_csv(filename, names=["wavelength", "r"])
            if phase_angle == "92b": phase_angle = 92
            if material == "ice":
                c = RED
                ax[1][1].scatter(df.wavelength, df.r, marker="o", s=s, color=c)
            else:
                c = BLACK
                ax[1][1].scatter(df.wavelength, df.r, marker="o", s=s, color=c)
    ax[0][1].set_xlabel("wavelength [nm]")
    ax[0][1].set_ylabel("I/F")
    ax[0][1].set_title(r"$\alpha$ = 11°")
    ax[1][1].set_xlabel("wavelength [nm]")
    ax[1][1].set_ylabel("I/F")
    ax[1][1].set_title(r"$\alpha$ = 51°")

    plt.tight_layout()
    plt.savefig("plots/ref.pdf")
    plt.show()

    plt.plot(wavelengths, ref_rock(wavelengths, 1.3).T)
    plt.show()


if __name__ == "__main__":
    main()
