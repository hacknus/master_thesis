import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.integrate import nquad
from hapke import hapke, hapke_ice, disk_int_hapke, hapke_scaled
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


if __name__ == "__main__":

    fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))
    phase_angles = np.linspace(0.01, 92, 100)
    i = phase_angles
    e = np.zeros(phase_angles.shape)

    phase11_rock = phase_clement(11, "rock")
    phase11_ice = phase_clement(11, "ice")
    phase51_rock = phase_clement(51, "rock")
    phase51_ice = phase_clement(51, "ice")

    ax[0].scatter([11] * len(phase11_rock), phase11_rock, color=BLACK, label=r"rock")
    ax[0].scatter([51] * len(phase51_rock), phase51_rock, color=BLACK)
    ax[0].scatter([11] * len(phase11_ice), phase11_ice, color=RED, label=r"ice")
    ax[0].scatter([51] * len(phase51_ice), phase51_ice, color=RED)
    r = hapke(phase_angles, 649)
    r_ice = hapke_ice(phase_angles)
    r_di = disk_int_hapke(phase_angles, 649)
    r_antoine = antoine_hapke(phase_angles)
    ax[0].plot(phase_angles, r, color=BLACK, ls="--", label="hapke i=5°")
    ax[0].plot(phase_angles, r_di, ls="-.", color=BLACK, label="hapke disk int fornasier")
    ax[0].plot(phase_angles,
               hapke_scaled(np.radians(phase_angles), np.radians(phase_angles), np.radians(phase_angles) * 0, 649),
               ls="-", color=ORANGE, label="hapke scaled i=alpha")
    ax[1].plot(phase_angles, r, color=BLACK, ls="--", label="hapke i=5°")
    ax[1].plot(phase_angles, r_di, ls="-.", color=BLACK, label="hapke disk int fornasier")
    ax[1].plot(phase_angles,
               hapke_scaled(np.radians(phase_angles), np.radians(phase_angles), np.radians(phase_angles) * 0, 649),
               ls="-", color=ORANGE, label="hapke scaled i=alpha")
    emission_angles_data = [0.746, 0.903, 0.440, 0.945, 0.903, 0.616]
    incidence_angles_data = [0.229, 1.039, 0.460, 0.751, 0.134, 1.035]
    phase_angles_data = [0.897, 0.888, 0.894, 0.887, 0.905, 0.886]

    for i, e, alpha in zip(incidence_angles_data, emission_angles_data, phase_angles_data):
        i = np.array([i])
        e = np.array([e])
        alpha = np.array([alpha])
        ax[1].scatter(np.degrees(alpha), hapke_scaled(alpha, i, e, 649), marker="x", color=ORANGE)

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

    ax[0].legend()
    ax[1].legend()
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
    ax[0].plot(wavelengths, rock, ls="--", color=BLACK, label="hapke fornasier (i=5°)")
    ax[0].plot(wavelengths, r, ls="-.", color=BLACK, label="hapke disk int fornasier")
    ax[0].plot(wavelengths,
               hapke_scaled(np.radians(phase_angles), np.radians(phase_angles), np.radians(phase_angles) * 0,
                            wavelengths).T, ls="-", color=ORANGE, label="hapke scaled i=alpha")
    ax[1].plot(wavelengths, rock, ls="--", color=BLACK, label="hapke fornasier (i=5°)")
    ax[1].plot(wavelengths, r, ls="-.", color=BLACK, label="hapke disk int fornasier")
    ax[1].plot(wavelengths,
               hapke_scaled(np.radians(phase_angles), np.radians(phase_angles), np.radians(phase_angles) * 0,
                            wavelengths).T, ls="-", color=ORANGE, label="hapke scaled i=alpha")

    for i, e, alpha in zip(incidence_angles_data, emission_angles_data, phase_angles_data):
        i = np.array([i])
        e = np.array([e])
        alpha = np.array([alpha])
        ax[1].scatter(649, hapke_scaled(alpha, i, e, 649), marker="x", color=ORANGE)

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

    ax[1].legend()
    ax[0].legend()
    plt.savefig("plots/hapke_deshapriya_clement.png")
    plt.show()
