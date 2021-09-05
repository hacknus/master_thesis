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

# This is Table 4, "Hapke 2002" line from Fornasier et al. 2015:
# Results from the Hapke (2002, and 2012) modeling from disk-resolved
# images taken with the NAC orange filter centered on 649 nm.
omega = 0.042
g = -0.37
b0_s = 2.5
h_s = 0.079
b0_c = 0.188
h_c = 0.017
theta = 15.


def fit_model():
    fig, ax = plt.subplots(nrows=1, sharex=True)
    phase_angle = 51

    w, mat = plot_clement(ax, phase_angle, "rock")
    ax.plot(w, mat, ls="--", color=BLACK, label="mean rock")
    w, mat = plot_clement(None, phase_angle, "ice")
    ax.plot(w, mat, ls="--", color=RED, label="mean ice")
    ax.legend()
    ax.set_title(f"phase angle={phase_angle}°")
    ax.set_xlabel("wavelengths [nm]")
    ax.set_ylabel("I/F")
    plt.show()


def fit_hapke():
    fig, ax = plt.subplots(nrows=1, sharex=True)

    phase_angles = np.linspace(0.01, 90, 100)
    i = phase_angles
    e = np.zeros(phase_angles.shape)

    pa_ice = []
    pa_rock = []
    i_f_ice = []
    i_f_rock = []

    for material in ["ice", "rock"]:
        for phase_angle in [51, 58, 89, 92, "92b"]:
            filename = f"data/deshapriya/67p_{material}_alpha_{phase_angle}.csv"
            df = pd.read_csv(filename, names=["wavelength", "r"])
            reflectance = interp1d(df["wavelength"], df["r"], fill_value="extrapolate", kind='quadratic')
            if material == "ice":
                c = RED
                if phase_angle == "92b":
                    pa_ice.append(92)
                else:
                    pa_ice.append(phase_angle)
                i_f_ice.append(float(reflectance(649)))
            else:
                c = BLACK
                if phase_angle == "92b":
                    pa_rock.append(92)
                else:
                    pa_rock.append(phase_angle)
                i_f_rock.append(float(reflectance(649)))

    phase11_rock = phase_clement(11, "rock")
    phase11_ice = phase_clement(11, "ice")
    phase51_rock = phase_clement(51, "rock")
    phase51_ice = phase_clement(51, "ice")

    ax.scatter([11] * len(phase11_rock), phase11_rock, color=BLACK, label=r"clement rock")
    ax.scatter([51] * len(phase51_rock), phase51_rock, color=BLACK)
    ax.scatter([11] * len(phase11_ice), phase11_ice, color=RED, label=r"clement ice")
    ax.scatter([51] * len(phase51_ice), phase51_ice, color=RED)

    ax.scatter(pa_rock, i_f_rock, color=BLACK, marker="x", label=r"deshapryia rock°")
    ax.scatter(pa_ice, i_f_ice, color=RED, marker="x", label=r"deshapryia rock°")

    pa_rock = list(pa_rock) + [11] * len(phase11_rock) + [51] * len(phase51_rock)
    i_f_rock = list(i_f_rock) + list(phase11_rock) + list(phase51_rock)
    pa_ice = list(pa_ice) + [11] * len(phase11_ice) + [51] * len(phase51_ice)
    i_f_ice = list(i_f_ice) + list(phase11_ice) + list(phase51_ice)

    omega = 0.042
    theta = 15.
    h_s = 0.079
    b0_s = 2.5
    h_c = 0.017
    b0_c = 0.188
    g = -0.37

    p0 = [omega, theta, h_s, b0_s, h_c, b0_c, g]
    bounds = ((0, 0, 0, 0, 0, 0, -1), (2 * np.pi, 30, 1, 2 * np.pi, 1, 1, 1))
    pa_rock = np.array(pa_rock)
    i_f_rock = np.array(i_f_rock)
    pa_rock = pa_rock[i_f_rock < 0.04]
    i_f_rock = i_f_rock[i_f_rock < 0.04]
    popt_rock, pcov_rock = curve_fit(hapke_int, pa_rock, i_f_rock, p0=p0, maxfev=10000, bounds=bounds)

    np.set_printoptions(precision=3)
    # print(np.array(p0))
    # print(np.array(popt_rock))

    r = hapke_int(phase_angles, *popt_rock)
    ax.plot(phase_angles, r, color=BLACK, ls="--", label="hapke rock")

    r = hapke_int(phase_angles, *p0)
    ax.plot(phase_angles, r, color=BLACK, ls="-.", label="hapke rock")

    plt.legend()
    plt.xlabel("phase angle [°]")
    plt.ylabel("I/F")
    plt.show()
    # plot clement
    # plot deshapriya
    # fit data
    # save hapke parameters


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


def plot_clement(ax=None, phase_angle=11, material="ice"):
    path = f"data/clement/phase{phase_angle}/"
    files = os.listdir(path)
    w = np.array([])
    mat = np.array([])
    std = np.array([])
    filters = 0
    for filename in files:
        df = pd.read_csv(path + filename)
        if "brf" in filename:
            if ax is not None: ax.errorbar(df.wavelengths, df.i_f, df["std"], capsize=3, capthick=0.4, color=RED,
                                           ecolor=RED,
                                           elinewidth=0.4,
                                           fmt='.')
            if material == "ice":
                w = np.append(w, np.array(df.wavelengths))
                mat = np.append(mat, np.array(df.i_f))
                std = np.append(std, np.array(df["std"]))
                filters = len(df.wavelengths)
        elif "brs" in filename:
            if ax is not None: ax.errorbar(df.wavelengths, df.i_f, df["std"], capsize=3, capthick=0.4, color=ORANGE,
                                           ecolor=ORANGE,
                                           elinewidth=0.4,
                                           fmt='.')
            if material == "ice":
                w = np.append(w, np.array(df.wavelengths))
                mat = np.append(mat, np.array(df.i_f))
                std = np.append(std, np.array(df["std"]))
                filters = len(df.wavelengths)
        elif "spc" in filename:
            if ax is not None: ax.errorbar(df.wavelengths, df.i_f, df["std"], capsize=3, capthick=0.4, color=GREEN,
                                           ecolor=GREEN,
                                           elinewidth=0.4,
                                           fmt='.')
            if material == "rock":
                w = np.append(w, np.array(df.wavelengths))
                mat = np.append(mat, np.array(df.i_f))
                std = np.append(std, np.array(df["std"]))
                filters = len(df.wavelengths)
        else:
            if ax is not None: ax.errorbar(df.wavelengths, df.i_f, df["std"], capsize=3, capthick=0.4, color=BLACK,
                                           ecolor=BLACK,
                                           elinewidth=0.4,
                                           fmt='.')
            if material == "rock":
                w = np.append(w, np.array(df.wavelengths))
                mat = np.append(mat, np.array(df.i_f))
                std = np.append(std, np.array(df["std"]))
                filters = len(df.wavelengths)
    # print(w.shape, filters)
    w = w.reshape(w.shape[0] // filters, filters)
    mat = mat.reshape(mat.shape[0] // filters, filters)
    # print(w.shape)
    # print(mat.shape)
    w = w.mean(axis=0)
    mat = mat.mean(axis=0)
    return w, mat


def ref_rock(wavelength, phase_angle):
    # r_rock = np.load("data/rock.npy")
    # rock = interp2d(np.linspace(250, 1100, 100), np.linspace(0.1, 100, 100), r_rock)
    rock = hapke2(phase_angle, wavelength)
    return rock


def ref_ice(wavelength, phase_angle):
    # r_ice = np.load("ice.npy")
    # ice = interp2d(np.linspace(250, 1100, 100), np.linspace(0.1, 100, 100), r_ice)
    # return ice(wavelength, phase_angle)
    return 2 * ref_rock(wavelength, phase_angle)


def hapke_int(phase, w, slope, hs, bs, hc, bc, spsf_par):
    i = phase
    e = np.zeros(phase.shape)
    return hapke_ref(i, e, phase=phase, w=w, slope=slope, hs=hs, bs=bs, hc=hc, bc=bc,
                     spsf_par=[spsf_par], spsf_type=10)


def plot_data():
    wavelengths = np.linspace(250, 1000, 100)

    pa_ice = []
    pa_rock = []
    i_f_ice = []
    i_f_rock = []

    for material in ["ice", "rock"]:
        for phase_angle in [51, 58, 89, 92, "92b"]:
            filename = f"data/deshapriya/67p_{material}_alpha_{phase_angle}.csv"
            df = pd.read_csv(filename, names=["wavelength", "r"])
            reflectance = interp1d(df["wavelength"], df["r"], fill_value="extrapolate", kind='quadratic')
            if material == "ice":
                c = RED
                if phase_angle == "92b":
                    pa_ice.append(92)
                else:
                    pa_ice.append(phase_angle)
                i_f_ice.append(reflectance(649))
            else:
                c = BLACK
                if phase_angle == "92b":
                    pa_rock.append(92)
                else:
                    pa_rock.append(phase_angle)
                i_f_rock.append(reflectance(649))

            plt.scatter(df.wavelength, df.r, s=10, color=c, label="ice")
            plt.plot(wavelengths, reflectance(wavelengths), color=c, ls="--")
    plt.xlabel("wavelength [nm]")
    plt.ylabel("I/F")
    plt.ylim(0, 0.15)
    plt.title("different phase angles")
    plt.savefig("deshapriya.png")
    plt.show()

    phase_angles = np.arange(0.001, 90)
    i = phase_angles
    e = np.zeros(phase_angles.shape)
    r = hapke_int(i, e, phase_angles, omega, theta, h_s, b0_s, h_c, b0_c, [g], 10)

    plt.plot(phase_angles, r, color=BLACK, ls="--", label="hapke rock")
    plt.plot(phase_angles, 2 * r, color=RED, ls="--", label="hapke ice?")
    plt.scatter(pa_ice, i_f_ice, color=RED, label="ice")
    plt.scatter(pa_rock, i_f_rock, color=BLACK, label="rock")
    plt.xlabel("phase angle")
    plt.ylabel("I/F")
    plt.title(r"$\lambda$ = 643 nm")
    plt.legend()
    plt.savefig("deshapriya_phase.png")
    plt.show()


if __name__ == "__main__":

    phase_angles = np.linspace(0.01, 92, 100)
    i = phase_angles
    e = np.zeros(phase_angles.shape)

    phase11_rock = phase_clement(11, "rock")
    phase11_ice = phase_clement(11, "ice")
    phase51_rock = phase_clement(51, "rock")
    phase51_ice = phase_clement(51, "ice")

    plt.scatter([11] * len(phase11_rock), phase11_rock, color=BLACK, label=r"rock $\alpha=11$°")
    plt.scatter([51] * len(phase51_rock), phase51_rock, color=BLACK, label=r"rock $\alpha=51$°")
    plt.scatter([11] * len(phase11_ice), phase11_ice, color=RED, label=r"ice $\alpha=11$°")
    plt.scatter([51] * len(phase51_ice), phase51_ice, color=RED, label=r"ice $\alpha=51$°")
    r = hapke(phase_angles, 649)
    r_ice = hapke_ice(phase_angles)
    r_di = disk_int_hapke(phase_angles, 649)
    plt.plot(phase_angles, r, color=BLACK, ls="--", label="hapke rock")
    plt.plot(phase_angles, r_ice, color=RED, ls="--", label="hapke ice?")
    plt.plot(phase_angles, r_di, ls="-.", color=BLACK, label="disk int hapke")

    for material in ["ice", "rock"]:
        for phase_angle in [51, 58, 89, 92, "92b"]:
            filename = f"data/deshapriya/67p_{material}_alpha_{phase_angle}.csv"
            df = pd.read_csv(filename, names=["wavelength", "r"])
            if phase_angle == "92b": phase_angle = 92
            if material == "ice":
                c = RED
                plt.scatter([phase_angle] * len(df.r), df.r, s=10, marker="x", color=c)
                # plt.plot(phase_angles, 2 * hapke(phase_angles, 649), color=c, ls="--")

            else:
                c = BLACK
                plt.scatter([phase_angle] * len(df.r), df.r, s=10, marker="x", color=c)
                # plt.plot(phase_angles, hapke(phase_angles, 649), color=c, ls="--")

    a = mlines.Line2D([], [], color=BLACK, marker='x', ls='', label='rock deshapryia')
    b = mlines.Line2D([], [], color=RED, marker='x', ls='', label='ice deshapryia')
    c = mlines.Line2D([], [], color=BLACK, marker='o', ls='', label='rock clement')
    d = mlines.Line2D([], [], color=RED, marker='o', ls='', label='ice clement')
    # etc etc
    plt.legend(handles=[a, b, c, d])
    plt.xlabel("phase angle [°]")
    plt.ylabel("I/F")
    plt.title("649 nm")
    plt.savefig("plots/hapke_deshapryia_clement_phase.png")
    plt.show()

    fig, ax = plt.subplots(nrows=1, sharex=True)
    phase_angles = np.array([51, 89])
    phase_angle = 51
    plot_clement(ax, phase_angle)

    wavelengths = np.linspace(200, 1100)
    rock = hapke(phase_angles, wavelengths).T
    # ice = np.ones(wavelengths.shape) * hapke_ice(phase_angles)[:, None]
    r = disk_int_hapke(phase_angles, wavelengths).T
    # rock = hapke(phase_angle, wavelengths)
    # ice = 2 * hapke(phase_angle, wavelengths)
    ax.plot(wavelengths, rock, ls="--", color=BLACK, label="rock clement")
    # ax.plot(wavelengths, ice.T, ls="--", color=RED, label="ice clement")
    ax.plot(wavelengths, r, ls="-.", color=BLACK, label="disk int hapke")
    # ax.legend()
    # ax.set_title(f"phase angle={phase_angle}°")
    # ax.set_xlabel("wavelengths [nm]")
    # ax.set_ylabel("I/F")
    # plt.savefig("plots/hapke_clement_wavelenghts.png")
    # plt.show()

    for material in ["ice", "rock"]:
        for phase_angle in [51, 58, 89, 92, "92b"]:
            filename = f"data/deshapriya/67p_{material}_alpha_{phase_angle}.csv"
            df = pd.read_csv(filename, names=["wavelength", "r"])
            if phase_angle == "92b": phase_angle = 92
            if material == "ice":
                c = RED
                # ax.scatter(df.wavelength, df.r, s=10, marker="x", color=c)
                # ax.plot(wavelengths, hapke(phase_angle, wavelengths), color=c, ls="--")

            else:
                c = BLACK
                # ax.scatter(df.wavelength, df.r, s=10, marker="x", color=c)
                # ax.plot(wavelengths, hapke(phase_angle, wavelengths), color=c, ls="--")
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("I/F")
    ax.set_ylim(0, 0.15)
    ax.set_title("different phase angles")

    a = mlines.Line2D([], [], color=BLACK, marker='x', ls='', label='rock deshapryia')
    b = mlines.Line2D([], [], color=RED, marker='x', ls='', label='ice deshapryia')
    c = mlines.Line2D([], [], color=BLACK, marker='o', ls='', label='rock clement')
    d = mlines.Line2D([], [], color=RED, marker='o', ls='', label='ice clement')
    # etc etc
    plt.legend(handles=[a, b, c, d])
    plt.savefig("plots/hapke_deshapriya_clement.png")
    plt.show()
