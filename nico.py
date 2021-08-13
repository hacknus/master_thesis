import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.integrate import quad
import scipy.constants as const
import pandas as pd
from make_filters import make_filter, init_filters, init_filters_thomas
from scipy.optimize import dual_annealing, fsolve
from reflectance import get_ref
from comet import ref_rock, ref_ice
from camera import Camera
from unibe import *


c1 = 595.16
w1 = 86.05
c2 = 690.00
w2 = 103.63
c3 = 788.60
w3 = 93.57
c4 = 899.94
w4 = 129.12


def get_mirror():
    df_mirror = pd.read_csv("data/mirrors_transmission.txt", delimiter="\s")
    M = interp1d(df_mirror.wavelength, df_mirror.transmission, fill_value="extrapolate")
    # percent
    return M


def get_detector():
    df_qe = pd.read_csv("data/qe.txt", delimiter=",")
    Q = interp1d(df_qe.Wavelength, df_qe.QE / 100, fill_value="extrapolate")
    # electrons per photons
    return Q


def get_solar():
    df_solar = pd.read_csv("data/solar.csv", delimiter=";", skiprows=1)
    S = interp1d(df_solar["Wavelength (nm)"], df_solar["Extraterrestrial W*m-2*nm-1"], fill_value="extrapolate")
    # W per meter squared per nanometer
    return S


if __name__ == "__main__":
    M = get_mirror()
    Q = get_detector()
    S = get_solar()

    CoCa = Camera()

    def integrand(w, N=4, alpha=0):
        return w * M(w) ** N * Q(w) * ref_rock(w, alpha) * S(w)


    phase_angle = np.arange(0, 100, 5)

    N = 4
    t = []
    centers = range(450, 1000, 50)

    for alpha in phase_angle:
        def func(t_exp):
            i = quad(integrand, c2 - w2 / 2, c2 + w2 / 2, args=(N, alpha))[0]
            signal = CoCa.A_Omega / CoCa.G * t_exp * i / (const.h * const.c * CoCa.r_h ** 2) * 1e-9
            return signal - 2 ** 14


        sol = fsolve(func, 0.001)
        print(alpha, sol)
        t.append(sol[0])

    wavelengths = np.linspace(300, 1100, 100)
    fig, axes = plt.subplots(nrows=2)

    axes[0].plot(phase_angle, t, color=BLACK, ls="--")
    axes[0].set_xlabel("phase angle [°]")
    axes[0].set_ylabel("exposure time [s]")

    axes[1].plot(wavelengths, ref_rock(wavelengths, phase_angle).T, color=BLACK, ls="--")
    axes[1].set_xlabel("wavelength [nm]")
    axes[1].set_ylabel("I/F")

    plt.show()
