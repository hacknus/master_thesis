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

read_out_noise = 5  # electrons
dark_current_noise = 20  # electrons / second
full_well_capacity = 33000  # electrons
peak_linear_charge = 27000  # electrons


def snr(s):
    return s / (np.sqrt(s) )# + read_out_noise + dark_current_noise)


c1 = 595.16
w1 = 86.05
c2 = 690.00
w2 = 103.63
c3 = 788.60
w3 = 93.57
c4 = 899.94
w4 = 129.12

filter_center = 650
filter_width = 100


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

    print(snr(27000))

    CoCa = Camera()


    def integrand(w, N=4, alpha=0):
        return w * M(w) ** N * Q(w) * ref_rock(w, alpha).T * S(w)


    phase_angle = np.arange(1, 90, 10)

    N = 4
    snr_vals = []
    snr_vals_80 = []
    signals = []
    signals_80 = []
    centers = range(450, 1000, 50)
    t_exp = 0.005
    t = []
    t_sat = []
    df = pd.read_csv("data/texp.csv")

    t10 = interp1d(df.alpha, df["texp10"], fill_value="extrapolate")
    t80 = interp1d(df.alpha, df["texp80"], fill_value="extrapolate")
    for alpha in phase_angle:
        i = quad(integrand, filter_center - filter_width / 2, filter_center + filter_width / 2, args=(N, alpha))[0]
        t_exp = t10(alpha)/1000
        signal = CoCa.A_Omega / CoCa.G * t_exp * i / (const.h * const.c * CoCa.r_h ** 2) * 1e-9
        snr_vals.append(snr(signal * CoCa.G))
        signals.append(signal)

        t_exp = t80(alpha)/1000
        signal = CoCa.A_Omega / CoCa.G * t_exp * i / (const.h * const.c * CoCa.r_h ** 2) * 1e-9
        snr_vals_80.append(snr(signal * CoCa.G))
        signals_80.append(signal)

        def func(t_exp):
            i = quad(integrand, filter_center - filter_width / 2, filter_center + filter_width / 2, args=(N, alpha))[0]
            signal = CoCa.A_Omega / CoCa.G * t_exp * i / (const.h * const.c * CoCa.r_h ** 2) * 1e-9
            return snr(signal * CoCa.G) - 100


        sol = fsolve(func, 0.001)
        print(alpha, sol)
        t.append(sol[0])


        def func(t_exp):
            i = quad(integrand, filter_center - filter_width / 2, filter_center + filter_width / 2, args=(N, alpha))[0]
            signal = CoCa.A_Omega / CoCa.G * t_exp * i / (const.h * const.c * CoCa.r_h ** 2) * 1e-9
            return snr(signal * CoCa.G) - 164.3


        sol = fsolve(func, 0.001)
        print(alpha, sol)
        t_sat.append(sol[0])

    wavelengths = np.linspace(300, 1100, 100)
    fig, axes = plt.subplots(nrows=4)

    axes[0].plot(phase_angle, np.array(t) * 1000, color=BLACK, ls="-", label="SNR 100")
    axes[0].plot(phase_angle, np.array(t_sat) * 1000, color=RED, ls="-", label="saturation")
    axes[0].set_xlabel("phase angle [°]")
    axes[0].set_ylabel(r"$t_{exp}$ [ms]")
    axes[0].legend()

    axes[1].plot(phase_angle, snr_vals, color=RED, ls="-")
    axes[1].plot(phase_angle, snr_vals_80, color=BLACK, ls="-")
    axes[1].axhline(snr(CoCa.G * 2 ** 14), ls="--", color=RED, label="saturation")
    axes[1].axhline(np.sqrt(full_well_capacity), ls="--", color=ORANGE, label="full well")
    axes[1].axhline(np.sqrt(peak_linear_charge), ls="--", color=GREEN, label="peak linear")
    axes[1].legend()
    axes[1].set_xlabel("phase angle [°]")
    axes[1].set_ylabel("SNR")

    axes[2].plot(phase_angle, signals, color=RED, ls="-")
    axes[2].plot(phase_angle, signals_80, color=BLACK, ls="-")
    axes[2].axhline(2 ** 14, ls="--", color=RED)
    axes[2].set_xlabel("phase angle [°]")
    axes[2].set_ylabel("signal [DN]")

    axes[3].plot(wavelengths, ref_rock(wavelengths, phase_angle).T, color=BLACK, ls="-")
    axes[3].set_xlabel("wavelength [nm]")
    axes[3].set_ylabel("I/F")

    plt.savefig("plots/snr.png")

    plt.show()
