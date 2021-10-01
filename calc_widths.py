import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad
import scipy.constants as const
import pandas as pd
from scipy.optimize import fsolve, minimize
from comet import ref_rock, ref_ice
from camera import Camera
from unibe import *
from make_filters import make_filter, init_filters_thomas
from SNR import snr

snr_target = 100


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


def solve_for_widths(coca, alpha=0):
    M = get_mirror()
    Q = get_detector()
    S = get_solar()

    def integrand(w, N=4, alpha=0):
        return w * M(w) ** N * Q(w) * ref_rock(w, alpha) * S(w)

    N = 4
    widths100 = []
    widths80 = []
    widths60 = []
    centers = range(400, 1050, 50)

    for filter_center in centers:
        def func(width):
            i = quad(integrand, filter_center - width / 2, filter_center + width / 2, args=(N, alpha))[
                0]
            signal = coca.A_Omega / coca.G * coca.t_exp * i / (const.h * const.c * coca.r_h ** 2) * 1e-9
            print(f"snr: {snr(signal * coca.G):4.2f}")
            return 100 - snr(signal * coca.G)

        sol = fsolve(func, 100)
        print(filter_center, sol)
        widths100.append(sol[0])

        def func(width):
            i = quad(integrand, filter_center - width / 2, filter_center + width / 2, args=(N, alpha))[
                0]
            signal = coca.A_Omega / coca.G * coca.t_exp * i / (const.h * const.c * coca.r_h ** 2) * 1e-9
            print(f"snr: {snr(signal * coca.G):4.2f}")
            return 80 - snr(signal * coca.G)

        sol = fsolve(func, 100)
        print(filter_center, sol)
        widths80.append(sol[0])

        def func(width):
            i = quad(integrand, filter_center - width / 2, filter_center + width / 2, args=(N, alpha))[
                0]
            signal = coca.A_Omega / coca.G * coca.t_exp * i / (const.h * const.c * coca.r_h ** 2) * 1e-9
            print(f"snr: {snr(signal * coca.G):4.2f}")
            return 60 - snr(signal * coca.G)

        sol = fsolve(func, 100)
        print(filter_center, sol)
        widths60.append(sol[0])
    widths100 = np.array(widths100)
    widths80 = np.array(widths80)
    widths60 = np.array(widths60)
    print(widths100)
    print(widths80)
    print(widths60)
    data = {"c": centers,
            "widths_100": widths100,
            "widths_80": widths80,
            "widths_60": widths60,
            }
    df = pd.DataFrame(data=data)
    df.to_csv(f"data/widths_snr.csv", index=False)
    return


def main(v=30, phase_angle=11):
    CoCa = Camera()
    df = pd.read_csv("data/texp.csv")

    t10 = interp1d(df.alpha, df["texp10"], fill_value="extrapolate")
    t80 = interp1d(df.alpha, df["texp80"], fill_value="extrapolate")
    t_exp = (t80(phase_angle) + t10(phase_angle)) / 2
    t_exp = t10(phase_angle) / (v / 10)
    t_exp /= 1000
    CoCa.t_exp = t_exp
    print(t_exp)
    solve_for_widths(CoCa, alpha=phase_angle)

    c = np.linspace(400, 1000, 100)
    df = pd.read_csv(f"data/widths_snr_{v}.csv")
    widths100_avg = df.widths_100
    widths80_avg = df.widths_80
    widths60_avg = df.widths_60
    centers = df.c
    width100 = interp1d(centers, widths100_avg, kind="quadratic", fill_value="extrapolate")
    width80 = interp1d(centers, widths80_avg, kind="quadratic", fill_value="extrapolate")
    width60 = interp1d(centers, widths60_avg, kind="quadratic", fill_value="extrapolate")
    plt.plot(c, width100(c), label="SNR 100")
    plt.plot(c, width80(c), label="SNR 80")
    plt.plot(c, width60(c), label="SNR 60")
    plt.xlabel("center [nm]")
    plt.ylabel("width [nm]")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
