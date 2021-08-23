import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad
import scipy.constants as const
import pandas as pd
from scipy.optimize import fsolve
from comet import ref_rock, ref_ice
from camera import Camera
from unibe import *
from make_filters import make_filter, init_filters_thomas


# solar spectra: https://www.pveducation.org/pvcdrom/appendices/standard-solar-spectra


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


def get_filter_from_data():
    df_filter = pd.read_csv("data/filter_BLU_final.txt", delimiter=",")
    F = interp1d(df_filter.wavelength, df_filter.transmission / 100, fill_value="extrapolate")
    return F


def make_filters_from_source(coca, initial=690, alpha=0):
    # centers, widths_up, widths_low = solve_for_widths(coca)
    # centers = np.arange(450, 1000, 50)
    # widths_low = np.array(
    #     [140.20811959, 69.80984747, 55.64778604, 56.76834028, 67.61291976, 70.6574801, 64.04351599, 61.0122352,
    #      69.82912625, 85.58533316, 115.04105122])
    # widths_up = np.array(
    #     [251.17422176, 155.77892878, 115.4626612, 116.97686207, 130.64200627, 136.28019314, 129.22874446,
    #      127.13557219, 140.69100047, 172.73874097, 237.95641359])
    #
    df = pd.read_csv(f"data/widths_alpha_{alpha}.csv")
    widths_low = df.wl
    widths_up = df.wu
    centers = df.c
    widths_avg = widths_low + (widths_up - widths_low) / 2
    width = interp1d(centers, widths_up, kind="quadratic", fill_value="extrapolate")
    width_std = interp1d(centers, (widths_up - widths_low) / 2, kind="quadratic", fill_value="extrapolate")

    # first filter:
    c1 = initial
    w1 = width(initial)
    w1_std = width_std(initial)

    def func(center):
        return center - width(center) / 2 - (c1 + w1 / 2)

    c2 = fsolve(func, c1 + w1)
    print(c2)
    c2 = c2[0]
    w2 = width(c2)
    w2_std = width_std(c2)

    def func(center):
        return center - width(center) / 2 - (c2 + w2 / 2)

    c3 = fsolve(func, c2 + w2)
    print(c3)
    c3 = c3[0]
    w3 = width(c3)
    w3_std = width_std(c3)

    def func(center):
        return center + width(center) / 2 - (c1 - w1 / 2)

    c0 = fsolve(func, c1 - 2 * w1)
    print(c0)
    c0 = c0[0]
    w0 = width(c0)
    w0_std = width_std(c0)

    print(f"filter BLU: c={c0:.2f}, w={w0:.2f} +/- {w0_std:.2f}")
    print(f"filter ORANGE: c={c1:.2f}, w={w1:.2f} +/- {w1_std:.2f}")
    print(f"filter RED: c={c2:.2f}, w={w2:.2f} +/- {w2_std:.2f}")
    print(f"filter NIR: c={c3:.2f}, w={w3:.2f} +/- {w3_std:.2f}")

    F0 = make_filter(c0, w0)
    F1 = make_filter(c1, w1)
    F2 = make_filter(c2, w2)
    F3 = make_filter(c3, w3)

    fig, axes = plt.subplots(nrows=3, sharex=True)

    wavelengths = np.linspace(300, 1100, 1000)

    axes[1].plot(wavelengths, F0(wavelengths), color=BLUE)
    axes[1].plot(wavelengths, F1(wavelengths), color=ORANGE)
    axes[1].plot(wavelengths, F2(wavelengths), color=RED)
    axes[1].plot(wavelengths, F3(wavelengths), color=BLACK)

    F0, F1, F2, F3 = init_filters_thomas()
    axes[2].plot(wavelengths, F0(wavelengths), color=BLUE)
    axes[2].plot(wavelengths, F1(wavelengths), color=ORANGE)
    axes[2].plot(wavelengths, F2(wavelengths), color=RED)
    axes[2].plot(wavelengths, F3(wavelengths), color=BLACK)

    axes[0].plot(centers, widths_low + (widths_up - widths_low) / 2,
                 color=RED, alpha=0.5)
    axes[0].fill_between(centers, widths_low, widths_up, color=RED, alpha=0.5)

    axes[0].scatter(c0, w0, label="BLUE", marker="x", color=BLUE)
    axes[0].scatter(c1, w1, label="ORANGE", marker="x", color=ORANGE)
    axes[0].scatter(c2, w2, label="RED", marker="x", color=RED)
    axes[0].scatter(c3, w3, label="NIR", marker="x", color=BLACK)

    w1 = 150
    w2 = 100
    w3 = 100
    w4 = 150
    c1 = 460
    c2 = 650
    c3 = 750
    c4 = 900
    axes[0].scatter(c1, w1, label="BLUE (NT)", color=BLUE)
    axes[0].scatter(c2, w2, label="ORANGE (NT)", color=ORANGE)
    axes[0].scatter(c3, w3, label="RED (NT)", color=RED)
    axes[0].scatter(c4, w4, label="NIR (NT)", color=BLACK)

    axes[2].set_xlabel("filter center [nm]")
    axes[1].set_ylabel("filter calculated")
    axes[2].set_ylabel("filter thomas")
    axes[0].set_ylabel("filter width [nm]")
    axes[0].set_title(f"t_exp ={CoCa.t_exp}s alpha={alpha}°")
    axes[0].legend()
    plt.savefig("plots/filter_widths_1_2.png")
    plt.show()


def make_filters_from_source2(coca, initial=690, alpha=0):
    # centers, widths_up, widths_low = solve_for_widths(coca)
    # centers = np.arange(450, 1000, 50)
    # widths_low = np.array(
    #     [140.20811959, 69.80984747, 55.64778604, 56.76834028, 67.61291976, 70.6574801, 64.04351599, 61.0122352,
    #      69.82912625, 85.58533316, 115.04105122])
    # widths_up = np.array(
    #     [251.17422176, 155.77892878, 115.4626612, 116.97686207, 130.64200627, 136.28019314, 129.22874446,
    #      127.13557219, 140.69100047, 172.73874097, 237.95641359])
    #
    df = pd.read_csv(f"data/widths_alpha_{alpha}.csv")
    widths_low = df.wl
    widths_up = df.wu
    centers = df.c
    widths_avg = widths_low + (widths_up - widths_low) / 2
    width = interp1d(centers, widths_up, kind="quadratic", fill_value="extrapolate")
    width_std = interp1d(centers, (widths_up - widths_low) / 2, kind="quadratic", fill_value="extrapolate")
    centers = np.linspace(400, 1000, 100)

    # first filter:
    c2 = initial
    w2 = width(initial)
    w2_std = width_std(initial)

    def func(center):
        return center - width(center) / 2 - (c2 + w2 / 2)

    c3 = fsolve(func, c2 + w2)
    print(c3)
    c3 = c3[0]
    w3 = width(c3)
    w3_std = width_std(c3)

    def func(center):
        return center + width(center) / 2 - (c2 - w2 / 2)

    c1 = fsolve(func, c2 - w2)
    print(c3)
    c1 = c1[0]
    w1 = width(c1)
    w1_std = width_std(c1)

    def func(center):
        return center + width(center) / 2 - (c1 - w1 / 2)

    c0 = fsolve(func, c1 - 2 * w1)
    print(c0)
    c0 = c0[0]
    w0 = width(c0)
    w0_std = width_std(c0)

    print(f"filter BLU: c={c0:.2f}, w={w0:.2f} +/- {w0_std:.2f}")
    print(f"filter GREEN: c={c1:.2f}, w={w1:.2f} +/- {w1_std:.2f}")
    print(f"filter ORANGE: c={c2:.2f}, w={w2:.2f} +/- {w2_std:.2f}")
    print(f"filter RNIR: c={c3:.2f}, w={w3:.2f} +/- {w3_std:.2f}")

    F0 = make_filter(c0, w0)
    F1 = make_filter(c1, w1)
    F2 = make_filter(c2, w2)
    F3 = make_filter(c3, w3)

    fig, axes = plt.subplots(nrows=3, sharex=True)

    wavelengths = np.linspace(300, 1100, 1000)

    axes[1].plot(wavelengths, F0(wavelengths), color=BLUE)
    axes[1].plot(wavelengths, F1(wavelengths), color=GREEN)
    axes[1].plot(wavelengths, F2(wavelengths), color=ORANGE)
    axes[1].plot(wavelengths, F3(wavelengths), color=RED)

    F0, F1, F2, F3 = init_filters_thomas()
    axes[2].plot(wavelengths, F0(wavelengths), color=BLUE)
    axes[2].plot(wavelengths, F1(wavelengths), color=ORANGE)
    axes[2].plot(wavelengths, F2(wavelengths), color=RED)
    axes[2].plot(wavelengths, F3(wavelengths), color=BLACK)

    axes[0].plot(centers, width(centers) - width_std(centers),
                 color=RED, alpha=0.5)
    axes[0].fill_between(centers, width(centers) - 2 * width_std(centers), width(centers), color=RED, alpha=0.5)

    axes[0].scatter(c0, w0, label="BLUE", marker="x", color=BLUE)
    axes[0].scatter(c1, w1, label="GREEN", marker="x", color=GREEN)
    axes[0].scatter(c2, w2, label="ORANGE", marker="x", color=ORANGE)
    axes[0].scatter(c3, w3, label="RNIR", marker="x", color=RED)

    w1 = 150
    w2 = 100
    w3 = 100
    w4 = 150
    c1 = 460
    c2 = 650
    c3 = 750
    c4 = 900
    axes[0].scatter(c1, w1, label="BLUE (NT)", color=BLUE)
    axes[0].scatter(c2, w2, label="ORANGE (NT)", color=ORANGE)
    axes[0].scatter(c3, w3, label="RED (NT)", color=RED)
    axes[0].scatter(c4, w4, label="NIR (NT)", color=BLACK)

    axes[2].set_xlabel("filter center [nm]")
    axes[1].set_ylabel("filter calculated")
    axes[2].set_ylabel("filter thomas")
    axes[0].set_ylabel("filter width [nm]")
    axes[0].set_title(f"t_exp ={CoCa.t_exp}s alpha={alpha}°")
    axes[0].legend()
    plt.savefig("plots/filter_widths_2_1.png")
    plt.show()


def solve_for_widths(coca, alpha=0):
    M = get_mirror()
    Q = get_detector()
    S = get_solar()

    def integrand_low(w, N=4, alpha=0):
        return w * M(w) ** N * Q(w) * ref_rock(w, alpha) * S(w)

    def integrand_up(w, N=4, alpha=0):
        return w * M(w) ** N * Q(w) * ref_ice(w, alpha) * S(w)

    phase_angle = np.arange(0, 100, 5)

    N = 4
    widths_up = []
    widths_low = []
    centers = range(450, 1000, 50)

    # plt.plot(centers, ref_rock(centers, alpha), color=BLACK, label="rock")
    # plt.plot(centers, ref_ice(centers, alpha), color=RED, label="ice")
    # filename = f"data/deshapriya/67p_rock_alpha_{alpha}.csv"
    # df = pd.read_csv(filename, names=["wavelength", "r"])
    # plt.scatter(df.wavelength, df.r, marker="x", color=BLACK)
    # filename = f"data/deshapriya/67p_ice_alpha_{alpha}.csv"
    # df = pd.read_csv(filename, names=["wavelength", "r"])
    # plt.scatter(df.wavelength, df.r, marker="x", color=RED)
    #
    # plt.ylabel("I/F")
    # plt.xlabel("wavelengths")
    # plt.legend()
    # plt.show()
    # exit()

    for filter_center in centers:
        def func(width):
            i = quad(integrand_up, filter_center - width / 2, filter_center + width / 2, args=(N, alpha))[
                0]
            signal = CoCa.A_Omega / CoCa.G * CoCa.t_exp * i / (const.h * const.c * CoCa.r_h ** 2) * 1e-9
            return signal - 2 ** 14

        sol = fsolve(func, 100)
        print(filter_center, sol)
        widths_low.append(sol[0])

        def func(width):
            i = quad(integrand_low, filter_center - width / 2, filter_center + width / 2, args=(N, alpha))[
                0]
            signal = CoCa.A_Omega / CoCa.G * CoCa.t_exp * i / (const.h * const.c * CoCa.r_h ** 2) * 1e-9
            return signal - 2 ** 14

        sol = fsolve(func, 100)
        print(filter_center, sol)
        widths_up.append(sol[0])
    widths_up = np.array(widths_up)
    widths_low = np.array(widths_low)
    data = {"c": centers,
            "wu": widths_up,
            "wl": widths_low
            }
    df = pd.DataFrame(data=data)
    df.to_csv(f"data/widths_alpha_{alpha}.csv", index=False)
    return


def plot_widths(centers, widths_up, widths_low):
    plt.plot(centers, widths_low + (widths_up - widths_low) / 2,
             color=RED, alpha=0.5)
    plt.fill_between(centers, widths_low, widths_up, color=RED, alpha=0.5)
    w1 = 150
    w2 = 100
    w3 = 100
    w4 = 150
    c1 = 460
    c2 = 650
    c3 = 750
    c4 = 900
    plt.scatter(c1, w1, label="BLUE", color=BLUE)
    plt.scatter(c2, w2, label="ORANGE", color=ORANGE)
    plt.scatter(c3, w3, label="RED", color=RED)
    plt.scatter(c4, w4, label="NIR", color=BLACK)
    plt.xlabel("filter center [nm]")
    plt.ylabel("filter width [nm]")
    plt.title(f"t_exp ={CoCa.t_exp}")
    plt.savefig("plots/filter_widths.png")
    plt.show()


if __name__ == "__main__":
    CoCa = Camera()
    phase_angle = 51
    t_exp = 0.025
    CoCa.t_exp = t_exp
    solve_for_widths(CoCa, alpha=phase_angle)
    make_filters_from_source2(CoCa, alpha=phase_angle)

    # plot_widths(c, w1, w2)
