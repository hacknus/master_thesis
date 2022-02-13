import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.lines as mlines
from scipy.integrate import quad
from comet import ref_rock, ref_ice
from hapke import hapke, hapke_ice, disk_int_hapke, hapke_scaled
from hapke_antoine import hapke_ref
import scipy.constants as const
import pandas as pd
from scipy.interpolate import interp1d, interp2d
from camera import Camera
from unibe import *
from SNR import snr
from calc_widths import main as main_widths
from filter_selector import main as main_filters
from solver import Solver
from reflectance_plot import main as main_reflectance
from motion_blurr import get_possible_detector_time


def plot_reflectance():
    main_reflectance()


def get_filters(mode="A", v=30):
    Sol = Solver(v)
    Sol.run(mode)


def plot_widths(v=30, alpha=11):
    main_widths(v, alpha)


def plot_filters(mode="A", v=30, alpha=11):
    main_filters(mode, v, alpha)


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


M = get_mirror()
Q = get_detector()
S = get_solar()


def integrand(w, alpha=0, ice=False):
    if ice:
        return w * M(w) * Q(w) * ref_ice(w, alpha).T * S(w)
    else:
        return w * M(w) * Q(w) * ref_rock(w, alpha).T * S(w)


def get_snr(r_h=1, alpha=11, v=30, mode="A", ice=False):
    CoCa = Camera()
    CoCa.r_h = r_h
    df = pd.read_csv("data/texp.csv")
    t10 = interp1d(df.alpha, df["texp10"], fill_value="extrapolate")
    df = pd.read_csv(f"data/filters_{mode}.csv")
    colors = [BLUE, ORANGE, RED, BLACK]
    centers = df.centers
    widths = df.widths
    print(f"calculating for mode = {mode}, v = {v} km/s, r_h = {r_h} a.u. and ice = {ice}, alpha = {alpha}")
    for filter_center, filter_width, color in zip(centers, widths, colors):
        t_exp = t10(alpha) / 1000 / (v / 10)
        t_exp = get_possible_detector_time(t_exp)
        i = quad(integrand, filter_center - filter_width / 2, filter_center + filter_width / 2,
                 args=(alpha, ice))[0]
        signal = CoCa.A_Omega / CoCa.G * t_exp * i / (const.h * const.c * CoCa.r_h ** 2) * 1e-9
        print(f"center = {filter_center:.1f}, width = {filter_width:.1f}, SNR = {snr(signal * CoCa.G):.1f}")


def plot_snr(r_h=1, mode="A", ice=False):
    relative_velocities = [10, 30, 80]
    CoCa = Camera()
    CoCa.r_h = r_h
    phase_angles = np.arange(0, 90, 10)
    df = pd.read_csv("data/texp.csv")
    t10 = interp1d(df.alpha, df["texp10"], fill_value="extrapolate")
    df = pd.read_csv(f"data/filters_{mode}.csv")
    colors = [BLUE, ORANGE, RED, BLACK]
    centers = df.centers
    widths = df.widths
    for v, ls in zip(relative_velocities, ["-", "-.", "--"]):
        print(f"calculating for v = {v} km/s")
        for filter_center, filter_width, color in zip(centers, widths, colors):
            snrs = []
            t_exp = t10(11) / 1000 / (v / 10)
            t_exp = get_possible_detector_time(t_exp)
            i = quad(integrand, filter_center - filter_width / 2, filter_center + filter_width / 2,
                     args=(11, ice))[0]
            signal = CoCa.A_Omega / CoCa.G * t_exp * i / (const.h * const.c * CoCa.r_h ** 2) * 1e-9
            print(f"v = {v}, center = {filter_center:.2f}, SNR = {snr(signal * CoCa.G):.2f}")
            for alpha in phase_angles:
                t_exp = t10(alpha) / 1000 / (v / 10)
                i = quad(integrand, filter_center - filter_width / 2, filter_center + filter_width / 2,
                         args=(alpha, ice))[0]
                signal = CoCa.A_Omega / CoCa.G * t_exp * i / (const.h * const.c * CoCa.r_h ** 2) * 1e-9
                snrs.append(snr(signal * CoCa.G))
            snrs_func = interp1d(phase_angles, snrs, fill_value="extrapolate", kind="quadratic")
            phase_angles_cont = np.linspace(0, 90, 200)
            plt.plot(phase_angles_cont, snrs_func(phase_angles_cont), color=color, ls=ls)
    l1 = lines.Line2D([], [], color='black', ls="-")
    l2 = lines.Line2D([], [], color='black', ls="-.")
    l3 = lines.Line2D([], [], color='black', ls="--")
    plt.legend(handles=[l1, l2, l3], labels=["v = 10 km/s", "v = 30 km/s", "v = 80 km/s"], fancybox=True, framealpha=1,
               shadow=True, borderpad=1)
    plt.xlabel("phase angle [°]")
    plt.ylabel("SNR")
    plt.savefig(f"plots/snrs_{r_h}au_{mode}_ice={ice}_new_new.pdf")
    plt.show()


if __name__ == "__main__":
    mode = "A"
    v = 30
    alpha = 11
    # plot_reflectance()
    # get_filters("A", v)
    # get_filters("B", v)
    # get_filters("C", v)
    get_filters("D", v)
    # plot_widths(v, alpha)
    # plot_filters("A", v, alpha)
    # plot_filters("B", v, alpha)
    # plot_filters("C", v, alpha)
    plot_filters("D", v, alpha)

    # plot_snr(mode="A")
    # plot_snr(mode="A", ice=True)
    # plot_snr(mode="B")
    # plot_snr(mode="B", ice=True)
    # plot_snr(mode="C")
    # plot_snr(mode="C", ice=True)
    plot_snr(mode="D")

    # get_snr(mode="A")
    # get_snr(mode="A", ice=True)
    # get_snr(mode="B")
    # get_snr(mode="B", ice=True)
    # get_snr(mode="C")
    get_snr(mode="D")
