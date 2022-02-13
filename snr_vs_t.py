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


def plot_snr(r_h=1, mode="A", ice=False, alpha=0):
    t_exps = np.linspace(0.22, 15, 10) / 1000
    relative_velocities = [10, 30, 80]
    CoCa = Camera()
    CoCa.r_h = r_h
    df = pd.read_csv("data/texp.csv")
    t10 = interp1d(df.alpha, df["texp10"], fill_value="extrapolate")
    df = pd.read_csv(f"data/filters_{mode}.csv")
    colors = [BLUE, ORANGE, RED, BLACK]
    centers = df.centers
    widths = df.widths
    for filter_center, filter_width, color in zip(centers, widths, colors):
        snrs = []
        for t_exp in t_exps:
            i = quad(integrand, filter_center - filter_width / 2, filter_center + filter_width / 2,
                     args=(alpha, ice))[0]
            signal = CoCa.A_Omega / CoCa.G * t_exp * i / (const.h * const.c * CoCa.r_h ** 2) * 1e-9
            snrs.append(snr(signal * CoCa.G))
        print(snrs)
        t_func = interp1d(snrs,t_exps, fill_value="extrapolate", kind="quadratic")
        snr_cont = np.linspace(0, 180, 100)
        plt.plot(snr_cont, t_func(snr_cont)*1000, color=color, ls="-")

    plt.ylabel("t_exp [ms]")
    plt.xlabel("SNR")
    plt.savefig(f"plots/snr_vs_t_{r_h}au_{mode}_ice={ice}_{alpha}deg.pdf")
    plt.show()


if __name__ == "__main__":
    plot_snr(mode="A", alpha=80)
