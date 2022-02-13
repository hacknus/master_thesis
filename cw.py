import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from unibe import *
from make_filters import make_filter
from scipy.interpolate import interp1d
from comet import ref_rock
import matplotlib.colors
from scipy.integrate import quad
from camera import Camera
import scipy.constants as const
from SNR import snr
from scipy.optimize import minimize, fsolve, leastsq
from motion_blurr import get_possible_detector_time

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


def integrand(w, alpha=0):
    return w * M(w) * Q(w) * ref_rock(w, alpha) * S(w)


if __name__ == "__main__":
    centers = np.arange(250, 1151, 50)

    alpha = 11
    signals = []

    coca = Camera()
    df = pd.read_csv("data/texp.csv")
    t = interp1d(df.alpha, df["texp10"], fill_value="extrapolate")
    t_exp = t(alpha) / (30 / 10) / 1000
    t_exp = get_possible_detector_time(t_exp)

    w = 100
    for c in centers:
        print(c)
        i = quad(integrand, c - w / 2, c + w / 2, args=(alpha))[
            0]
        signal = coca.A_Omega / coca.G * t_exp * i / (
                const.h * const.c * coca.r_h ** 2) * 1e-9
        signals.append(signal)
    signals = np.array(signals)
    plt.plot(centers, signals, BLACK, label="w=100nm")
    plt.ylabel("signals [DN]")
    plt.xlabel("centers")
    d = {
        "centers": centers,
        "signal": signals
    }
    df = pd.DataFrame(data=d)
    df.to_csv("data/cw.csv", index=False)
    plt.legend()
    plt.savefig("plots/cw.png")
    plt.show()
