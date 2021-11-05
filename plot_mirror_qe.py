import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from unibe import *


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
    wavelengths = np.linspace(300, 1050, 1000)
    plt.plot(wavelengths, 100 * M(wavelengths), color=BLACK)
    plt.plot(wavelengths, 100 * M(wavelengths)**4, color=BLACK, ls="--")
    plt.plot(wavelengths, 100 * M(wavelengths)**0.25, color=BLACK, ls=":")
    plt.xlabel("wavelength [nm]")
    plt.ylabel("reflection [%]")
    plt.savefig("plots/mirrors.pdf")
    plt.show()

    plt.plot(wavelengths, 100*Q(wavelengths), color=BLACK)
    plt.xlabel("wavelength [nm]")
    plt.ylabel("QE [%]")
    plt.savefig("plots/qe.pdf")
    plt.show()

    plt.plot(wavelengths, S(wavelengths), color=BLACK)
    plt.xlabel("wavelength [nm]")
    plt.ylabel(r"solar flux [Wm$^{-2}$nm$^{-1}$]")
    plt.savefig("plots/solar.pdf")
    plt.show()
