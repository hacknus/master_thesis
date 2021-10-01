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


class Engine:

    def __init__(self, v, alpha):
        df = pd.read_csv("data/texp.csv")
        t = interp1d(df.alpha, df["texp10"], fill_value="extrapolate")
        self.t_exp = t(alpha) / (v / 10) / 1000
        self.coca = Camera()
        self.snr_target = 100
        self.alpha = alpha
        self.M = get_mirror()
        self.Q = get_detector()
        self.S = get_solar()
        df = pd.read_csv(f"data/widths.csv")
        widths = df.widths
        centers = df.c
        self.width = interp1d(centers, widths, kind="linear", fill_value="extrapolate")
        print(f"t_exp: {self.t_exp}")

    def integrand(self, w, n=4, alpha=0):
        return w * self.M(w) ** n * self.Q(w) * ref_rock(w, alpha) * self.S(w)

    def get_width_lower(self, center, edge):
        return center + self.width(center) / 2 - edge

    def get_width_upper(self, center, edge):
        return center - self.width(center) / 2 - edge

    # def run(self, seed=650):
    #     ora_width = self.width(seed)
    #     edge = seed - ora_width / 2
    #     blu_center = fsolve(self.get_width_lower, seed - ora_width, args=(edge,))[0]
    #     blu_width = self.width(blu_center)
    #     edge = seed + ora_width / 2
    #     red_center = fsolve(self.get_width_upper, seed + ora_width, args=(edge,))[0]
    #     red_width = self.width(red_center)
    #     edge = seed + ora_width / 2 + red_width
    #     nir_center = fsolve(self.get_width_upper, seed + ora_width + red_width, args=(edge,))[0]
    #     nir_width = self.width(nir_center)
    #
    #     print(f"BLUE: center = {blu_center:4.2f}, width = {blu_width:4.2f}")
    #     print(f"ORA: center = {seed:4.2f}, width = {ora_width:4.2f}")
    #     print(f"RED: center = {red_center:4.2f}, width = {red_width:4.2f}")
    #     print(f"NIR: center = {nir_center:4.2f}, width = {nir_width:4.2f}")

    def run(self, seed=650):
        n = 4

        def func(widths):
            blu_width, ora_width, red_width, nir_width = widths

            # BLU
            edge = seed - ora_width / 2
            i = quad(self.integrand, edge - blu_width, edge, args=(n, self.alpha))[
                0]
            blu_signal = self.coca.A_Omega / self.coca.G * self.t_exp * i / (
                    const.h * const.c * self.coca.r_h ** 2) * 1e-9

            # ORA
            i = quad(self.integrand, seed - ora_width / 2, seed + ora_width / 2, args=(n, self.alpha))[
                0]
            ora_signal = self.coca.A_Omega / self.coca.G * self.t_exp * i / (
                    const.h * const.c * self.coca.r_h ** 2) * 1e-9

            # RED
            edge = seed + ora_width / 2
            i = quad(self.integrand, edge, edge + red_width, args=(n, self.alpha))[
                0]
            red_signal = self.coca.A_Omega / self.coca.G * self.t_exp * i / (
                    const.h * const.c * self.coca.r_h ** 2) * 1e-9

            # NIR
            edge = seed + ora_width / 2 + red_width
            i = quad(self.integrand, edge, edge + nir_width, args=(n, self.alpha))[
                0]
            nir_signal = self.coca.A_Omega / self.coca.G * self.t_exp * i / (
                    const.h * const.c * self.coca.r_h ** 2) * 1e-9

            snr_diff_blu = self.snr_target - snr(blu_signal * self.coca.G)
            snr_diff_ora = self.snr_target - snr(ora_signal * self.coca.G)
            snr_diff_red = self.snr_target - snr(red_signal * self.coca.G)
            snr_diff_nir = self.snr_target - snr(nir_signal * self.coca.G)

            print(f"BLU snr = {snr(blu_signal * self.coca.G):.1f} width = {widths[0]:.1f}")
            print(f"ORA snr = {snr(ora_signal * self.coca.G):.1f} width = {widths[1]:.1f}")
            print(f"RED snr = {snr(red_signal * self.coca.G):.1f} width = {widths[2]:.1f}")
            print(f"NIR snr = {snr(nir_signal * self.coca.G):.1f} width = {widths[3]:.1f}")

            return snr_diff_blu + snr_diff_ora + snr_diff_red + snr_diff_nir

        sol = minimize(func, (150, 100, 100, 150))

        blu_width = sol.x[0]
        ora_width = sol.x[1]
        red_width = sol.x[2]
        nir_width = sol.x[3]

        blu_center = seed - ora_width / 2 - blu_width / 2
        ora_center = seed
        red_center = seed + ora_width / 2 + red_width / 2
        nir_center = seed + ora_width / 2 + red_width + nir_width / 2

        print(f"BLUE: center = {blu_center:4.2f}, width = {blu_width:4.2f}")
        print(f"ORA: center = {seed:4.2f}, width = {ora_width:4.2f}")
        print(f"RED: center = {red_center:4.2f}, width = {red_width:4.2f}")
        print(f"NIR: center = {nir_center:4.2f}, width = {nir_width:4.2f}")


if __name__ == "__main__":
    FilterCalculator = Engine(30, 11)
    FilterCalculator.run()
