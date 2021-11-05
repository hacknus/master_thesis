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


M = get_mirror()
Q = get_detector()
S = get_solar()


def integrand(w, alpha=0):
    return w * M(w) * Q(w) * ref_rock(w, alpha) * S(w)


class Solver:

    def __init__(self, v=30):
        self.df = pd.read_csv("data/cw.csv")
        self.coca = Camera()
        df = pd.read_csv("data/texp.csv")
        t = interp1d(df.alpha, df["texp10"], fill_value="extrapolate")
        self.v = v
        self.t_exp = t(11) / 1000 / (self.v / 10)  # cw.csv is calculated for alpha = 11 deg
        self.s = interp1d(self.df.centers, self.df.signal, fill_value="extrapolate", kind="quadratic")
        self.snr_target = 100

    def signal(self, center, width, t_exp):
        return np.max([0, self.s(center) / 100 * width / self.t_exp * t_exp])

    def run(self, mode="A"):
        def func(widths):
            blu_width, ora_width, red_width, nir_width, seed = widths
            t_exp = self.t_exp
            if mode != "C":
                # BLU
                blu_center = seed - ora_width / 2 - blu_width / 2
                blu_signal = self.signal(blu_center, blu_width, t_exp)

                # ORA
                ora_center = seed
                ora_signal = self.signal(ora_center, ora_width, t_exp)

                # RED
                red_center = seed + ora_width / 2 + red_width / 2
                red_signal = self.signal(red_center, red_width, t_exp)

                # NIR
                nir_center = seed + ora_width / 2 + red_width + nir_width / 2
                nir_signal = self.signal(nir_center, nir_width, t_exp)
            else:
                # BLU
                blu_center = seed - red_width / 2 - ora_width - blu_width / 2
                blu_signal = self.signal(blu_center, blu_width, t_exp)

                # ORA
                ora_center = seed - red_width / 2 - ora_width / 2
                ora_signal = self.signal(ora_center, ora_width, t_exp)

                # RED
                red_center = seed
                red_signal = self.signal(red_center, red_width, t_exp)

                # NIR
                nir_center = seed + red_width / 2 + nir_width / 2
                nir_signal = self.signal(nir_center, nir_width, t_exp)

            snr_diff_blu = self.snr_target - snr(blu_signal * self.coca.G)
            snr_diff_ora = self.snr_target - snr(ora_signal * self.coca.G)
            snr_diff_red = self.snr_target - snr(red_signal * self.coca.G)
            snr_diff_nir = self.snr_target - snr(nir_signal * self.coca.G)

            print(f"BLU snr = {snr(blu_signal * self.coca.G):.1f} width = {widths[0]:.1f}")
            print(f"ORA snr = {snr(ora_signal * self.coca.G):.1f} width = {widths[1]:.1f}")
            print(f"RED snr = {snr(red_signal * self.coca.G):.1f} width = {widths[2]:.1f}")
            print(f"NIR snr = {snr(nir_signal * self.coca.G):.1f} width = {widths[3]:.1f}")
            print(f"total bandwidth = {np.sum(widths[:4]):.1f}")

            widths_delta = (1100 - 400) - np.sum(widths[:4])
            return np.max([snr_diff_blu, snr_diff_ora, snr_diff_red, snr_diff_nir]) + widths_delta ** 2

        if mode == "A":
            bounds = ((0, 250), (0, 250), (0, 250), (0, 250), (650, 651))
            seed = 650
        elif mode == "B":
            bounds = ((0, 250), (0, 250), (0, 250), (0, 250), (500, 650))
            seed = 650
        else:
            bounds = ((0, 250), (0, 250), (0, 250), (0, 250), (650, 650.1))
            seed = 650
        sol = minimize(func, (150, 100, 100, 150, seed), bounds=bounds)

        blu_width = sol.x[0]
        ora_width = sol.x[1]
        red_width = sol.x[2]
        nir_width = sol.x[3]
        seed = sol.x[4]

        if mode != "C":
            blu_center = seed - ora_width / 2 - blu_width / 2
            ora_center = seed
            red_center = seed + ora_width / 2 + red_width / 2
            nir_center = seed + ora_width / 2 + red_width + nir_width / 2
        else:
            blu_center = seed - red_width / 2 - ora_width - blu_width / 2
            ora_center = seed - red_width / 2 - ora_width / 2
            red_center = seed
            nir_center = seed + red_width / 2 + nir_width / 2

        print(f"BLUE: center = {blu_center:4.2f}, width = {blu_width:4.2f}")
        print(f"ORA: center = {ora_center:4.2f}, width = {ora_width:4.2f}")
        print(f"RED: center = {red_center:4.2f}, width = {red_width:4.2f}")
        print(f"NIR: center = {nir_center:4.2f}, width = {nir_width:4.2f}")
        print(f"t_exp = {self.t_exp * 1000:.2f} ms")

        centers = [blu_center, ora_center, red_center, nir_center]
        widths = [blu_width, ora_width, red_width, nir_width]

        d = {
            "centers": centers,
            "widths": widths
        }
        df = pd.DataFrame(data=d)
        df.to_csv(f"data/filters_{mode}.csv", index=False)

        return
        phase_angles = np.arange(1, 90, 10)
        snrs = np.zeros((len(phase_angles), 4))
        k = 0
        df = pd.read_csv("data/texp.csv")
        t = interp1d(df.alpha, df["texp10"], fill_value="extrapolate")
        for alpha in phase_angles:
            t_exp = t(alpha) / (self.v / 10) / 1000
            j = 0
            for c, w in zip(centers, widths):
                i = quad(integrand, c - w / 2, c + w / 2, args=(alpha))[
                    0]
                signal = coca.A_Omega / coca.G * t_exp * i / (
                        const.h * const.c * coca.r_h ** 2) * 1e-9
                snrs[k, j] = snr(signal * coca.G)
                j += 1
            k += 1
        plt.plot(phase_angles, snrs[:, 0], label="BLU", color=BLUE)
        plt.plot(phase_angles, snrs[:, 1], label="ORA", color=ORANGE)
        plt.plot(phase_angles, snrs[:, 2], label="RED", color=RED)
        plt.plot(phase_angles, snrs[:, 3], label="NIR", color=BLACK)
        plt.legend()
        plt.title(rf"$v$={self.v} km/s")
        plt.ylabel("SNR [-]")
        plt.xlabel(r"$\alpha$ [°]")
        plt.savefig("plots/snrs.pdf")
        plt.show()

        # wavelengths = np.linspace(350, 1100, 1000)
        # F0 = make_filter(blu_center, blu_width)
        # F1 = make_filter(ora_center, ora_width)
        # F2 = make_filter(red_center, red_width)
        # F3 = make_filter(nir_center, nir_width)
        # plt.plot(wavelengths, 100 * F0(wavelengths), color=BLUE)
        # plt.plot(wavelengths, 100 * F1(wavelengths), color=ORANGE)
        # plt.plot(wavelengths, 100 * F2(wavelengths), color=RED)
        # plt.plot(wavelengths, 100 * F3(wavelengths), color=BLACK)
        # plt.plot(wavelengths, np.zeros(wavelengths.shape), color=BLACK)
        # plt.show()


if __name__ == "__main__":
    Sol = Solver()
    Sol.run("B")
