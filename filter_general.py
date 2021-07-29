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


def get_reflectance():
    mat_low = np.load("mat_low.npy")
    ref_low = interp2d(np.arange(0, 100), np.linspace(300, 1100, 100), mat_low)
    mat_up = np.load("mat_up.npy")
    ref_up = interp2d(np.arange(0, 100), np.linspace(300, 1100, 100), mat_up)
    # percent
    return ref_low, ref_up


def plot_signals():
    M = get_mirror()
    Q = get_detector()
    S = get_solar()
    F = get_filter_from_data()
    F, F2, F3, F4 = init_filters(400, [200, 200, 200, 200])
    F, F2, F3, F4 = init_filters_thomas()
    ref_low, ref_up = get_reflectance()

    def integrand_low(w, N=4, alpha=0):
        return w * M(w) ** N * Q(w) * ref_low(alpha, w)[0] * S(w)

    def integrand_up(w, N=4, alpha=0):
        return w * M(w) ** N * Q(w) * ref_up(alpha, w)[0] * S(w)

    phase_angle = np.arange(0, 100, 5)
    signals_low = np.zeros(phase_angle.shape)
    signals_up = np.zeros(phase_angle.shape)
    signals_std = np.zeros(phase_angle.shape)
    N = 4
    for i, alpha in enumerate(phase_angle):
        print(i, alpha)
        integral_low, int_err = quad(integrand_low, 300, 1100, args=(N, alpha))
        integral_up, int_err = quad(integrand_up, 300, 1100, args=(N, alpha))

        widths_up = []
        widths_low = []
        centers = range(450, 1000, 50)
        for filter_center in centers:
            def func(width):
                G = 2.5  # electron per DN
                A = (7e-6) ** 2
                Omega = 0.64e-10
                r_h = 1  # A.U.
                t_exp = 0.001  # seconds
                i = quad(integrand_up, filter_center - width / 2, filter_center + width / 2, args=(N, alpha))[
                    0]
                signal = G * A * Omega * t_exp * i / (const.h * const.c) / r_h ** 2
                return signal - 2 ** 14

            sol = fsolve(func, 100)
            print(filter_center, sol)
            widths_up.append(sol[0])

            def func(width):
                G = 2.5  # electron per DN
                A = (7e-6) ** 2
                Omega = 0.64e-10
                r_h = 1  # A.U.
                t_exp = 0.001  # seconds
                i = quad(integrand_low, filter_center - width / 2, filter_center + width / 2, args=(N, alpha))[
                    0]
                signal = G * A * Omega * t_exp * i / (const.h * const.c) / r_h ** 2
                return signal - 2 ** 14

            sol = fsolve(func, 100)
            print(filter_center, sol)
            widths_low.append(sol[0])
        widths_up = np.array(widths_up)
        widths_low = np.array(widths_low)
        plt.plot(centers, widths_low + (widths_up - widths_low) / 2,
                 color="#e6002e", alpha=0.5)
        plt.fill_between(centers, widths_low, widths_up, color="#e6002e", alpha=0.5)
        w1 = 150
        w2 = 100
        w3 = 100
        w4 = 150
        c1 = 460
        c2 = 550
        c3 = 750
        c4 = 900
        plt.scatter(c1, w1, label="BLUE", color="#4767af")
        plt.scatter(c2, w2, label="GREEN", color="#466553")
        plt.scatter(c3, w3, label="RED", color="#e6002e")
        plt.scatter(c4, w4, label="NIR", color="black")
        plt.xlabel("filter center [nm]")
        plt.ylabel("filter width [nm]")
        plt.savefig("filter_widths.png")
        plt.show()

        wavelengths = np.linspace(300, 1100, 1000)
        plt.plot(wavelengths, integrand_low(wavelengths) + (integrand_up(wavelengths) - integrand_low(wavelengths)) / 2,
                 color="#e6002e", alpha=0.5)
        plt.fill_between(wavelengths, integrand_low(wavelengths), integrand_up(wavelengths), color="#e6002e", alpha=0.5)
        plt.show()
        exit()

        G = 2.5  # electron per DN
        A = 0.135 ** 2 * 4 * np.pi
        A = (7e-6) ** 2
        Omega = 0.64e-10
        r_h = 1  # A.U.
        t_exp = 0.0001  # seconds
        signal_low = G * A * Omega * t_exp * integral_low / (const.h * const.c) / r_h ** 2
        signal_up = G * A * Omega * t_exp * integral_up / (const.h * const.c) / r_h ** 2
        # TODO: calculate with uncertainties of other parameters
        signals_low[i] = signal_low
        signals_up[i] = signal_up

    fig, axes = plt.subplots(nrows=6, sharex=True)

    wavelengths = np.linspace(300, 1100, 1000)

    axes[0].plot(wavelengths, 100 * Q(wavelengths), color="black")
    axes[0].set_ylabel(r"$Q$ [%]")

    axes[1].plot(wavelengths, 100 * M(wavelengths), color="black")
    axes[1].set_ylabel(r"$M$ [%]")

    axes[2].plot(wavelengths, 100 * F(wavelengths), color="#4767af")  # blue
    axes[2].plot(wavelengths, 100 * F2(wavelengths), color="#466553") # green
    axes[2].plot(wavelengths, 100 * F3(wavelengths), color="#e6002e") # red
    axes[2].plot(wavelengths, 100 * F4(wavelengths), color="black")
    axes[2].set_ylabel(r"$T$ [%]")

    axes[3].plot(wavelengths,
                 100 * (ref_low(0, wavelengths) + (ref_up(0, wavelengths) - ref_low(0, wavelengths)) / 2),
                 color="#e6002e")
    axes[3].fill_between(wavelengths, 100 * ref_low(0, wavelengths).T[0], 100 * ref_up(0, wavelengths).T[0],
                         color="#e6002e",
                         alpha=0.5)
    axes[3].set_ylabel(r"$R$ [%]")
    axes[3].legend()

    axes[4].plot(wavelengths, S(wavelengths), color="black", label=r"$F_\odot$")
    axes[4].set_ylabel(r"$F_\odot$ [Wm$^{-2}$nm$^{-1}$]")

    axes[5].plot(wavelengths, M(wavelengths) ** 4 * ref_low(0, wavelengths)[0] * S(wavelengths),
                 color="#e6002e",
                 label=r"$F_\odot(\omega)M(\omega)^NR(\omega)$")
    axes[5].set_ylabel(r"$F_\odot'$ [Wm$^{-2}$nm$^{-1}$]")
    axes[5].set_xlabel(r"$\lambda$ [nm]")

    plt.savefig("filters.png")
    plt.show()

    plt.plot(phase_angle, signals_low + (signals_up - signal_low) / 2, color="#e6002e")
    plt.fill_between(phase_angle, signals_low, signals_up, color="#e6002e", alpha=0.5)
    plt.fill_between(phase_angle, 0, 8 * 2.5, color="black", alpha=0.5)
    plt.axhline(2 ** 14, color="#e6002e", ls="--")
    plt.xlabel(r"$\alpha$ [°]")
    plt.ylabel(r"$S$ [DN]")
    plt.title(fr"$t$ = {t_exp} s")
    plt.savefig("signal.png")
    plt.show()


def get_filter_from_data():
    df_filter = pd.read_csv("data/filter_BLU_final.txt", delimiter=",")
    F = interp1d(df_filter.wavelength, df_filter.transmission / 100, fill_value="extrapolate")
    return F


def calc_signal(center=400, w1=200, w2=200, w3=200, w4=200, maxima=[0.98, 0.98, 0.98, 0.98], alpha=0):
    M = get_mirror()
    Q = get_detector()
    S = get_solar()
    F1, F2, F3, F4 = init_filters(center, [w1, w2, w3, w4], maxima)
    ref = get_reflectance()
    N = 4
    G = 2.5  # electron per DN
    A = (7e-6) ** 2
    Omega = 0.64e-10
    r_h = 1  # A.U.
    t_exp = 0.001  # seconds

    signals = []
    for F in [F1, F2, F3, F4]:
        def integrand(w, N=4, alpha=0):
            return w * M(w) ** N * F(w) * Q(w) * ref(alpha, w)[0] * S(w)

        integral, int_err = quad(integrand, 300, 1100, args=(N, alpha))
        signal = G * A * Omega * t_exp * integral / (const.h * const.c) / r_h ** 2
        signals.append(signal)
    return np.array(signals)


if __name__ == "__main__":
    plot_signals()
