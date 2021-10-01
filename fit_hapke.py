import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
from unibe import *

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
from unibe import *


def E_1(x, theta):
    return np.exp(-2 / (np.pi * np.tan(theta) * np.tan(x)[:, None]))


def E_2(x, theta):
    return np.exp(-1 / (np.pi * np.tan(theta) ** 2 * np.tan(x)[:, None] ** 2))


def hapke_roughness(mu, mu0, slope, alpha):
    """
    This is the hapke roughness function as described by hapke for the case of
    e <= i. Psi = phase angle.
    :param mu: cos(e)
    :type mu: numpy array
    :param mu0: cos(i)
    :type mu0: numpy array
    :param slope: theta
    :type slope: numpy array
    :param alpha: phase angle
    :type alpha: numpy array
    :return: roughness correction, mu' and mu0'
    :rtype: dictionary
    """
    i = np.arccos(mu0)
    e = np.arccos(mu)
    e[e == 0] = 1e-7
    theta = slope
    psi = alpha  # np.zeros(alpha.shape)
    f = np.exp(-2 * np.tan(psi[:, None] / 2))
    chi = 1 / np.sqrt(1 + np.pi * np.tan(theta) ** 2)

    eta_e = chi * (np.cos(e)[:, None] + np.sin(e)[:, None] * np.tan(theta) * (E_2(e, theta) / (2 - E_1(e, theta))))
    eta0_e = chi * (np.cos(i)[:, None] + np.sin(i)[:, None] * np.tan(theta) * (E_2(i, theta) / (2 - E_1(i, theta))))

    mu0_e = chi * (np.cos(i)[:, None] + np.sin(i)[:, None] * np.tan(theta) * (
            E_2(i, theta) - np.sin(psi[:, None] / 2) ** 2 * E_2(e, theta)) / (
                           2 - E_1(i, theta) - psi[:, None] / np.pi * E_1(e, theta)))
    mu_e = chi * (np.cos(e)[:, None] + np.sin(e)[:, None] * np.tan(theta) * (
            np.cos(psi[:, None]) * E_2(i, theta) + np.sin(psi[:, None] / 2) ** 2 * E_2(e, theta)) / (
                          2 - E_1(i, theta) - psi[:, None] / np.pi * E_1(e, theta)))

    S = mu_e / eta_e * mu0[:, None] / eta0_e * chi / (1 - f + f * chi * (mu[:, None] / eta_e))
    output = {'sfun': S, 'imue': mu0_e, 'emue': mu_e}
    return output


def single_part_scat_func(alpha, g):
    # Single parameter henyey-Greenstein
    f = (1.0 - g ** 2) / ((1.0 + 2.0 * g * np.cos(alpha[:, None]) + g ** 2) ** 1.5)
    return f


def h_function(x, w):
    x = np.array(x)
    w = np.array(w)
    gamma = np.sqrt(1 - w)
    r0 = (1 - gamma) / (1 + gamma)
    f0 = 1. - (w * x) * (r0 + ((1. - (2. * r0 * x)) / 2.) * np.log((1. + x) / x))
    f = 1. / f0
    return (f)


def hapke(alpha, i, e, wavelength):
    w = 0.034
    theta = 28
    g = -0.42
    b0 = 2.25
    hs = 0.061
    b0_s = b0
    omega = w
    k = 1.2

    theta = np.radians(theta)

    slope = theta
    mu0 = np.cos(i)
    mu = np.cos(e)

    rc = hapke_roughness(mu, mu0, slope, alpha)
    mu0 = rc['imue'][0]
    mu = rc['emue'][0]

    k_opp = (1. / hs) * np.tan(alpha[:, None] / 2.)
    b_0 = 1. / (1. + k_opp)
    bsh = 1. + b0_s * b_0

    roughness_correction = rc['sfun']

    f = k * omega / (4. * np.pi) * (mu0 / (mu0 + mu)) * (
            single_part_scat_func(alpha, g) * bsh + h_function(np.cos(i)[:, None] / k, omega * k)
            * h_function(np.cos(e)[:, None] / k, omega * k) - 1.) * roughness_correction

    r_val = f
    iof_val = np.pi * r_val
    brdf_val = r_val / np.cos(i)[:, None]
    reff_val = np.pi * brdf_val

    f = iof_val

    ref = np.array([0.007365, 0.007775, 0.01095, 0.01209, 0.01483, 0.016885, 0.017385, 0.01799,
                    0.0193, 0.02068, 0.021645])
    wavelengths = np.array([269, 360, 481, 536, 649, 701, 744, 805, 882, 932, 989])
    ref = interp1d(df.wavelengths, ref, fill_value="extrapolate")

    return f / ref(649) * ref(wavelength)


def wave(ax, phase_angle=51):
    path = f"data/clement/phase{phase_angle}/"
    files = os.listdir(path)
    for filename in files:
        df = pd.read_csv(path + filename)
        if "br" in filename:
            ax.errorbar(df.wavelengths, df.i_f, df["std"], capsize=3, capthick=0.4, color=RED,
                        ecolor=RED,
                        elinewidth=0.4,
                        fmt='.')
        if "ur" in filename:
            ax.errorbar(df.wavelengths, df.i_f, df["std"], capsize=3, capthick=0.4, color=BLACK,
                        ecolor=BLACK,
                        elinewidth=0.4,
                        fmt='.')
    for material in ["ice", "rock"]:
        filename = f"data/deshapriya/67p_{material}_alpha_{phase_angle}.csv"
        df = pd.read_csv(filename, names=["wavelength", "r"])
        if material == "ice":
            c = RED
            ax[1].scatter(df.wavelength, df.r, marker="o", color=c)

        else:
            c = BLACK
            ax[1].scatter(df.wavelength, df.r, marker="o", color=c)


if __name__ == "__main__":
    wavelengths = np.array([269, 360, 481, 536, 649, 701, 744, 805, 882, 932, 989])
    phase_angle = 11
    path = f"data/clement/phase{phase_angle}/"
    files = os.listdir(path)
    for filename in files:
        df = pd.read_csv(path + filename)
        if "ur" not in filename:
            continue
        print(filename)
        plt.scatter(11, df.i_f[df.wavelengths == 649], color=BLACK)
    phase_angle = 51
    path = f"data/clement/phase{phase_angle}/"
    files = os.listdir(path)
    for filename in files:
        df = pd.read_csv(path + filename)
        if "ur" not in filename:
            continue
        print(filename)
        plt.scatter(51, df.i_f[df.wavelengths == 649], color=BLACK)
    emission_angles = [0.746, 0.903, 0.440, 0.945, 0.903, 0.616]
    incidence_angles = [0.229, 1.039, 0.460, 0.751, 0.134, 1.035]
    phase_angles = [0.897, 0.888, 0.894, 0.887, 0.905, 0.886]
    color = [RED,ORANGE]
    for c, i, e, alpha in zip(color, incidence_angles[1:3], emission_angles[1:3], phase_angles[1:3]):
        i = np.array([i])
        e = np.array([e])
        alpha = np.array([alpha])
        plt.scatter(51, hapke(alpha, i, e, 649), ls="--",
                    label=f"i={np.degrees(i[0]):.1f}, e={np.degrees(e[0]):.1f}")
        phase_angles = np.radians(np.linspace(1, 90, 100))
        print(e, alpha, i)
        print(e, alpha - i)
        print(i, alpha - e)
        plt.plot(np.degrees(phase_angles), hapke(phase_angles, i, phase_angles - i, 649),color=c, ls="--", label="i fixed")
        plt.plot(np.degrees(phase_angles), hapke(phase_angles, e, phase_angles - e, 649),color=c, ls="-.", label="e fixed")
    plt.ylabel("I/F")
    plt.xlabel("phase angle")
    plt.legend()
    plt.savefig("envelope.png")
    plt.show()

    for filename in files:
        df = pd.read_csv(path + filename)
        if "ur" not in filename:
            continue
        plt.scatter(df.wavelengths, df.i_f, color=BLACK)

    emission_angles = [0.746, 0.903, 0.440, 0.945, 0.903, 0.616]
    incidence_angles = [0.229, 1.039, 0.460, 0.751, 0.134, 1.035]
    phase_angles = [0.897, 0.888, 0.894, 0.887, 0.905, 0.886]
    for i, e, alpha in zip(incidence_angles[1:3], emission_angles[1:3], phase_angles[1:3]):
        i = np.array([i])
        e = np.array([e])
        alpha = np.array([alpha])
        for w in wavelengths:
            plt.scatter(w, hapke(alpha, i, e, w), ls="--", color=RED)
    plt.legend()
    plt.show()
