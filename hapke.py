import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def E_1(x, theta):
    return np.exp(-2 / (np.pi * np.tan(theta) * np.tan(x)[:, None]))


def E_2(x, theta):
    return np.exp(-1 / (np.pi * np.tan(theta) ** 2 * np.tan(x)[:, None] ** 2))


def hapke_roughness(mu, mu0, alpha, slope):
    i = np.arccos(mu0)
    e = np.arccos(mu)
    e = np.ones(mu.shape) * 1e-7
    theta = slope
    psi = np.zeros(theta.shape)
    f = np.exp(-2 * np.tan(psi / 2))
    chi = 1 / np.sqrt(1 + np.pi * np.tan(theta) ** 2)

    eta_e = chi * (np.cos(e)[:, None] + np.sin(e)[:, None] * np.tan(theta) * (E_2(e, theta) / (2 - E_1(e, theta))))
    eta0_e = chi * (np.cos(i)[:, None] + np.sin(i)[:, None] * np.tan(theta) * (E_2(i, theta) / (2 - E_1(i, theta))))

    mu0_e = chi * (np.cos(i)[:, None] + np.sin(i)[:, None] * np.tan(theta) * (
            E_2(i, theta) - np.sin(psi / 2) ** 2 * E_2(e, theta)) / (
                           2 - E_1(i, theta) - psi / np.pi * E_1(e, theta)))
    mu_e = chi * (np.cos(e)[:, None] + np.sin(e)[:, None] * np.tan(theta) * (
            np.cos(psi) * E_2(i, theta) + np.sin(psi / 2) ** 2 * E_2(e, theta)) / (
                          2 - E_1(i, theta) - psi / np.pi * E_1(e, theta)))

    S = mu_e / eta_e * mu0[:, None] / eta0_e * chi / (1 - f + f * chi * (mu[:, None] / eta_e))
    output = {'sfun': S, 'imue': mu0_e, 'emue': mu_e}
    return output


def single_part_scat_func(alpha, g):
    # Single parameter henyey-Greenstein
    f = (1.0 - g ** 2) / ((1.0 + 2.0 * g * np.cos(alpha[:, None]) + g ** 2) ** 1.5)
    return (f)


def h_function(x, w):
    x = np.array(x)
    w = np.array(w)
    gamma = np.sqrt(1 - w)
    r0 = (1 - gamma) / (1 + gamma)
    f0 = 1. - (w * x) * (r0 + ((1. - (2. * r0 * x)) / 2.) * np.log((1. + x) / x))
    f = 1. / f0
    return (f)


def hapke(alpha, wavelength):
    df = pd.read_csv("data/hapke.csv")
    w_func = interp1d(df.wavelengths, df.w, fill_value="extrapolate", kind="linear")
    theta_func = interp1d(df.wavelengths, df.theta, fill_value="extrapolate", kind="linear")
    g_func = interp1d(df.wavelengths, df.g, fill_value="extrapolate", kind="linear")
    b0_func = interp1d(df.wavelengths, df.b0, fill_value="extrapolate", kind="linear")
    hs_func = interp1d(df.wavelengths, df.hs, fill_value="extrapolate", kind="linear")

    if type(alpha) != np.ndarray:
        alpha = [alpha]

    w = w_func(wavelength)
    theta = theta_func(wavelength)
    g = g_func(wavelength)
    b0 = b0_func(wavelength)
    hs = hs_func(wavelength)
    b0_s = b0
    omega = w

    alpha = np.radians(alpha)
    theta = np.radians(theta)

    slope = theta
    i = alpha
    if type(alpha) != np.ndarray:
        e = 0
    else:
        e = np.zeros(alpha.shape)

    mu0 = np.cos(i)
    mu = np.cos(e)

    rc = hapke_roughness(mu, mu0, alpha, slope)
    mu0 = rc['imue']
    mu = rc['emue']
    k = 1.

    k_opp = (1. / hs) * np.tan(alpha[:, None] / 2.)
    b_0 = 1. / (1. + k_opp)
    bsh = 1. + b0_s * b_0

    roughness_correction = rc['sfun']

    f = k * (omega / (4. * np.pi)) * (mu0 / (mu0 + mu)) * (
            single_part_scat_func(alpha, g) * bsh + h_function(mu0 / k, omega)
            * h_function(mu / k, omega) - 1.) * roughness_correction

    r_val = f
    iof_val = np.pi * r_val
    brdf_val = r_val / np.cos(i)[:, None]
    reff_val = np.pi * brdf_val

    f = iof_val
    f = np.squeeze(f)

    return f


def hapke_ice(alpha):
    if type(alpha) != np.ndarray:
        alpha = [alpha]

    w = 0.047
    theta = 15
    g = -0.335
    b0 = 2.38
    hs = 0.06
    b0_s = b0
    omega = w

    alpha = np.radians(alpha)
    theta = np.radians(theta)

    slope = theta
    i = alpha
    if type(alpha) != np.ndarray:
        e = 0
    else:
        e = np.zeros(alpha.shape)

    mu0 = np.cos(i)
    mu = np.cos(e)

    rc = hapke_roughness(mu, mu0, alpha, slope)
    mu0 = rc['imue']
    mu = rc['emue']
    k = 1.2

    k_opp = (1. / hs) * np.tan(alpha[:, None] / 2.)
    b_0 = 1. / (1. + k_opp)
    bsh = 1. + b0_s * b_0

    roughness_correction = rc['sfun']

    f = k * (omega / (4. * np.pi)) * (mu0 / (mu0 + mu)) * (
            single_part_scat_func(alpha, g) * bsh + h_function(mu0 / k, omega)
            * h_function(mu / k, omega) - 1.) * roughness_correction

    r_val = f
    iof_val = np.pi * r_val
    brdf_val = r_val / np.cos(i)[:, None]
    reff_val = np.pi * brdf_val

    f = iof_val
    f = np.squeeze(f)

    return f


if __name__ == "__main__":
    phase_angles = np.arange(1, 90)
    wavelengths = np.linspace(300, 1100, 1000)
    # plt.plot(phase_angles, hapke_int(phase_angles, 500))
    # plt.plot(wavelengths, hapke(phase_angles, 600))
    # plt.plot(wavelengths, hapke(phase_angles, 700))
    print(hapke(phase_angles, 700).shape)
    plt.plot(wavelengths, hapke(51, wavelengths).T, ls="--")
    # plt.plot(phase_angles, hapke(phase_angles, 600), ls="--")
    # plt.plot(phase_angles, hapke(phase_angles, 700), ls="--")
    plt.show()
