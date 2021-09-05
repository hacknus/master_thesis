import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


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
    psi = alpha  # np.zeros(theta.shape)
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
    """
    This is the hapke model (antoine adaptation)

    :param alpha: phase angle
    :type alpha: numpy array
    :param wavelength: wavelength
    :type wavelength: numpy array
    :return: I/F
    :rtype: numpy array
    """
    df = pd.read_csv("data/fournasier_hapke.csv")
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

    rc = hapke_roughness(mu, mu0, slope, alpha)
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
    """
    This is hapke model with fixed parameters described by hasslemann et al (2017) on page S565

    :param alpha: phase angle
    :type alpha: numpy array
    :return: I/F
    :rtype: numpy array
    """
    if type(alpha) != np.ndarray:
        alpha = [alpha]

    w = 0.047
    theta = 15
    g = -0.335
    b0 = 2.38
    hs = 0.06
    b0_s = b0
    omega = w
    k = 1.198

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

    rc = hapke_roughness(mu, mu0, slope, alpha)
    mu0 = rc['imue']
    mu = rc['emue']

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


def disk_int_hapke(alpha, wavelength):
    """
    This is the disk integrated hapke model described by fournasier et. al. (2015) on page 5.

    :param alpha: phase angle
    :type alpha: numpy array
    :param wavelength: wavelength
    :type wavelength: numpy array
    :return: I/F
    :rtype: numpy array
    """
    df = pd.read_csv("data/fournasier_hapke.csv")
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

    alpha = np.radians(alpha)
    theta = np.radians(theta)

    r = (1 - np.sqrt(1 - w)) / (1 + np.sqrt(1 - w))
    k_opp = (1. / hs) * np.tan(alpha[:, None] / 2.)
    b_0 = 1. / (1. + k_opp)
    bsh = 1. + b0_s * b_0

    K = np.exp(
        -0.32 * theta * np.sqrt(np.tan(theta) * np.tan(alpha[:, None] / 2)) - 0.52 * theta * np.tan(theta) * np.tan(
            alpha[:, None] / 2))

    return K * ((w / 8 * (bsh * single_part_scat_func(alpha, g) - 1) + r / 2 * (1 - r)) * (
            1 - np.sin(alpha[:, None] / 2) * np.tan(alpha[:, None] / 2) * np.log(
        1 / np.tan(alpha[:, None] / 4))) + 2 / (
                        3 * np.pi) * r ** 2 * (
                        np.sin(alpha[:, None]) + (np.pi - alpha[:, None]) * np.cos(alpha[:, None])))


if __name__ == "__main__":
    phase_angles = np.arange(1, 90)
    wavelengths = np.linspace(300, 1100, 1000)
    plt.plot(phase_angles, hapke(phase_angles, 649), label="hapke")
    plt.plot(phase_angles, disk_int_hapke(phase_angles, 649), label="disk integrated hapke")
    plt.plot(phase_angles, hapke_ice(phase_angles), label="hapke ice")
    plt.xlabel("phase angle [°]")
    plt.ylabel("I/F")
    plt.legend()
    plt.show()

    plt.plot(wavelengths, hapke(11, wavelengths).T, ls="--", label="hapke")
    plt.plot(wavelengths, disk_int_hapke(11, wavelengths).T, ls="--", label="disk integrated hapke")
    plt.plot(wavelengths, hapke_ice(np.ones(wavelengths.shape)*11).T, ls="--", label="hapke ice")
    plt.xlabel("wavelength [nm]")
    plt.ylabel("I/F")
    plt.legend()
    plt.show()