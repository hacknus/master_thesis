import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import nquad
from hapke_model import hapke_ref
import pandas as pd
from scipy.interpolate import interp1d, interp2d
from unibe import *

omega = 0.042
g = -0.37
b0_s = 2.5
h_s = 0.079
b0_c = 0.188
h_c = 0.017
theta = 15.


def ref_rock(wavelength, phase_angle):
    r_rock = np.load("rock.npy")
    rock = interp2d(np.linspace(250, 1100, 100), np.linspace(0.1, 100, 100), r_rock)
    return rock(wavelength, phase_angle)


def ref_ice(wavelength, phase_angle):
    # r_ice = np.load("ice.npy")
    # ice = interp2d(np.linspace(250, 1100, 100), np.linspace(0.1, 100, 100), r_ice)
    # return ice(wavelength, phase_angle)
    return 2 * ref_rock(wavelength, phase_angle)


def hapke_int(i, e, phase, w, slope, hs, bs, hc, bc, spsf_par, spsf_type):
    return hapke_ref(i, e, phase=phase, w=w, slope=slope, hs=hs, bs=bs, hc=hc, bc=bc,
                     spsf_par=spsf_par, spsf_type=spsf_type)


def plot_data():
    wavelengths = np.linspace(250, 1000, 100)

    pa_ice = []
    pa_rock = []
    i_f_ice = []
    i_f_rock = []

    for material in ["ice", "rock"]:
        for phase_angle in [51, 58, 89, 92, "92b"]:
            filename = f"data/deshapriya/67p_{material}_alpha_{phase_angle}.csv"
            df = pd.read_csv(filename, names=["wavelength", "r"])
            reflectance = interp1d(df["wavelength"], df["r"], fill_value="extrapolate", kind='quadratic')
            if material == "ice":
                c = RED
                if phase_angle == "92b":
                    pa_ice.append(92)
                else:
                    pa_ice.append(phase_angle)
                i_f_ice.append(reflectance(649))
            else:
                c = BLACK
                if phase_angle == "92b":
                    pa_rock.append(92)
                else:
                    pa_rock.append(phase_angle)
                i_f_rock.append(reflectance(649))

            plt.scatter(df.wavelength, df.r, s=10, color=c, label="ice")
            plt.plot(wavelengths, reflectance(wavelengths), color=c, ls="--")
    plt.xlabel("wavelength [nm]")
    plt.ylabel("I/F")
    plt.ylim(0, 0.15)
    plt.title("different phase angles")
    plt.savefig("deshapriya.png")
    plt.show()

    phase_angles = np.arange(0.001, 90)
    i = phase_angles
    e = np.zeros(phase_angles.shape)
    r = hapke_int(i, e, phase_angles, omega, theta, h_s, b0_s, h_c, b0_c, [g], 10)

    plt.plot(phase_angles, r, color=BLACK, ls="--", label="hapke rock")
    plt.plot(phase_angles, 2*r, color=RED, ls="--", label="hapke ice?")
    plt.scatter(pa_ice, i_f_ice, color=RED, label="ice")
    plt.scatter(pa_rock, i_f_rock, color=BLACK, label="rock")
    plt.xlabel("phase angle")
    plt.ylabel("I/F")
    plt.title(r"$\lambda$ = 643 nm")
    plt.legend()
    plt.savefig("deshapriya_phase.png")
    plt.show()


if __name__ == "__main__":


    plot_data()
    exit()
    material = "ice"
    phase_angle = 58
    filename = f"data/deshapriya/67p_{material}_alpha_{phase_angle}.csv"
    df = pd.read_csv(filename, names=["wavelength", "r"])
    reflectance = interp1d(df["wavelength"], df["r"], fill_value="extrapolate", kind='quadratic')


    def normalized_reflectance(wavelength):
        return 3 * reflectance(wavelength) / reflectance(649)


    def ref(wavelength, phase_angles):
        if type(phase_angles) == np.ndarray:
            i = phase_angles
            e = np.zeros(phase_angles.shape)
            return hapke_int(i, e, phase_angles, omega, theta, h_s, b0_s, h_c, b0_c, [g], 10) * normalized_reflectance(
                wavelength)[:, None]
        else:
            i = phase_angles
            e = 0
            return hapke_int(i, e, phase_angles, omega, theta, h_s, b0_s, h_c, b0_c, [g], 10) * normalized_reflectance(
                wavelength)[:, None]


    wavelengths = np.linspace(250, 1100, 100)
    phase_angles = np.linspace(0.1, 100, 100)
    img = ref(wavelengths, phase_angles)
    plt.imshow(img)
    np.save(f"{material}.npy", img.T)
    plt.show()

    r_rock = np.load(f"{material}.npy")
    rock = interp2d(np.linspace(250, 1100, 100), np.linspace(0.1, 100, 100), r_rock)

    for material in ["ice", "rock"]:
        for phase_angle in [51, 58, 89, 92]:
            filename = f"data/deshapriya/67p_{material}_alpha_{phase_angle}.csv"
            df = pd.read_csv(filename, names=["wavelength", "r"])
            reflectance = interp1d(df["wavelength"], df["r"], fill_value="extrapolate", kind='quadratic')
            plt.title(f"alpha={phase_angle}°")
            if material == "ice":
                c = RED
                plt.scatter(df.wavelength, df.r, s=10, color=c, label="ice")
                plt.plot(wavelengths, rock(wavelengths, phase_angle), color=c, ls="--")
                plt.xlabel("wavelength [nm]")
                plt.ylabel("I/F")
                plt.ylim(0, 0.15)
                plt.show()
            else:
                c = "black"
                plt.scatter(df.wavelength, df.r, s=10, color=c, label="rock")
                plt.plot(wavelengths, rock(wavelengths, phase_angle), color=c, ls="--")
                plt.xlabel("wavelength [nm]")
                plt.ylabel("I/F")
                plt.ylim(0, 0.15)
                plt.show()
