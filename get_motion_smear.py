import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from camera import Camera
from unibe import *
from scipy.integrate import quad
from comet import ref_rock
from scipy.interpolate import interp1d
import scipy.constants as const
from SNR import snr
from scipy.optimize import fsolve
import matplotlib.lines as lines


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


def integrand(w, N=4, alpha=0):
    return w * M(w) ** N * Q(w) * ref_rock(w, alpha).T * S(w)


def omega(t, v, b):
    return (v / b) / (1 + (v / b * t) ** 2) / np.pi * 180


def main(mode="A", r_h=1.0, b=1000):
    relative_velocities = [10, 30, 80]
    CoCa = Camera()
    CoCa.r_h = r_h
    theta = np.sqrt(CoCa.Omega) / np.pi * 180
    N = 4
    phase_angles = np.arange(0, 70, 10)
    df = pd.read_csv(f"data/filters_{mode}.csv")
    colors = [BLUE, ORANGE, RED, BLACK]
    filters = ["BLU", "ORA", "RED", "NIR"]
    centers = df.centers
    widths = df.widths
    fig, axes = plt.subplots(nrows=2, sharex=True)

    print(theta)
    d = {}

    for v, ls in zip(relative_velocities, ["-", "-.", "--"]):
        print(f"calculating for v = {v} km/s")
        t = np.linspace(0, 300, 100)
        o = omega(t, v, b)
        print("omegas init")
        print(o)
        alpha = np.zeros(t.shape)
        for i in range(1, len(t), 1):
            alpha[i] = alpha[i - 1] + (t[i] - t[i - 1]) * o[i]
        print("alphas init")
        print(alpha)
        timestamp = interp1d(alpha, t, fill_value="extrapolate")
        d_sub = {}
        for filter_center, filter_width, color, filter in zip(centers, widths, colors, filters):
            texps = []
            omegas = []
            for ti, alpha in zip(timestamp(phase_angles), phase_angles):
                def func(t_exp):
                    i = quad(integrand, filter_center - filter_width / 2, filter_center + filter_width / 2,
                             args=(N, alpha))[0]
                    signal = CoCa.A_Omega / CoCa.G * t_exp * i / (const.h * const.c * CoCa.r_h ** 2) * 1e-9
                    return snr(signal * CoCa.G) - 100

                sol = fsolve(func, 0.0001)[0]
                texps.append(sol)
                omegas.append(omega(ti, v, b))
            texps = np.array(texps)
            omegas = np.array(omegas)
            print("t exp:")
            print(texps)
            print("omegas:")
            print(omegas)
            print("t exp * omega:")
            print(texps * omegas)
            print("theta")
            print(theta)
            smear = interp1d(phase_angles, omegas * texps / theta, fill_value="extrapolate", kind="quadratic")
            exposure_time = interp1d(phase_angles, texps * 1000, fill_value="extrapolate", kind="quadratic")
            phase_angles_cont = np.linspace(0, 60, 100)
            axes[0].plot(phase_angles_cont, smear(phase_angles_cont), color=color, ls=ls)
            axes[1].plot(phase_angles_cont, exposure_time(phase_angles_cont), color=color, ls=ls)
            d_sub[filter] = texps * 1000
        d[v] = d_sub
    l1 = lines.Line2D([], [], color='black', ls="-")
    l2 = lines.Line2D([], [], color='black', ls="-.")
    l3 = lines.Line2D([], [], color='black', ls="--")
    axes[0].legend(handles=[l1, l2, l3], labels=["v = 10 km/s", "v = 30 km/s", "v = 80 km/s"], fancybox=True,
                   framealpha=1,
                   shadow=True, borderpad=1)
    axes[1].set_xlabel("phase angle [°]")
    axes[1].set_ylabel(r"$t_{exp}$ [ms]")
    axes[0].set_ylabel("# pixels")
    df = pd.DataFrame(data=d)
    df.to_csv(f"data/texps_SNR100_{r_h}au_{mode}.csv", index=False)
    plt.savefig(f"plots/motion_smear_{r_h}au_{mode}.pdf")
    plt.show()


if __name__ == "__main__":
    main(mode="A", r_h=0.85)
