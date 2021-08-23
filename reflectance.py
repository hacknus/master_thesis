import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import nquad
import pandas as pd

from scipy.interpolate import interp1d


# This is Table 4, "Hapke 2002" line from Fornasier et al. 2015:
# Results from the Hapke (2002, and 2012) modeling from disk-resolved
# images taken with the NAC orange filter centered on 649 nm.

def get_ref(phase_angle, i=None, e=None):
    omega = 0.042
    g = -0.37
    b0_s = 2.5
    h_s = 0.079
    b0_c = 0.188
    h_c = 0.017
    theta = 15.
    r = []
    r_err = []
    if type(phase_angle) is np.ndarray:
        if i is not None and e is not None:
            for alpha, ii, ee in zip(phase_angle, i, e):
                r_i = hapke_int(ii, ee, alpha, omega, theta, h_s, b0_s, h_c, b0_c, [g], 10)
                r_err_i = 0
                r.append(r_i)
                r_err.append(r_err_i)
        else:
            for alpha in phase_angle:
                r_i, r_err_i = nquad(hapke_int, [[0, np.pi / 2], [0, np.pi]],
                                     args=(alpha, omega, theta, h_s, b0_s, h_c, b0_c,
                                           [g], 10))
                r.append(r_i)
                r_err.append(r_err_i)
        return np.array(r), np.array(r_err)
    else:
        if i is not None and e is not None:
            r_i = hapke_int(i, e, phase_angle, omega, theta, h_s, b0_s, h_c, b0_c, [g], 10)
            r_err_i = 0
        else:
            r_i, r_err_i = nquad(hapke_int, [[0, np.pi / 2], [0, np.pi]],
                                 args=(phase_angle, omega, theta, h_s, b0_s, h_c, b0_c,
                                       [g], 10))
        return r_i, r_err_i


def hapke_int(i, e, phase, w, slope, hs, bs, hc, bc, spsf_par, spsf_type):
    return hapke_ref(i, e, phase=phase, w=w, slope=slope, hs=hs, bs=bs, hc=hc, bc=bc,
                     spsf_par=spsf_par, spsf_type=spsf_type)


def R_up(w):
    df_67p = pd.read_csv("data/filacchione_67p_virtis_augsep2014.dat", delimiter="\s", skiprows=0)
    print(df_67p.head())
    comet = interp1d(df_67p["wavelength"], df_67p["I/F_body"], fill_value="extrapolate")
    return comet(w) / comet(649) * 0.25 * 2
    return (0.08 - 0.05) / (743 - 480) * w / ((0.08 - 0.05) / (743 - 480) * 649)


def R_low(w):
    df_67p = pd.read_csv("data/filacchione_67p_virtis_augsep2014.dat", delimiter="\s", skiprows=0)
    print(df_67p.head())
    comet = interp1d(df_67p["wavelength"], df_67p["I/F_body"], fill_value="extrapolate")
    return comet(w) / comet(649) * 0.25 / 2
    return R_up(w) / 10


if __name__ == "__main__":
    phase_angle = np.linspace(0, 100, 100)
    # phase_angle = np.linspace(0, 91, 1000)
    # i = phase_angle
    # e = np.zeros(phase_angle.shape)
    # r_single = hapke_ref(i, e, phase=phase_angle, w=omega, slope=theta, hs=h_s, bs=b0_s, hc=h_c, bc=b0_c,
    #                      spsf_par=[g], spsf_type=10)

    r, r_err = get_ref(phase_angle, i=phase_angle, e=np.zeros(phase_angle.shape))
    d = {
        "alpha": phase_angle,
        "r": r,
        "r_err": r_err
    }

    df = pd.DataFrame(data=d)
    df.to_csv("ref.csv", index=False)
    # plt.plot(phase_angle, r * R(480), color="blue", label=r"$\lambda=$480nm (integrated)")
    # plt.plot(phase_angle, r * R(649), color="lightgreen", label=r"$\lambda=$649nm (integrated)")
    # plt.plot(phase_angle, r * R(743), color="red", label=r"$\lambda=$743nm (integrated)")
    plt.plot(phase_angle,
             get_ref(phase_angle=phase_angle, i=phase_angle, e=np.zeros(phase_angle.shape))[0] * R_up(480),
             ls="--",
             color="#4767af", label=r"$\lambda=$480nm $i=\alpha$ $e=0$")  # blue
    plt.plot(phase_angle,
             get_ref(phase_angle=phase_angle, i=phase_angle, e=np.zeros(phase_angle.shape))[0] * R_up(649),
             ls="--",
             color="#466553", label=r"$\lambda=$649nm $i=\alpha$ $e=0$")  # green
    plt.plot(phase_angle,
             get_ref(phase_angle=phase_angle, i=phase_angle, e=np.zeros(phase_angle.shape))[0] * R_up(743),
             ls="--",
             color="#e6002e", label=r"$\lambda=$743nm $i=\alpha$ $e=0$")  # red
    # plt.fill_between(phase_angle, r - r_err, r + r_err, color="red", alpha=0.5)
    plt.xlabel(r"$\alpha$ [°]")
    plt.ylabel("reflectance")
    plt.legend()
    plt.savefig("reflectance_coca.png")
    plt.show()

    l = np.linspace(300, 1100, 100)
    img_r, img_l = np.meshgrid(r, l)
    img_low = img_r * R_low(l)[:, None]
    img_up = img_r * R_up(l)[:, None]

    np.save("mat_low.npy", img_low)
    np.save("mat_up.npy", img_up)

    plt.yticks(np.arange(0, 100, 12.5), np.arange(300, 1100, 100))
    plt.xticks(np.arange(0, 100, 10), np.arange(0, 100, 10))
    plt.ylabel(r"$\lambda$ [nm]")
    plt.xlabel(r"$\alpha$ [deg]")
    plt.imshow(img_low)
    plt.colorbar()
    plt.title("reflectance")
    plt.savefig("lambda_vs_phase_angle.png")
    plt.show()
