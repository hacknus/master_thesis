import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from unibe import *
from make_filters import make_filter
from scipy.interpolate import interp1d
from comet import ref_rock
from scipy.integrate import quad
from camera import Camera
import scipy.constants as const
from SNR import snr


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


class DraggableScatter():
    epsilon = 10

    def __init__(self, scatter, filters, fig, v, mode):

        df = pd.read_csv(f"data/widths.csv")
        self.width = interp1d(df.c, df.widths * 2, kind="linear", fill_value="extrapolate")
        self.scatter = scatter
        self.filters = filters
        self._ind = None
        self.ax = scatter.axes
        self.canvas = self.ax.figure.canvas
        self.filter_data = [
            {"center": self.scatter.get_offsets()[0][0],
             "width": self.scatter.get_offsets()[0][1]},
            {"center": self.scatter.get_offsets()[1][0],
             "width": self.scatter.get_offsets()[1][1]},
            {"center": self.scatter.get_offsets()[2][0],
             "width": self.scatter.get_offsets()[2][1]},
            {"center": self.scatter.get_offsets()[3][0],
             "width": self.scatter.get_offsets()[3][1]},
        ]
        self.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        plt.tight_layout()
        plt.show()
        fig.savefig(f"plots/selected_filters_v{v}_mode_{mode}.pdf")
        for filter in self.filter_data:
            print(filter)

    def get_ind_under_point(self, event):
        xy = np.asarray(self.scatter.get_offsets())
        xyt = self.ax.transData.transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]

        d = np.sqrt((xt - event.x) ** 2 + (yt - event.y) ** 2)
        ind = d.argmin()

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        if event.button != 1:
            return
        self._ind = None

    def motion_notify_callback(self, event):
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata
        if x > 1000: x = 1000
        if x < 400: x = 400
        wavelengths = np.linspace(300, 1100, 1000)
        filter = make_filter(x, y)
        self.filter_data[self._ind]["center"] = x
        self.filter_data[self._ind]["width"] = y  # float(self.width(x))
        # y = self.width(x)
        self.filters[self._ind].set_ydata(100 * filter(wavelengths))
        xy = np.asarray(self.scatter.get_offsets())
        xy[self._ind] = np.array([x, y])
        self.scatter.set_offsets(xy)
        self.canvas.draw_idle()


def main(mode, v=30, alpha=11):
    fig, axes = plt.subplots(nrows=2, sharex=True)

    if mode != "C":
        wavelengths = np.linspace(300, 1100, 1000)
    else:
        wavelengths = np.linspace(200, 1100, 1000)

    df = pd.read_csv(f"data/widths_snr.csv")
    widths100_avg = df.widths_100
    widths80_avg = df.widths_80
    widths60_avg = df.widths_60
    centers = df.c
    width100 = interp1d(centers, widths100_avg, kind="quadratic", fill_value="extrapolate")
    width80 = interp1d(centers, widths80_avg, kind="quadratic", fill_value="extrapolate")
    width60 = interp1d(centers, widths60_avg, kind="quadratic", fill_value="extrapolate")
    axes[0].plot(np.linspace(380, 1000, 100), width100(np.linspace(380, 1000, 100)), color=BLACK, zorder=-1,
                 label="SNR 100")
    axes[0].plot(np.linspace(380, 1000, 100), width80(np.linspace(380, 1000, 100)), ls="-.", color=BLACK, zorder=-1,
                 label="SNR 80")
    axes[0].plot(np.linspace(380, 1000, 100), width60(np.linspace(380, 1000, 100)), ls="--", color=BLACK, zorder=-1,
                 label="SNR 60")
    axes[0].legend()
    if mode != "C":
        axes[0].set_ylim(0.8 * min(df.widths_60), max(df.widths_80))
    else:
        axes[0].set_xticks(np.arange(200,1101,100))
        axes[0].set_xticklabels(np.arange(200, 1101, 100))
    axes[0].set_ylabel("width [nm]")
    axes[0].set_xlabel("center [nm]")
    axes[1].set_xlabel("wavelength [nm]")
    axes[1].set_ylabel("transmission [%]")
    c1 = 460
    c2 = 650
    c3 = 764
    c4 = 900

    df = pd.read_csv(f"data/filters_{mode}.csv")
    c1, c2, c3, c4 = df.centers
    w1, w2, w3, w4 = df.widths

    F0 = make_filter(c1, w1)
    F1 = make_filter(c2, w2)
    F2 = make_filter(c3, w3)
    F3 = make_filter(c4, w4)
    if mode == "C":
        f0, = axes[1].plot(wavelengths, 100 * F0(wavelengths), color=BLUE)
        f1, = axes[1].plot(wavelengths, 100 * F1(wavelengths), color=GREEN)
        f2, = axes[1].plot(wavelengths, 100 * F2(wavelengths), color=ORANGE)
        f3, = axes[1].plot(wavelengths, 100 * F3(wavelengths), color=RED)
    else:
        f0, = axes[1].plot(wavelengths, 100 * F0(wavelengths), color=BLUE)
        f1, = axes[1].plot(wavelengths, 100 * F1(wavelengths), color=ORANGE)
        f2, = axes[1].plot(wavelengths, 100 * F2(wavelengths), color=RED)
        f3, = axes[1].plot(wavelengths, 100 * F3(wavelengths), color=BLACK)
    axes[1].plot(wavelengths, np.zeros(wavelengths.shape), color=BLACK)
    M = get_mirror()
    Q = get_detector()
    S = get_solar()
    signal = M(wavelengths) * Q(wavelengths) * ref_rock(wavelengths,alpha) * S(wavelengths)
    signal = signal / np.max(signal) * 100
    # axes[1].plot(wavelengths, signal.T, color=BLACK)

    df = pd.read_csv("data/texp.csv")
    t = interp1d(df.alpha, df["texp10"], fill_value="extrapolate")
    t_exp = t(alpha) / (v / 10) / 1000

    # axes[0].set_title(f"phase angle = {alpha}°")
    if mode == "C":
        scatter = axes[0].scatter([c1, c2, c3, c4], [w1, w2, w3, w4],
                                  color=[BLUE, GREEN, ORANGE, RED], edgecolor=BLACK)
    else:
        scatter = axes[0].scatter([c1, c2, c3, c4], [w1, w2, w3, w4],
                                  color=[BLUE, ORANGE, RED, BLACK], edgecolor=BLACK)
    DraggableScatter(scatter, [f0, f1, f2, f3], fig, v, mode)


if __name__ == "__main__":
    main("B", alpha=11)
