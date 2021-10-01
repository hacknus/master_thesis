import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.integrate import nquad
from hapke import hapke, hapke_ice, disk_int_hapke, hapke_scaled
import pandas as pd
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import curve_fit
from unibe import *
import os


def ref_rock(wavelength, phase_angle, kind="linear"):
    phase_angle = np.radians(phase_angle)
    return hapke_scaled(phase_angle, phase_angle, np.zeros(phase_angle.shape), wavelength)


def ref_ice(wavelength, phase_angle):
    phase_angle = np.radians(phase_angle)
    return 3 * hapke_scaled(phase_angle, phase_angle, np.zeros(phase_angle.shape), wavelength)


if __name__ == "__main__":
    pass
