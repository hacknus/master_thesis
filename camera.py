import numpy as np


class Camera:

    def __init__(self, r_h=1, t_exp=0.001):
        self.G = 3.5  # electron per DN
        self.d_pixel = 7e-6  # meters
        self.A_pixel = self.d_pixel ** 2
        self.d_aperture = 0.135  # meters
        self.A_aperture = (self.d_aperture / 2) ** 2 * np.pi
        self.focal_length = 0.880  # meters
        self.Omega = 0.64e-10
        self.A_Omega = self.A_aperture * self.A_pixel / self.focal_length ** 2
        self.r_h = r_h  # A.U.
        self.t_exp = t_exp  # seconds
