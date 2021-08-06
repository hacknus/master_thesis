class Camera:

    def __init__(self, r_h=1, t_exp=0.0001):
        self.G = 2.5  # electron per DN
        self.A = 7e-6 ** 2
        self.Omega = 0.64e-10
        self.r_h = r_h  # A.U.
        self.t_exp = t_exp  # seconds
