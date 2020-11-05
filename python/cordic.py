import numpy as np


class Cordic:

    def __init__(self, N, scale):
        self.N = N
        self.scale = scale

        self.n = np.arange(N)
        atans = np.arctan(2.0 ** (-1 * self.n))
        k = []
        value = 1.0
        for i in self.n:
            value = value * np.sqrt(1.0 + 2.0 ** (-2 * i))
            k.append(1.0 / value)

        k = np.array(k) * scale
        atans = np.array(atans) * scale
        self.k = np.array(k, dtype=np.int)
        self.atans = np.array(atans, dtype=np.int)
        print("k:")
        print(self.k)
        print("atans:")
        print(self.atans)

    def sin(self, phi):

        if phi > np.pi and phi <= 2*np.pi:
            phi -= np.pi
        phi = phi * self.scale
        phi = int(phi)

        Vx = 1.0 * self.scale
        Vy = 0.0
        for i in self.n:
            Vxold = Vx
            Vyold = Vy
            if phi < 0:
                t = Vyold
                for j in range(i):
                    t = t / 2
                Vx = Vxold + t
                t = Vxold
                for j in range(i):
                    t = t / 2
                Vy = Vyold - t
                phi = phi + self.atans[i]
            else:
                t = Vyold
                for j in range(i):
                    t = t / 2
                Vx = Vxold - t
                t = Vxold
                for j in range(i):
                    t = t / 2
                Vy = Vyold + t
                phi = phi - self.atans[i]
        Vx = Vx * self.k[i] / self.scale ** 2
        Vy = Vy * self.k[i] / self.scale ** 2
        return Vy

if __name__ == "__main__":

    C = Cordic(30, 1e9)
    phis = np.linspace(-np.pi/2, np.pi/2, 10)

    for phi in phis:
        s = np.sin(phi)
        c = C.sin(phi)
        print(f"sin({phi/np.pi*180:.2f}°): true = {s:.10f}, cordic = {c:.10f}")
