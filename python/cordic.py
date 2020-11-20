import numpy as np
import sys
import time
import matplotlib.pyplot as plt

def check_32bit(a, signed=True):
    if signed:
        if 2**31 - np.abs(a) < 0:
            print(f" {a} is exceeds signed 32bit int bounds +/- {2**31}.")
            exit()
    else:
        if 2**32 - np.abs(a) < 0:
            print(f" {a} is exceeds unsigned 32bit int bounds +/- {2**32}.")
            exit()

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

        self.p = 14
        self.k = int(k[-1] / (self.scale >> self.p))

        check_32bit(self.k, False)
        check_32bit(np.pi * scale, False)
        self.atans = np.array(atans, dtype=np.int)
        print("k:")
        print(self.k)
        print("atans:")
        print(list(self.atans))

    def sin(self, phi):

        sign = 1
        if phi > int(np.pi * self.scale / 2) and phi < int(np.pi * self.scale * 3 / 2):
            phi = phi - int(np.pi * self.scale)
            sign = -1
        # elif phi > int(np.pi * self.scale) and phi < int(np.pi * self.scale * 3 / 2):
        #     phi = phi - int(np.pi * self.scale)
        #     sign = -1
        elif phi > int(np.pi * self.scale * 3 / 2):
            phi = phi - int(np.pi * self.scale * 2)
            sign = 1

        """
        if phi < -int(np.pi * self.scale / 2):
            phi = phi + int(np.pi * self.scale)
            sign = -1
        """

        Vx = self.scale
        Vy = 0
        for i in self.n:
            Vxold = Vx
            Vyold = Vy
            if phi < 0:
                t = Vyold
                t = t >> i
                Vx = Vxold + t
                t = Vxold
                t = t >> i
                Vy = Vyold - t
                phi = phi + self.atans[i]
            else:
                t = Vyold
                t = t >> i
                Vx = Vxold - t
                t = Vxold
                t = t >> i
                Vy = Vyold + t
                phi = phi - self.atans[i]
        Vy = (Vy >> self.p) * self.k
        check_32bit(Vy)
        return Vy*sign

if __name__ == "__main__":

    N = 256
    C = Cordic(24, 10000000)
    sin_table = np.sin(np.linspace(0, 2*np.pi, N))
    phis = np.linspace(0, 2*np.pi, 100) % (2*np.pi) * C.scale
    phis = np.array(phis, dtype=np.int)
    ds = []
    ds_table = []
    for phi in phis:
        s = np.sin(phi / C.scale)
        c = C.sin(phi)
        d = s - c / C.scale
        ds.append(d)
        ds_table.append(s - sin_table[int(phi/(2 * C.scale * np.pi) * N)])

        print(f"sin({phi/np.pi*180/C.scale:.2f}°): diff = {d:.10f}")
        #print(f"sin({phi/np.pi*180/C.scale:.2f}°): true = {s:.10f}, cordic = {c / C.scale:.10f}")
        #string = int(c*20+20)*" " + int(20-c*20)*"#"
        #sys.stdout.write(f"\r \r {string}")
        #sys.stdout.flush()
        #time.sleep(0.1)
    plt.scatter(phis / C.scale, ds, label="cordic")
    plt.scatter(phis / C.scale, ds_table, label = "table")
    plt.ylabel("Amplitudenfehler")
    plt.xlabel(r"$\phi$ [rad]")
    plt.legend()
    plt.show()