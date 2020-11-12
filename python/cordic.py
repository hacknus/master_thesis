import numpy as np
import sys
import time

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
        self.k = int(k[-1])
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
        Vy = (Vy >> 13) * self.k/int(self.scale/2**13)
        return Vy*sign

if __name__ == "__main__":

    C = Cordic(20, 10000000)
    phis = np.linspace(0, 2*np.pi, 10) % (2*np.pi) * C.scale
    phis = np.array(phis, dtype=np.int)
    for phi in phis:
        s = np.sin(phi/ C.scale)
        c = C.sin(phi)
        print(f"sin({phi/np.pi*180/C.scale:.2f}°): true = {s:.10f}, cordic = {c / C.scale:.10f}")
        #string = int(c*20+20)*" " + int(20-c*20)*"#"
        #sys.stdout.write(f"\r \r {string}")
        #sys.stdout.flush()
        #time.sleep(0.1)