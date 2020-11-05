import numpy as np



def cordic(phi,N):

    if phi > np.pi and phi <= 2*np.pi:
        phi -= np.pi

    n = np.arange(N)
    atans = np.arctan(2.0 ** (-1 * n))
    k = []
    value = 1.0
    for i in n:
        value = value * np.sqrt(1.0 + 2.0 ** (-2 * i))
        k.append(1.0 / value)

    Vx, Vy = 1.0, 0.0
    for i in n:
        Vxold = Vx
        Vyold = Vy
        if phi < 0:
            Vx = Vxold + Vyold * 2.0 ** (-1 * i)
            Vy = Vyold - Vxold * 2.0 ** (-1 * i)
            phi = phi + atans[i]
        else:
            Vx = Vxold - Vyold * 2.0 ** (-1 * i)
            Vy = Vyold + Vxold * 2.0 ** (-1 * i)
            phi = phi - atans[i]
    Vx, Vy = Vx * k[i], Vy * k[i]
    return Vy


if __name__ == "__main__":
    phis = np.linspace(0,np.pi/2,10)
    for phi in phis:
        sin = cordic(phi,10)
        print(f" phi = {phi/np.pi*180:.2f}° : true sine = {np.sin(phi):.10f}, cordic phi = {sin:.10f}")
