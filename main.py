# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
import numpy as np


class Motor:

    def __init__(self, dt=0.001):
        self.x = 0
        self.v = 0
        self.a = 0
        self.dt = dt

    def move(self):
        # leapfrog
        v_temp = self.v + self.a * self.dt/2
        self.x += v_temp * self.dt
        self.v = v_temp + self.a * self.dt/2


def main_para(time_span=500, max_vel=5000, max_accel=6000, accel_inc=6000):

    ACCEL_TIME = 0.16
    SMOOTH_FAC = 0.15
    BACKLASH = 0.05
    CRUISE = 0.13

    x = [0]
    v = [0]
    a = [0]

    t = np.linspace(0, 0.5, 1000)
    t_new = [0]
    dt = t[1] - t[0]
    print("dt: {:.2f} ms".format(dt * 1000))
    for ti in t:
        if ti < BACKLASH:
            at = 0
            vt = 400 / 60 * 2 * np.pi
            xt = x[-1] + vt * dt
            a.append(at)
            v.append(vt)
            x.append(xt)
            t_new.append(ti)
        elif ti > BACKLASH and ti < ACCEL_TIME*SMOOTH_FAC + BACKLASH:
            at = a[-1] + accel_inc/(ACCEL_TIME*SMOOTH_FAC)*dt
            vt = v[-1] + at * dt
            xt = x[-1] + vt * dt + at / 2 * dt ** 2
            a.append(at)
            v.append(vt)
            x.append(xt)
            t_new.append(ti)
        elif ti > ACCEL_TIME * SMOOTH_FAC + BACKLASH and ti < BACKLASH + ACCEL_TIME * (1 - SMOOTH_FAC):
            at = a[-1]
            vt = v[-1] + at * dt
            xt = x[-1] + vt * dt + at / 2 * dt ** 2
            a.append(at)
            v.append(vt)
            x.append(xt)
            t_new.append(ti)
        elif ti > ACCEL_TIME * (1 - SMOOTH_FAC) + BACKLASH and ti < ACCEL_TIME + BACKLASH:
            at = a[-1] - accel_inc/(ACCEL_TIME*SMOOTH_FAC)*dt
            vt = v[-1] + at * dt
            xt = x[-1] + vt * dt + at / 2 * dt ** 2
            a.append(at)
            v.append(vt)
            x.append(xt)
            t_new.append(ti)
        elif ti > ACCEL_TIME + BACKLASH and ti < ACCEL_TIME + BACKLASH + CRUISE:
            at = a[-1]
            vt = v[-1] + at * dt
            xt = x[-1] + vt * dt + at / 2 * dt ** 2
            a.append(at)
            v.append(vt)
            x.append(xt)
            t_new.append(ti)
        elif ti > ACCEL_TIME + BACKLASH + CRUISE and ti < ACCEL_TIME + BACKLASH + CRUISE + ACCEL_TIME * SMOOTH_FAC:
            at = a[-1] - accel_inc/(ACCEL_TIME*SMOOTH_FAC)*dt
            vt = v[-1] + at * dt
            xt = x[-1] + vt * dt + at / 2 * dt ** 2
            a.append(at)
            v.append(vt)
            x.append(xt)
            t_new.append(ti)
        elif ti > ACCEL_TIME + BACKLASH + CRUISE + ACCEL_TIME * SMOOTH_FAC and ti < ACCEL_TIME + BACKLASH + CRUISE + ACCEL_TIME * (1 - SMOOTH_FAC):
            at = a[-1]
            vt = v[-1] + at * dt
            xt = x[-1] + vt * dt + at / 2 * dt ** 2
            a.append(at)
            v.append(vt)
            x.append(xt)
            t_new.append(ti)
        elif ti > ACCEL_TIME + BACKLASH + CRUISE + ACCEL_TIME * (1 - SMOOTH_FAC) and ti < ACCEL_TIME + BACKLASH + CRUISE + ACCEL_TIME:
            at = a[-1] + accel_inc/(ACCEL_TIME*SMOOTH_FAC)*dt
            vt = v[-1] + at * dt
            xt = x[-1] + vt * dt + at / 2 * dt ** 2
            a.append(at)
            v.append(vt)
            x.append(xt)
            t_new.append(ti)

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, sharex=True)
    ax0.plot(t_new, np.array(x) * 60 / (2*np.pi) / 80)
    ax1.plot(t_new, np.array(v) * 60 / (2*np.pi))
    ax2.plot(t_new, np.array(a))
    ax0.set_ylabel("pos [deg]")
    ax1.set_ylabel("vel [rpm]")
    ax2.set_ylabel("acc [rad/s2]")
    ax2.set_xlabel("t [s]")
    plt.show()

def main(time_span=500, max_vel=5000, max_accel=6000, accel_inc=200):

    bldc = Motor()

    t = np.linspace(0, 500, 1000)

    x = np.zeros(len(t))
    v = np.zeros(len(t))
    a = np.zeros(len(t))

    x_end = 35  # degrees

    p = 1

    for i in range(len(t)-1):
        if p > 0 and a[i] > max_accel:
            p = 0
        if p == 0 and v[i] > 0.8*max_vel:
            p = -1
        if p == -1 and a[i] < 0:
            p = 0
        a[i+1] = a[i] + p*accel_inc
        bldc.a = a[i+1]
        bldc.move()
        v[i+1] = bldc.v
        x[i+1] = bldc.x

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, sharex=True)
    ax0.plot(t,x)
    ax1.plot(t,v)
    ax2.plot(t,a)
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_para()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
