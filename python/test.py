# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
import numpy as np

# TODO: analytical compared to numeric motor simulation with discrete steps
# TODO: criteria for FPGA is position (motor position)


def values(x0, v0, a0, m, ta, tb):
    xt = x0 + v0*(tb - ta) + a0/2*(tb - ta)**2 + m/6*(tb - ta)**3
    vt = v0 + a0*(tb - ta) + m/2*(tb - ta)**2
    at = a0 + m*(tb - ta)
    return at, vt, xt

time_scale = 1000*1000
pos_scale = time_scale**3/100000
t = np.linspace(0, 0.5*time_scale, int(0.5*time_scale))
amax = 1500/time_scale**2*pos_scale
vmax = 1500/9.549/time_scale*pos_scale
xmax = 35*80/180*np.pi*pos_scale
m = 100000/time_scale**3*pos_scale
print(f"m: {m}")

print(amax / m, vmax / amax)

t1 = amax / m
t2 = vmax / amax - t1 + t1
t3 = t2 + t1
print(xmax)
print("position t3: ", 3*m/2*(t2-t1)*t1**2 + m/2*t1*(t2-t1)**2 + m*t1**3)
xtogo = xmax - 2*(3*m/2*(t2-t1)*t1**2 + m/2*t1*(t2-t1)**2 + m*t1**3)
t_const = xtogo / (m*t1*(t2-t1) + m*t1**2)
print("velocity: ", (m*t1*(t2-t1) + m*t1**2))
t4 = t3 + t_const
t5 = t4 + t1
t6 = t4 + t2
t7 = t4 + t3
print(t1, t2, t3, t4, t5, t6, t7)

pos = []
vel = []
acc = []

dt = t[1] - t[0]
print(f"dt: {dt}")
n_a = [0]
n_v = [0]
n_x = [0]

for i, ti in enumerate(t):
    if ti <= t1:
        a, v, x = values(0, 0, 0, m, 0, ti)
        atemp = m*dt
        t1i = i
    elif ti >= t1 and ti < t2:
        a, v, x = values(pos[t1i], vel[t1i], m * t1, 0, t1, ti)
        atemp = 0
        t2i = i
    elif ti >= t2 and ti < t3:
        a, v, x = values(pos[t2i], vel[t2i], m * t1, -m, t2, ti)
        atemp = -m*dt
        t3i = i
    elif ti >= t3 and ti < t4:
        a, v, x = values(pos[t3i], vel[t3i], 0, 0, t3, ti)
        atemp = 0
        t4i = i
    elif ti >= t4 and ti < t5:
        a, v, x = values(pos[t4i], vel[t4i], 0, -m, t4, ti)
        atemp = -m*dt
        t5i = i
    elif ti >= t5 and ti < t6:
        a, v, x = values(pos[t5i], vel[t5i], -m*(t5-t4), 0, t5, ti)
        atemp = 0
        t6i = i
    elif ti >= t6 and ti < t7:
        a, v, x = values(pos[t6i], vel[t6i], -m*(t5-t4), m, t6, ti)
        atemp = m*dt
    else:
        atemp = 0
    pos.append(x)
    vel.append(v)
    acc.append(a)
    n_a.append(n_a[-1] + atemp)
    n_v.append(n_v[-1] + n_a[-1] * dt)
    n_x.append(n_x[-1] + n_v[-1] * dt + n_a[-1]/2*dt**2)

print("Sanity test:")
a, v, x = values(0, 0, 0, m, 0, t1)
print("a(t1) = {:.10f}".format(a))
print("x(t1) = {:.10f}".format(x))
a, v, x = values(x, v, a, 0, t1, t2)
a, v, x = values(x, v, a, -m, t2, t3)
print("v(t3) = {:.10f}".format(v))
print("x(t3) = {:.10f}".format(x))
a, v, x = values(x, v, a, 0, t3, t4)
a, v, x = values(x, v, a, -m, t4, t5)
a, v, x = values(x, v, a, 0, t5, t6)
a, v, x = values(x, v, a, m, t6, t7)
print("v(t7) = {:.10f}".format(v*9.549))
print("x(t7) = {:.10f}".format(x/80/np.pi*180))

print("true values:")
print(f"v(t3) = {max(vel)*9.549:.10f}, {max(n_v)*9.549:.10f}")
print(f"v(t7) = {vel[-1]*9.549:.10f}, {n_v[-1]*9.549:.10f}")
print(f"x(t7) = {max(pos)/80/np.pi*180:.10f}, {max(n_x)/80/np.pi*180:.10f}")

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, sharex=True)
ax0.plot(t, np.array(pos)/80/np.pi*180)
ax0.plot(t, np.array(n_x[1:])/80/np.pi*180,ls="--")
#ax0.plot(t, np.array(pos))
ax0.set_ylabel("pos [deg]")
ax1.plot(t, np.array(vel)*9.549)
ax1.plot(t, np.array(n_v[1:])*9.549,ls="--")
ax1.set_ylabel(r"vel [rpm]")
ax2.plot(t, acc)
ax2.plot(t, n_a[1:],ls="--")
ax2.set_ylabel(r"acc [rad/s$^2$]")
ax2.set_xlabel("t [s]")
plt.savefig("curves.pdf", transparent=True)
plt.show()