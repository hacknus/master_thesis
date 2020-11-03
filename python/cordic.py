import numpy as np
import matplotlib.pyplot as plt



import math

# I know CORDIC is only valid for inputs between
# -pi/2 and pi/2; I am not exactly sure what I need
# to do add to make it where any input is acceptable
# I believe is keep on adding/subtracting pi, but I don't
# get why this works?
def cordic_trig(beta,N=40):
    # in hardware, put this in a table.
    def K_vals(n):
        K = []
        acc = 1.0
        for i in range(0, n):
            acc = acc * (1.0/np.sqrt(1 + 2.0**(-2*i)))
            K.append(acc)
        return K

    #K = K_vals(N)
    K = 0.6072529350088812561694
    # emulation for hardware lookup table
    atans = [np.arctan(2.0**(-i)) for i in range(0,N)]
    #print K
    #print atans
    x = 1
    y = 0

    for i in range(0,N):
        d = 1.0
        if beta < 0:
            d = -1.0

        x = (x - (d*(2.0**(-i))*y))
        y = ((d*(2.0**(-i))*x) + y)
        # in hardware put the atan values in a table
        beta = beta - (d*np.arctan(2**(-i)))
    return (K*x, K*y)

if __name__ == '__main__':
    beta = math.pi/6.0
    print("Actual cos(%f) = %f" % (beta, np.cos(beta)))
    print("Actual sin(%f) = %f" % (beta, np.sin(beta)))
    cos_val, sin_val = cordic_trig(beta)
    print("CORDIC cos(%f) = %f" % (beta, cos_val))
    print("CORDIC sin(%f) = %f" % (beta, sin_val))