import numpy as np


def make_filter(center=480, width=200, maximum=0.98):
    def F(w):
        if isinstance(w, np.ndarray):
            f = np.zeros(w.shape)
            f[(center - width / 2 < w) & (center + width / 2 > w)] = maximum
        else:
            if center - width / 2 < w < center + width / 2:
                f = maximum
            else:
                f = 0
        return f

    return F


def init_filters(center, widths, maxima=[0.98, 0.98, 0.98, 0.98]):
    w1 = widths[0]
    w2 = widths[1]
    w3 = widths[2]
    w4 = widths[3]
    c1 = center
    c2 = center + w1 / 2 + w2 / 2
    c3 = center + w1 / 2 + w2 + w3 / 2
    c4 = center + w1 / 2 + w2 + w3 + w4 / 2

    F1 = make_filter(c1, w1, maxima[0])
    F2 = make_filter(c2, w2, maxima[1])
    F3 = make_filter(c3, w3, maxima[2])
    F4 = make_filter(c4, w4, maxima[3])

    return F1, F2, F3, F4


def init_filters_thomas(maxima=[0.98, 0.98, 0.98, 0.98]):
    w1 = 150
    w2 = 100
    w3 = 100
    w4 = 150
    c1 = 460
    c2 = 550
    c3 = 750
    c4 = 900

    F1 = make_filter(c1, w1, maxima[0])
    F2 = make_filter(c2, w2, maxima[1])
    F3 = make_filter(c3, w3, maxima[2])
    F4 = make_filter(c4, w4, maxima[3])

    return F1, F2, F3, F4
