#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A straightforward Python implementation of the classical Hapke model with different 
options based on the equations in the books and papers.

This is a direct Python translation of my IDL package 'hapke_ref_2013ap' with the 
exception of the Anisotropic Multiple Scattering (AMS) which is not yet implemented

Created on Sat Feb 18 21:08:15 2017

27/03/2019: Update the routines to accept all inputs as floats, lists or numpy arrays

@author: Antoine Pommerol, UniBE
"""

import numpy as np


# _____ Calculation of the phase angle from incidence, emission and azimuth angles

def phase_angle(i, e, a, radians=0):
    if radians == 0:
        cos_g = np.cos(np.radians(i)) * np.cos(np.radians(e)) + np.sin(np.radians(i)) * np.sin(np.radians(e)) * np.cos(
            np.radians(a))
    if radians != 0:
        cos_g = np.cos(i) * np.cos(e) + np.sin(i) * np.sin(e) * np.cos(a)
    g = np.degrees(np.arccos(cos_g))
    return (g)


# _____ 1-term Henyey-Greenstein function

def hg1(g, a, radians=0):
    if radians == 0:
        hg = (1.0 - a ** 2) / ((1.0 + 2.0 * a * np.cos(g) + a ** 2) ** 1.5)
    if radians != 0:
        hg = (1.0 - a ** 2) / ((1.0 + 2.0 * a * np.cos(np.radians(g)) + a ** 2) ** 1.5)
    return (hg)


# _____ Various definitions of the single particle scattering functions

def single_part_scat_func(g, par=[0], ch=0, radians=0):
    if radians == 0:
        cos_g = np.cos(np.radians(g))
    else:
        cos_g = np.cos(g)
    a = par

    # _____ 0-Parameter phase function _________________________________________

    if ch == 0:
        f = np.repeat(1.0, np.size(g))  # Istropic phase function
    if ch == 2:
        f = 3. / 4. * (1. + cos_g ** 2.)

    # _____ 1-Parameter phase function _________________________________________

    # Single parameter henyey-Greenstein
    if ch == 10:
        a = a[0]
        aa = a ** 2
        f = (1.0 - aa) / ((1.0 + 2.0 * a * cos_g + aa) ** 1.5)

    # Single parameter Henyey-Greenstein formulation from the original formula (1941)
    if ch == 11:
        a = a[0]
        aa = a ** 2
        f = (1. / (4. * np.pi)) * (1.0 - aa) / ((1.0 - 2.0 * a * cos_g + aa) ** 1.5)

    # Double Henyey-Greenstein formulation with Hockey-stick relationship (Hapke, 2012)
    if ch == 12:
        b = a[0]
        c = 3.29 * np.exp(-17.4 * b ** 2) - 0.908
        f = (1. + c) / 2. * (1. - b ** 2) / (1. - 2. * b * cos_g + b ** 2) ** 1.5 + (1. - c) / 2. * (1. - b ** 2) / (
                    1. + 2. * b * cos_g + b ** 2) ** 1.5

    # First-order Legendre polynomial (Hapke, 2002)
    if ch == 13:
        f = 1. + a[0] * cos_g

    # _____ 2-Parameters phase function ________________________________________

    # Double Henyey-Greenstein formulation from MacGuire and Hapke (1995) and Hapke (2012). c is positive for bwd scatterers and negative for fwd scatterers
    if ch == 20:
        b = a[0]
        c = a[1]
        f = (1. + c) / 2. * (1. - b ** 2) / (1. - 2. * b * cos_g + b ** 2) ** 1.5 + (1. - c) / 2. * (1. - b ** 2) / (
                    1. + 2. * b * cos_g + b ** 2) ** 1.5

    # Double Henyey-Greenstein formulation from Grundy (1999)
    if ch == 21:
        fwd = hg1(g, np.absolute(a[0]), radians=radians)
        bkwd = hg1(g, -np.absolute(a[0]), radians=radians)
        f = a[1] * fwd + (1.0 - a[1]) * bkwd

    # Double Henyey-Greenstein formulation from Domingue al. (1997)
    if ch == 22:
        b = a[0]
        c = a[1]
        f = ((1. - c) * (1. - b ** 2.)) / ((1. + 2. * b * cos_g + b ** 2.) ** 1.5) + (c * (1. - b ** 2.)) / (
                    (1. - 2. * b * cos_g + b ** 2.) ** 1.5)

    # Double Henyey-Greenstein formulation from Johnson et al. (2006, 2013). The "c" in this equation is actually the fwd fraction, noted c' (prime) in the papers
    if ch == 23:
        b = a[0]
        c = a[1]
        f = (c * (1. - b ** 2)) / ((1. + 2. * b * cos_g + b ** 2) ** 1.5) + ((1. - c) * (1. - b ** 2)) / (
                    (1. - 2. * b * cos_g + b ** 2) ** 1.5)

    # _____ 3-Parameters phase function ________________________________________

    # Triple Henyey-Greenstein formulation similar to the double formulation by Hapke
    if ch == 30:
        b0 = a[0]
        b1 = a[1]
        c = a[2]
        f = (1. + c) / 2. * (1. - b0 ** 2) / (1. + 2. * b0 * cos_g + b0 ** 2) ** 1.5 + (1. - c) / 2. * (
                    1. - b1 ** 2) / (1. - 2. * b1 * cos_g + b1 ** 2) ** 1.5

        # Triple Henyey-Greenstein formulation from Grundy (1999)
    if ch == 31:
        fwd = hg1(g, a[0], radians=radians)
        bkwd = hg1(g, a[1], radians=radians)
        f = a[2] * fwd + (1.0 - a[2]) * bkwd

    # Triple Henyey-Greenstein formulation from Domingue et al. (1997)
    if ch == 32:
        b = a[0]
        c = a[1]
        d = a[2]
        f = ((1. - c) * (1. - b ** 2.)) / ((1. + 2. * b * cos_g + b ** 2.) ** 1.5) + (c * (1. - d ** 2.)) / (
                    (1. + 2. * d * cos_g + d ** 2.) ** 1.5)

    return (f)


# _____ The H-function for the multiple scattering, with different approximations

def h_function(x, w, h1986=0, h1993=0):
    x = np.array(x)
    w = np.array(w)

    f = np.repeat([1.], np.size(x))

    z = np.nonzero(x != 0)

    if (np.size(z) != 0):

        gamma = np.sqrt(1 - w)
        r0 = (1 - gamma) / (1 + gamma)

        if (h1986 != 0):
            f[z] = (1. + (2. * x[z])) / (1. + (2. * gamma * x[z]))

        if (h1993 != 0):
            lnf = np.log((1. + x[z]) / x[z])
            f0 = 1. - (1. - gamma) * x[z] * (r0 + (1. - 0.5 * r0 - r0 * x[z]) * lnf)
            f[z] = 1. / f0

        if (h1986 == 0 and h1993 == 0):
            f0 = 1. - (w * x[z]) * (r0 + ((1. - (2. * r0 * x[z])) / 2.) * np.log((1. + x[z]) / x[z]))
            f[z] = 1. / f0

    return (f)


# _____ The macroscopic roughness correction, following the equations in the 1993 book

def hapke_roughness(emu, imu, g, theta):
    z = np.flatnonzero(theta != 0)

    if (np.size(z)) != 0:
        emue = emu[z]
        imue = imu[z]
        emue0 = emu[z]
        imue0 = imu[z]
        sfun = emu[z]

    # ancillary values

    tanthe = np.tan(theta[z])
    costhe = np.cos(theta[z])
    cotthe = 1.0 / tanthe
    cotthe2 = cotthe ** 2.

    i = np.arccos(imu[z])
    sini = np.sin(i)
    e = np.arccos(emu[z])
    sine = np.sin(e)

    cosg = np.cos(g[z])
    cosphi = np.repeat(1.0, np.size(z))

    i_e = np.array(i * e)
    zz = np.flatnonzero(i_e != 0.)

    if np.size(zz) != 0: cosphi[zz] = (cosg[zz] - imu[z[zz]] * emu[z[zz]]) / (sini[zz] * sine[zz])

    zz = np.flatnonzero(cosphi >= 1.0)
    if np.size(zz) != 0: cosphi[zz] = 1.0

    zz = np.flatnonzero(cosphi <= -1.0)
    if np.size(zz) != 0: cosphi[zz] = -1.0

    phi = np.arccos(cosphi)
    sinphi2_2 = np.sin(phi / 2.0) ** 2

    gold = 1.0e-7

    z0 = np.flatnonzero(np.abs(sini) < gold)
    if np.size(z0) != 0: cosphi[z0] = gold

    z0 = np.flatnonzero(np.abs(sine) < gold)
    if np.size(z0) != 0: sine[z0] = gold

    coti = imu[z] / sini
    coti2 = coti ** 2.
    cote = emu[z] / sine
    cote2 = cote ** 2.

    e1i = np.exp(-2.0 / np.pi * cotthe * coti)  # eqn. 12.45b, p. 344
    e2i = np.exp(-1.0 / np.pi * cotthe2 * coti2)  # eqn. 12.45c, p. 344
    e1e = np.exp(-2.0 / np.pi * cotthe * cote)
    e2e = np.exp(-1.0 / np.pi * cotthe2 * cote2)

    chi = 1.0 / np.sqrt(1.0 + np.pi * tanthe ** 2.)  # eqn. 12.45a, p. 344
    fg = np.exp(-2.0 * np.tan(phi / 2.0))  # eqn. 12.51, p. 345

    emue0 = chi * (emu[z] + sine * tanthe * e2e / (2.0 - e1e))
    imue0 = chi * (imu[z] + sini * tanthe * e2i / (2.0 - e1i))

    # e >= i

    zz = np.flatnonzero(e >= i)
    zz = np.array(zz)

    if np.size(zz) != 0:
        denom = 2.0 - e1e[zz] - (phi[zz] / np.pi) * e1i[zz]
        imue[zz] = chi[zz] * (
                    imu[z[zz]] + sini[zz] * tanthe[zz] * (cosphi[zz] * e2e[zz] + sinphi2_2[zz] * e2i[zz]) / denom)
        emue[zz] = chi[zz] * (emu[z[zz]] + sine[zz] * tanthe[zz] * (e2e[zz] - sinphi2_2[zz] * e2i[zz]) / denom)
        sfun[zz] = emue[zz] / emue0[zz] * imu[z[zz]] / imue0[zz] * chi[zz] / (
                    1.0 - fg[zz] + fg[zz] * chi[zz] * imu[z[zz]] / imue0[zz])

    # e < i
    zz = np.nonzero(e < i)
    zz = np.array(zz)
    if np.size(zz) != 0:
        denom = 2.0 - e1i[zz] - (phi[zz] / np.pi) * e1e[zz]
        imue[zz] = chi[zz] * (imu[z[zz]] + sini[zz] * tanthe[zz] * (e2i[zz] - sinphi2_2[zz] * e2e[zz]) / denom)
        emue[zz] = chi[zz] * (
                    emu[z[zz]] + sine[zz] * tanthe[zz] * (cosphi[zz] * e2i[zz] + sinphi2_2[zz] * e2e[zz]) / denom)
        sfun[zz] = emue[zz] / emue0[zz] * imu[z[zz]] / imue0[zz] * chi[zz] / (
                    1.0 - fg[zz] + fg[zz] * chi[zz] * emu[z[zz]] / emue0[zz])

    output = {'sfun': sfun, 'imue': imue, 'emue': emue}

    return (output)


# _____ The main function to calculate the reflectance for a given geometry and a given set of Hapke parameters

def hapke_ref(i, e, **kwargs):
    if 'spsf_type' in kwargs:
        spsf_type = np.array(kwargs['spsf_type'])
    else:
        spsf_type = np.array([0])

    if 'spsf_par' in kwargs:
        spsf_par = np.array(kwargs['spsf_par'])
    else:
        spsf_par = np.array([0])

    if 'w' in kwargs:
        w = kwargs['w']
        if not (isinstance(w, np.ndarray)):
            if not (isinstance(w, list)): w = [w]
            w = np.array(w)
    else:
        w = np.array([0.5])

    if 'h1986' in kwargs:
        h1986 = 1
    else:
        h1986 = 0
    if 'h1993' in kwargs:
        h1993 = 1
    else:
        h1993 = 0

    if not (isinstance(i, np.ndarray)):
        if not (isinstance(i, list)): i = [i]
        i = np.array(i)

    if not (isinstance(e, np.ndarray)):
        if not (isinstance(e, list)): e = [e]
        e = np.array(e)

    n = np.size(i)

    mu0 = np.cos(np.radians(i))
    mu = np.cos(np.radians(e))

    if 'phase' in kwargs:
        phase = kwargs['phase']
        if not (isinstance(phase, np.ndarray)):
            if not (isinstance(phase, list)): phase = [phase]
            phase = np.array(phase)
        if np.size(phase) == n:
            phase_ang = np.radians(phase)
        else:
            phase_ang = np.radians(np.repeat(phase[0], n))

    if 'azimuth' in kwargs:
        azimuth = kwargs['azimuth']
        if not (isinstance(azimuth, np.ndarray)):
            if not (isinstance(azimuth, list)): azimuth = [azimuth]
            phase = np.array(azimuth)
        if np.size(azimuth) == n:
            a = azimuth
        else:
            a = np.repeat(azimuth[0], n)
        phase_ang = np.radians(phase_angle(i, e, a))

    if 'phase' not in kwargs and 'azimuth' not in kwargs:
        a = np.repeat(0., n)
        phase_ang = np.radians(phase_angle(i, e, a))

    # Roughness correction

    roughness_correction = 1.

    if ('slope' in kwargs) and (np.array(kwargs['slope']) != 0.):

        slope = kwargs['slope']
        if not (isinstance(slope, np.ndarray)):
            if not (isinstance(slope, list)): slope = [slope]
            slope = np.array(slope)

        if np.size(slope) < n: slope = np.repeat(slope[0], n)

        rc = hapke_roughness(mu, mu0, phase_ang, np.radians(slope))

        roughness_correction = rc['sfun']
        mu0 = rc['imue']
        mu = rc['emue']

    # Porosity correction

    k = 1.

    if 'por_helf2011' in kwargs:

        hs = kwargs['hs']
        if not (isinstance(hs, np.ndarray)):
            if not (isinstance(hs, list)): hs = [hs]
            hs = np.array(hs)

        phi = hs
        ph = 1.209 * (phi ** (2. / 3.))
        k = - np.log(1. - ph) / ph
        hs = -0.3102 * (phi ** (1. / 3.)) * np.log(1. - ph)

    # Opposition effects

    if 'bs' in kwargs or 'bc' in kwargs:

        # CBOE Opposition effect

        if 'bc' in kwargs:

            hc = kwargs['hc']
            if not (isinstance(hc, np.ndarray)):
                if not (isinstance(hc, list)): hc = [hc]
                hc = np.array(hc)

            bc = kwargs['bc']
            if not (isinstance(bc, np.ndarray)):
                if not (isinstance(bc, list)): bc = [bc]
                bc = np.array(bc)

            k_opp = (1. / hc) * np.tan(phase_ang / 2.)

            b_0 = np.repeat(1., n)

            wh = np.flatnonzero(k_opp != 0)

            b_0[wh] = (1. + (1. - np.exp(-k_opp[wh])) / k_opp[wh]) / (2. * (1. + k_opp[wh]) ** 2.)

            bcb = 1. + bc * b_0
        else:
            bcb = 1.

        # SHOE Opposition effect

        if 'bs' in kwargs:

            hs = kwargs['hs']
            if not (isinstance(hs, np.ndarray)):
                if not (isinstance(hs, list)): hs = [hs]
                hs = np.array(hs)

            bs = kwargs['bs']
            if not (isinstance(bs, np.ndarray)):
                if not (isinstance(bs, list)): bs = [bs]
                bs = np.array(bs)

            k_opp = (1. / hs) * np.tan(phase_ang / 2.)
            b_0 = 1. / (1. + k_opp)
            bsh = 1. + bs * b_0
        else:
            bsh = np.repeat(1., n)
    else:
        bsh = np.repeat(1., n)
        bcb = np.repeat(1., n)

    # General formulas for reflectance, in the isotropic and anisotropic MS cases _____

    if 'ams' in kwargs:
        print('Not implemented yet...')
        f = 1.
    #        f = k * (w / (4.*np.pi)) * (mu0/(mu0+mu)) * (single_part_scat_func(phase_ang, CH=spsf_type, PAR=spsf_par, /RADIANS) * bsh $
    #                                       + hapke2002_big_m_2013ap(mu0/k, mu/k, W=w, CH=spsf_type, PAR=spsf_par, BIG_P=big_p, H1986=h1986, H1993=h1993, SLOW=slow, NOCORNT=nocornt)) $
    #                                       * bcb * roughness_correction $
    else:
        f = k * (w / (4. * np.pi)) * (mu0 / (mu0 + mu)) * (
                    single_part_scat_func(phase_ang, ch=spsf_type, par=spsf_par, radians=1) * bsh + h_function(mu0 / k,
                                                                                                               w,
                                                                                                               h1986=h1986,
                                                                                                               h1993=h1993) * h_function(
                mu / k, w, h1986=h1986, h1993=h1993) - 1.) * bcb * roughness_correction

    # Units conversions for the output reflectance values

    r_val = f
    iof_val = np.pi * r_val
    brdf_val = r_val / np.cos(np.radians(i))
    reff_val = np.pi * brdf_val

    f = reff_val

    if 'iof' in kwargs: f = iof_val
    if 'brdf' in kwargs: f = brdf_val
    if 'bid_ref' in kwargs: f = r_val
    if 'reff' in kwargs: f = reff_val

    f = np.squeeze(f)

    return (f)
