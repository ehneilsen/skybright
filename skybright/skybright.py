#!/usr/bin/env python
"""A model for the sky brightness
"""
from math import pi, cos, acos, sin, sqrt, log10
from datetime import datetime, tzinfo, timedelta
from time import strptime
from calendar import timegm
from copy import deepcopy
from string import strip
from sys import argv
from collections import namedtuple, OrderedDict
from argparse import ArgumentParser
from ConfigParser import ConfigParser
import warnings
from warnings import warn

import numpy as np

try:
    from palpy import rdplan as rdplan_rad
    from palpy import gmst as gmst_rad
    from palpy import dmoon 
    from palpy import evp 
    from palpy import dsep as dsep_rad
except ImportError:
    from pyslalib.slalib import sla_rdplan as rdplan_rad
    from pyslalib.slalib import sla_gmst as gmst_rad
    from pyslalib.slalib import sla_dmoon as dmoon
    from pyslalib.slalib import sla_evp as evp
    from pyslalib.slalib import sla_dsep as dsep_rad
    
palpy_body = {'sun': 0,
              'moon': 3}

warnings.simplefilter("always")

def zd_rad(ha, decl, latitude):
    "Calculate the zenith distance in radians, given horizon coordinates"
    cos_zd = np.cos(decl)*np.cos(latitude)*np.cos(ha) + np.sin(decl)*np.sin(latitude)
    zd = np.arccos(cos_zd)
    return zd


def body_zd_rad(body, latitude, longitude, mjd):
    "Calculate the zenith distance in radians, given a body"
    body_id = palpy_body[body]
    ra, decl, diam = rdplan_rad(mjd, body_id, longitude, latitude)
    lst = gmst_rad(mjd) + longitude
    ha = lst - ra
    return zd_rad(ha, decl, latitude)


def body_zd(body, latitude, longitude, mjd):
    """Calculate the zenith distance of a solar system body, in degrees

    Reproduce a value calculated by http://ssd.jpl.nasa.gov/horizons.cgi
    for the moon
    >>> latitude = -30.16527778
    >>> longitude = -70.815
    >>> mjd = 51778.47
    >>> zd = body_zd('moon', latitude, longitude, mjd)
    >>> print "%3.1f" % zd
    48.0

    and for the sun:

    >>> latitude = -30.16527778
    >>> longitude = -70.815
    >>> mjd = 51778.595
    >>> print "%3.1f" % body_zd('sun', latitude, longitude, mjd)
    55.0
    """

    zd_rad = body_zd_rad(body, np.radians(latitude), np.radians(longitude), mjd)
    zd = np.degrees(zd_rad)
    return zd


def elongation_rad(mjd):
    "Calculate the elongation of the moon in radians"
    pv = dmoon(mjd)
    moon_distance = (sum([x**2 for x in pv[:3]]))**0.5
    
    dvb, dpb, dvh, dph = evp(mjd,-1)         
    sun_distance = (sum([x**2 for x in dph[:3]]))**0.5

    a  = np.arccos(
        (-pv[0]*dph[0] - pv[1]*dph[1] - pv[2]*dph[2])/
        (moon_distance*sun_distance))
    return a


def elongation(mjd):
    """Calculate the elongation of the moon

    Reproduce a value calculated by http://ssd.jpl.nasa.gov/horizons.cgi
    >>> mjd = 51778.47
    >>> elong = elongation(mjd)
    >>> print "%3.1f" % elong
    94.0
    """
    a = np.degrees(elongation_rad(mjd))
    return a


def moon_brightness(mjd):
    """The brightness of the moon (relative to full)

    The value here matches about what I expect from the value in 
    Astrophysical Quantities corresponding to the elongation calculated by
    http://ssd.jpl.nasa.gov/horizons.cgi
    >>> mjd = 51778.47
    >>> print "%3.2f" % moon_brightness(mjd)
    0.10
    """
    alpha = 180.0-elongation(mjd)
    # Allen's _Astrophysical Quantities_, 3rd ed., p. 144
    # return 2.512**(-0.026*abs(alpha) - 4E-9*(alpha**4))
    return flux(0.026*abs(alpha) + 4E-9*(alpha**4), 0)


# def body_brightness(mjd, body, sun_m):
#     if body=='moon':
#         return moon_brightness(mjd)
#     elif body=='sun':
#         return 2.512**(-12.74-sun_m)

#     raise NotImplementedError()

def body_brightness(mjd, body, sun_m):
    # moon_max_Vmag = -12.73
    # sun_Vmag = -26.76
    # sun_moon_mag_diff = sun_Vmag - moon_max_Vmag
    
    if body=='moon':
        return moon_brightness(mjd)
    elif body=='sun':
        return flux(sun_m, 0)

    raise NotImplementedError()


def body_twilight_rad(latitude, longitude, mjd, body, twi1=-2.52333, twi2=0.01111):
    z = np.degrees(body_zd_rad(body, latitude, longitude, mjd))
    if z<90:
        return 1.0
    if z>108:
        return 0.0
    if z>100:
        twi0 = -1*(twi1*90+ twi2*90*90)
        logfrac = twi0 + twi1*z + twi2*z*z
    else:
        logfrac = 137.11-2.52333*z+0.01111*z*z

    frac = 10**logfrac
    frac = 1.0 if frac>1.0 else frac
    frac = 0.0 if frac<0.0 else frac
    return frac


def airmass_rad(zd):
    z = min(zd, np.pi/2)
    a = 462.46 + 2.8121/(np.cos(z)**2 + 0.22*np.cos(z) + 0.01)
    x = sqrt( (a*np.cos(z))**2 + 2 * a + 1 ) - a * cos(z)
    return x


def airmass(zd):
    """Calculate the airmass

    Reproduce Bemporad's empirical values (reported in Astrophysical Quantities)
    >>> print "%5.3f" % airmass(0.0)
    1.000
    >>> print "%5.3f" % airmass(45.0)
    1.413
    >>> print "%3.1f" % airmass(80.0)
    5.6
    """
    x = airmass_rad(np.radians(zd))
    return x


def body_airmass_rad(body, latitude, longitude, mjd):
    zd = body_zd_rad(body, latitude, longitude, mjd)
    x = airmass_rad(zd)
    return x

def cosrho_rad(mjd, ra, decl, latitude, longitude, body):
    body_idx = palpy_body[body]
    body_ra, body_decl, diam = rdplan_rad(mjd, body_idx, longitude, latitude)
    rho = dsep_rad(ra, decl, body_ra, body_decl)
    return np.cos(rho)


def cosrho(mjd, ra, decl, latitude, longitude, body):
    """Calculate the cosine of the angular separation between the moon and a point on the sky

    Test with results near and far from the moon position reported by
    http://ssd.jpl.nasa.gov/horizons.cgi
    >>> latitude = -30.16527778
    >>> longitude = -70.815
    >>> mjd = 51778.47
    >>> print "%4.2f" % cosrho(mjd, 51.15, 15.54, latitude, longitude, 'moon')
    1.00
    >>> print "%4.2f" % cosrho(mjd, 51.15, 105.54, latitude, longitude, 'moon')
    0.00
    """
    cr = cosrho_rad(mjd,
                    np.radians(ra),
                    np.radians(decl),
                    np.radians(latitude),
                    np.radians(longitude),
                    body)
    return cr


def mjd(datedict):
    """Convert a dictionary wi/ year, month, day, hour minute to MJD

    >>> testd = {'year': 2000, 'month': 8, 'day': 22, 'hour': 11, 'minute': 17}
    >>> print "%7.2f" % mjd(testd)
    51778.47
    """
    tstring = '%(year)04d-%(month)02d-%(day)02dT%(hour)02d:%(minute)02d:00Z' % datedict
    d = strptime(tstring,'%Y-%m-%dT%H:%M:%SZ')
    posixtime = timegm(d)
    mjd = 40587.0+posixtime/86400.0
    return mjd


def magadd(m1, m2):
    """Add the flux corresponding to two magnitudes, and return the corresponding magnitude
    """
    return -2.5*log10( 10**(-0.4*m1) + 10**(-0.4*m2))


def magn(f, m0=23.9):
    """Return the AB magnitude corresponding to the given flux in microJanskys
    """
    if f <= 0:
        #return 99.9
        return np.nan
    return m0 - 2.5*log10( f )


def flux(m, m0=23.9):
    return 10**(-0.4*(m-m0))


def airglowshell_rad(mzen, h, ra, decl, mjd, k, latitude, longitude, r0=6375.0):
    lst = gmst_rad(mjd) + longitude
    ha = lst - ra
    z = zd_rad(ha, decl, latitude)
    x = airmass_rad(z)
    mag = mzen + 1.25*log10(1.0-(r0/(h+r0))*(sin(z))**2) + k*(x-1)
    return mag


def airglowshell(mzen, h, ra, decl, mjd, k, latitude, longitude, r0=6375.0):
    """Return the surface brightness from an airglow shell

    >>> latitude = -30.16527778
    >>> longitude = -70.815
    >>>
    >>> ra = 28.71208
    >>> decl = 0.74225
    >>> mjd = 51808.33
    >>> k = 0.0583989
    >>> 
    >>> mzen = 20.15215
    >>> h = 300.0
    >>> m_inf = 22.30762
    >>> print "%3.1f" % magadd(m_inf, airglowshell(mzen, h, ra, decl, mjd, k, latitude, longitude))
    19.8
    """
    mag = airglowshell_rad(mzen, h, np.radians(ra), np.radians(decl), mjd, k, np.radians(latitude), np.radians(longitude), r0=r0)
    return mag


def bodyterm2_rad(ra, decl, mjd, k, latitude, longitude, body, twi1, twi2):
    lst = gmst_rad(mjd) + longitude
    ha = lst - ra
    z = zd_rad(ha, decl, latitude)
    x = airmass_rad(z)
    xm = body_airmass_rad(body, latitude, longitude, mjd)
    term = (10**(-0.4*k*x)-10**(-0.4*k*xm))/(-0.4*k*(x-xm))
    term = term * body_twilight_rad(latitude, longitude, mjd, body, twi1, twi2)
    return term


def bodyterm2(ra, decl, mjd, k, latitude, longitude, body, twi1, twi2):
    """The term in the scattered light function common to Mie and Rayleigh scattering

    >>> latitude = -30.16527778
    >>> longitude = -70.815
    >>> 
    >>> ra = 28.71208
    >>> decl = 0.74225
    >>> mjd = 51808.33
    >>> k = 0.0583989
    >>> twi1, twi2 = -2.52333, 0.01111
    >>> print "%4.2f" % bodyterm2(ra, decl, mjd, k, latitude, longitude, 'moon', twi1, twi2)
    2.08
    """
    term = bodyterm2_rad(np.radians(ra),
                         np.radians(decl),
                         mjd,
                         k,
                         np.radians(latitude),
                         np.radians(longitude),
                         body,
                         twi1,
                         twi2)
    return term


def rayleigh_frho(mjd, ra, decl, latitude, longitude, body):
    """Calculate the Rayleigh scattered light

    >>> latitude = -30.16527778
    >>> longitude = -70.815
    >>> 
    >>> ra = 28.71208
    >>> decl = 0.74225
    >>> mjd = 51808.33
    >>>
    >>> print "%4.3f" % rayleigh_frho(mjd, ra, decl, latitude, longitude, 'moon')
    0.874
    """
    mu = cosrho(mjd, ra, decl, latitude, longitude, body)
    return 0.75*(1.0+mu**2)


def rayleigh(m, ra, decl, mjd, k, latitude, longitude, body, sun_m, twi1, twi2):
    """Calculate the Rayleigh scattered light

    >>> latitude = -30.16527778
    >>> longitude = -70.815
    >>> 
    >>> ra = 28.71208
    >>> decl = 0.74225
    >>> mjd = 51808.33
    >>> k = 0.0583989
    >>>
    >>> m = -4.2843
    >>> sun_m, twi1, twi2 = -26.74, -2.52333, 0.01111
    >>> print "%4.2f" % rayleigh(m, ra, decl, mjd, k, latitude, longitude, 'moon', sun_m, twi1, twi2)    
    21.71
    """
    term1 = flux(m + magn(
            rayleigh_frho(mjd, ra, decl, latitude, longitude, body)))
    term2 = bodyterm2(ra, decl, mjd, k, latitude, longitude, body, twi1, twi2)
    return magn(term1 * term2 * body_brightness(mjd, body, sun_m))


def mie_frho(g, mjd, ra, decl, latitude, longitude, body):
    """Calculate the Mie scattered light

    >>> latitude = -30.16527778
    >>> longitude = -70.815
    >>> 
    >>> ra = 28.71208
    >>> decl = 0.74225
    >>> mjd = 51808.33
    >>>
    >>> g = 0.65
    >>> print "%3.2f" % mie_frho(g, mjd, ra, decl, latitude, longitude, 'moon')
    0.38
    """
    mu = cosrho(mjd, ra, decl, latitude, longitude, body)
    return 1.5*((1.0-g**2)/(2.0+g**2)) * (1.0 + mu) * (1.0 + g**2 - 2.0*g*mu*mu)**(-1.5)


def mie(g, c, ra, decl, mjd, k, latitude, longitude, body, sun_m, twi1, twi2):
    term1 = mie_frho(g, mjd, ra, decl, latitude, longitude, body)
    term2 = bodyterm2(ra, decl, mjd, k, latitude, longitude, body, twi1, twi2)
    return magn(c * term1 * term2 * body_brightness(mjd, body, sun_m))


def skymag(m_inf, m_zen, h, g, mie_c, rayl_m, ra, decl, mjd, k, latitude, longitude, offset=0.0,
           sun_m=-14.0, twi1=-2.52333, twi2=0.01111):
    """Calculate the total surface brightness of the sky

    >>> latitude = -30.16527778
    >>> longitude = -70.815
    >>> 
    >>> ra = 36.0
    >>> decl = -30.0
    >>> mjd = 58000.3
    >>> k = 0.08
    >>>
    >>> m_zen = 21.4
    >>> h = 90.0
    >>> m_inf = 30.0
    >>> rayl_m = -3.8
    >>> g = 0.74
    >>> mie_c = 62.0
    >>> sun_m, twi1, twi2 = -14.0, -1.0, 0.003
    >>> print "%4.2f" % skymag(m_inf, m_zen, h, g, mie_c, rayl_m, ra, decl, mjd, k, latitude, longitude, 0.0, sun_m, twi1, twi2)
    19.49
    """

    zd = np.degrees(zd_rad(gmst_rad(mjd) + np.radians(longitude) - np.radians(ra),
                           np.radians(decl),
                           np.radians(latitude)))
    if zd > 90:
        raise NotImplementedError("Sky model does not work for pointings below the horizon!")
    
    if body_zd('sun', latitude, longitude, mjd) < 98:
        raise NotImplementedError("Sky model does not work during the day")

    mags = [m_inf,
            airglowshell(m_zen, h, ra, decl, mjd, k, latitude, longitude)]
    
    if body_zd('moon', latitude, longitude, mjd) < 107.8:
        mags += [rayleigh(rayl_m, ra, decl, mjd, k, latitude, longitude, 'moon', sun_m, twi1, twi2),
                 mie(g, mie_c, ra, decl, mjd, k, latitude, longitude, 'moon', sun_m, twi1, twi2)]

    if body_zd('sun', latitude, longitude, mjd) < 107.8:
        mags += [rayleigh(rayl_m, ra, decl, mjd, k, latitude, longitude, 'sun', sun_m, twi1, twi2),
                 mie(g, mie_c, ra, decl, mjd, k, latitude, longitude, 'sun', sun_m, twi1, twi2)]

    m = reduce(magadd, mags)
    m = m + offset
    return m

class MoonSkyModel(object):
    def __init__(self, model_config):
        self.longitude = model_config.getfloat("Observatory Position",
                                               "longitude")
        self.latitude = model_config.getfloat("Observatory Position",
                                              "latitude")
        self.elevation = model_config.getfloat("Observatory Position",
                                               "elevation")

        self.k = OrderedDict()
        self.m_inf = OrderedDict()
        self.m_zen = OrderedDict()
        self.h = OrderedDict()
        self.rayl_m = OrderedDict()
        self.g = OrderedDict()
        self.mie_c = OrderedDict()
        self.offset = OrderedDict()
        self.sun_m = OrderedDict()
        self.twi1 = OrderedDict()
        self.twi2 = OrderedDict()

        for i, band in enumerate(model_config.get("sky","filters").split()):
            i = model_config.get("sky","filters").split().index(band)
            self.k[band] = float(model_config.get("sky","k").split()[i])
            self.m_inf[band] = float(model_config.get("sky","m_inf").split()[i])
            self.m_zen[band] = float(model_config.get("sky","m_zen").split()[i])
            self.h[band] = float(model_config.get("sky","h").split()[i])
            self.rayl_m[band] = float(model_config.get("sky","rayl_m").split()[i])
            self.g[band] = float(model_config.get("sky","g").split()[i])
            self.mie_c[band] = float(model_config.get("sky","mie_c").split()[i])
            self.offset[band] = 0.0
            self.sun_m[band] = float(model_config.get("sky","sun_m").split()[i])
            self.twi1[band] = float(model_config.get("sky","twi1").split()[i])
            self.twi2[band] = float(model_config.get("sky","twi2").split()[i])

    def __call__(self, mjd, ra, decl, band):
        try:
            m = skymag(self.m_inf[band], self.m_zen[band], self.h[band], 
                       self.g[band], self.mie_c[band], self.rayl_m[band], 
                       ra, decl, mjd, 
                       self.k[band], self.latitude, self.longitude, 
                       self.offset[band],
                       self.sun_m[band], self.twi1[band], self.twi2[band])
        except Exception, e:
            warn("Bad sky magnitude for mjd=%f, ra=%f, decl=%f, filter=%s: %s" % (
                mjd, ra, decl, band, str(e)))
            m=np.nan
        return m
        
    def dark_skymag(self, band):
        dsm = magadd(self.m_inf[band], self.m_zen[band])
        return dsm

    def dark_skymag_diff(self, band):
        delta_skymag = self.skymag[band]-self.dark_skymag[band]
        return delta_skymag

    @property
    def down(self, mjd):
        """Return true iff the both sun and moon are down
        """
        return moon_zd(self.latitude, self.longitude, mjd) > 108.0 and \
            body_zd('sun', self.latitude, self.longitude, mjd) > 108.0

if __name__=='__main__':
    parser = ArgumentParser('Estimate the sky brightness')
    parser.add_argument("-m", "--mjd", type=float,
                        help="Modified Julian Date (float) (UTC)")
    parser.add_argument("-r", "--ra", type=float,
                        help="the RA (decimal degrees)")
    parser.add_argument("-d", "--dec", type=float,
                        help="the declination (decimal degrees)")
    parser.add_argument("-f", "--filter", 
                        help="the filter")
    parser.add_argument("-c", "--config",
                        help="the configuration file")

    args = parser.parse_args()

    model_config = ConfigParser() 
    model_config.read(args.config)

    longitude = model_config.getfloat("Observatory Position",
                                      "longitude")
    latitude = model_config.getfloat("Observatory Position",
                                     "latitude")
    elevation = model_config.getfloat("Observatory Position",
                                      "elevation")

    print "Moon zenith distance: %f" % body_zd('moon', latitude, longitude, args.mjd)
    print "Sun zenith distance: %f" % body_zd('sun', latitude, longitude, args.mjd)
    print "Elongation of the moon: %f" % elongation(args.mjd)
    print "Moon brightness: %f" % moon_brightness(args.mjd)
    print "Pointing angle with moon: %f" % np.degrees(np.arccos(
        cosrho(args.mjd, args.ra, args.dec, latitude, longitude, 'moon')))
    

    lst = np.degrees(gmst_rad(args.mjd)) + longitude
    ha = lst - args.ra
    z = np.degrees(zd_rad(np.radians(ha), np.radians(args.dec), np.radians(latitude)))
    print "Pointing zenith distance: %f" % z
    print "Airmass: %f" % airmass(z)

    sky_model = MoonSkyModel(model_config)
    
    print "Sky brightness at pointing: %f" % sky_model(args.mjd, args.ra, args.dec, args.filter)

