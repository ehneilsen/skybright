#!/usr/bin/env python
"""A model for the sky brightness
"""
from functools import partial
from math import pi, cos, acos, sin, sqrt, log10
from datetime import datetime, tzinfo, timedelta
from time import strptime
from calendar import timegm
from copy import deepcopy
from sys import argv
from collections import namedtuple, OrderedDict
from argparse import ArgumentParser
try:
    from ConfigParser import ConfigParser
except:
    from configparser import ConfigParser
    
import numexpr
from numexpr import NumExpr
import warnings
from warnings import warn

import numpy as np

try:
    from palpy import rdplan as rdplan_not_vectorized
    from palpy import gmst as gmst_not_vectorized
    from palpy import dmoon 
    from palpy import evp 
except ImportError:
    from pyslalib.slalib import sla_rdplan as rdplan_not_vectorized
    from pyslalib.slalib import sla_gmst as gmst_not_vectorized
    from pyslalib.slalib import sla_dmoon as dmoon
    from pyslalib.slalib import sla_evp as evp
    
palpy_body = {'sun': 0,
              'moon': 3}

MAG0 = 23.9

# warnings.simplefilter("always")

rdplan = np.vectorize(rdplan_not_vectorized)

def gmst(mjd):
    # Follow Meeus chapter 12
    big_t = numexpr.evaluate("(mjd - 51544.5)/36525")
    st = np.radians(np.mod(numexpr.evaluate("280.46061837 + 360.98564736629*(mjd-51544.5) + 0.000387933*big_t*big_t - big_t*big_t*big_t/38710000"), 360))
    return st
    
def ang_sep(ra1, decl1, ra2, decl2):
    # haversine formula
    return numexpr.evaluate("2*arcsin(sqrt(cos(decl1)*cos(decl2)*(sin(((ra1-ra2)/2))**2) + (sin((decl1-decl2)/2))**2))")

## Works and is trivially faster, but less flexible w.r.t. data types
#
# ang_sep = NumExpr("2*arcsin(sqrt(cos(decl1)*cos(decl2)*(sin(((ra1-ra2)/2))**2) + (sin((decl1-decl2)/2))**2))",
#                  (('ra1', np.float64), ('decl1', np.float64), ('ra2', np.float64), ('decl2', np.float64)))

def calc_zd(latitude, ha, decl):
    # zenith is always at ha=0, dec=latitude, by defn.
    return ang_sep(ha, decl, 0, latitude)

def calc_airmass(cos_zd):
    a = numexpr.evaluate("462.46 + 2.8121/(cos_zd**2 + 0.22*cos_zd + 0.01)")
    airmass = numexpr.evaluate("sqrt((a*cos_zd)**2 + 2*a + 1) - a * cos_zd")
    airmass[cos_zd < 0] = np.nan
    return airmass

def calc_airglow(r0, h, m_zen, k, sin_zd, airmass):
    airglow = numexpr.evaluate("10**(-0.4*(m_zen + 1.25*log10(1.0 - (r0/(h+r0))*(sin_zd**2)) + k*(airmass-1) - MAG0))")
    return airglow

def calc_scat_extinction(k, x0, x):
    if len(np.shape(x0)) == 0:
        x0p = calc_airmass(0) if np.isnan(x0) else x0
    else:
        x0p = np.where(np.isnan(x0), calc_airmass(0), x0)
        
    extinct = numexpr.evaluate("(10**(-0.4*k*x) - 10**(-0.4*k*x0p))/(-0.4*k*(x-x0p))")
    return extinct

def elongation_not_vectorized(mjd):
    "Calculate the elongation of the moon in radians"
    pv = dmoon(mjd)
    moon_distance = (sum([x**2 for x in pv[:3]]))**0.5
    
    dvb, dpb, dvh, dph = evp(mjd,-1)         
    sun_distance = (sum([x**2 for x in dph[:3]]))**0.5

    a  = np.degrees(np.arccos(
        (-pv[0]*dph[0] - pv[1]*dph[1] - pv[2]*dph[2])/
        (moon_distance*sun_distance)))
    return a

elongation = np.vectorize(elongation_not_vectorized)

def calc_moon_brightness(mjd):
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
    return 10**(-0.4*(0.026*abs(alpha) + 4E-9*(alpha**4)))


def one_calc_twilight_fract(z, twi1=-2.52333, twi2=0.01111):
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

def calc_twilight_fract(zd, twi1=-2.52333, twi2=0.01111):
    z = zd if len(np.shape(zd)) > 0 else np.array(zd)

    logfrac = numexpr.evaluate("137.11-2.52333*z+0.01111*z*z")
    logfrac[z>100] = numexpr.evaluate("twi1*z + twi2*z*z - (twi1*90 + twi2*90*90)")[z>100]
    frac = 10**logfrac
    frac = np.where(z<90, 1.0, frac)
    frac = np.where(z>108, 0.0, frac)
    frac = np.where(frac>1.0, 1.0, frac)
    frac = np.where(frac<0.0, 0.0, frac)
    return frac


def calc_body_scattering(brightness, body_zd_deg, cos_zd, body_ra, body_decl, ra, decl,
                         twi1, twi2, k, airmass, body_airmass, rayl_m, mie_m, g,
                         rayleigh=True, mie=True):
    if len(np.shape(brightness)) == 0:
        brightness = np.array(brightness)

    brightness = np.where(body_zd_deg > 107.8, 0, brightness)

    body_twi = body_zd_deg > 90
    brightness[body_twi] = brightness[body_twi]*calc_twilight_fract(body_zd_deg[body_twi], twi1, twi2)

    extinct = calc_scat_extinction(k, body_airmass, airmass)

    cos_rho = numexpr.evaluate("cos(2*arcsin(sqrt(cos(decl)*cos(body_decl)*(sin(((ra-body_ra)/2))**2) + (sin((decl-body_decl)/2))**2)))")

    rayleigh_frho = numexpr.evaluate("0.75*(1.0+cos_rho**2)") if rayleigh else np.zeros_like(cos_rho)
    mie_frho =  numexpr.evaluate("1.5*((1.0-g**2)/(2.0+g**2)) * (1.0 + cos_rho) * (1.0 + g**2 - 2.0*g*cos_rho*cos_rho)**(-1.5)") if mie else np.zeros_like(cos_rho)
    mie_frho = np.where(mie_frho<0, 0.0, mie_frho)
    
    # Fitter sometimes explores values of g resulting mie_frho being negative.
    # Force a physical result.
    mie_frho = np.where(mie_frho<0, 0.0, mie_frho)
    
    rayl_c = 10**(-0.4*(rayl_m-MAG0))
    mie_c = 10**(-0.4*(mie_m-MAG0))
    flux = brightness*extinct*(rayl_c*rayleigh_frho + mie_c*mie_frho)

    return flux

class MoonSkyModel(object):
    def __init__(self, model_config):
        self.longitude = model_config.getfloat("Observatory Position",
                                               "longitude")
        self.latitude = model_config.getfloat("Observatory Position",
                                              "latitude")

        self.k = OrderedDict()
        self.m_inf = OrderedDict()
        self.m_zen = OrderedDict()
        self.h = OrderedDict()
        self.rayl_m = OrderedDict()
        self.g = OrderedDict()
        self.mie_m = OrderedDict()
        self.offset = OrderedDict()
        self.sun_dm = OrderedDict()
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
            self.mie_m[band] = float(model_config.get("sky","mie_m").split()[i])
            self.offset[band] = 0.0
            self.sun_dm[band] = float(model_config.get("sky","sun_dm").split()[i])
            self.twi1[band] = float(model_config.get("sky","twi1").split()[i])
            self.twi2[band] = float(model_config.get("sky","twi2").split()[i])

        self.calc_zd = partial(calc_zd, np.radians(self.latitude))
        self.r0 = 6375.0
        self.twilight_nan = True

    def __call__(self, mjd, ra_deg, decl_deg, band, sun=True, moon=True):
        if len(np.shape(band)) < 1:
            return self.single_band_call(mjd, ra_deg, decl_deg, band, sun=sun, moon=moon)

        mags = np.empty_like(ra_deg, dtype=np.float64)
        mags.fill(np.nan)

        for this_band in np.unique(band):
            these = band == this_band
            mjd_arg = mjd if len(np.shape(mjd))==0 else mjd[these]
            mags[these] = self.single_band_call(mjd_arg, ra_deg[these], decl_deg[these], this_band, sun=sun, moon=moon)

        return mags
            
        
    def single_band_call(self, mjd, ra_deg, decl_deg, band, sun=True, moon=True):
        longitude = np.radians(self.longitude)
        latitude = np.radians(self.latitude)

        ra = np.radians(ra_deg)
        decl = np.radians(decl_deg)
        k = self.k[band]
        twi1 = self.twi1[band]
        twi2 = self.twi2[band]
        m_inf = self.m_inf[band]
        
        lst = gmst(mjd) + longitude
        ha = lst - ra
        sun_ra, sun_decl, diam = rdplan(mjd, 0, longitude, latitude)
        sun_ha = lst - sun_ra
        sun_zd = self.calc_zd(sun_ha, sun_decl)
        sun_zd_deg = np.degrees(sun_zd)
        if len(np.shape(sun_zd_deg)) == 0 and self.twilight_nan:
            if sun_zd_deg < 98:
                m = np.empty_like(ra)
                m.fill(np.nan)
                return m

        sun_cos_zd = np.cos(sun_zd)
        sun_airmass = calc_airmass(sun_cos_zd)
        
        moon_ra, moon_decl, diam = rdplan(mjd, 3, longitude, latitude)
        moon_ha = lst - moon_ra
        moon_zd = self.calc_zd(moon_ha, moon_decl)
        moon_cos_zd = np.cos(moon_zd)
        moon_airmass = calc_airmass(moon_cos_zd)
        moon_zd_deg = np.degrees(moon_zd)

        # Flux from infinity
        sky_flux = np.empty_like(ra)
        sky_flux.fill(10**(-0.4*(m_inf-MAG0)))
        
        # Airglow
        zd = self.calc_zd(ha, decl)
        sin_zd = np.sin(zd)
        cos_zd = np.cos(zd)
        airmass = calc_airmass(cos_zd)
        airglow_flux = calc_airglow(self.r0, self.h[band], self.m_zen[band], k, sin_zd, airmass)

        sky_flux += airglow_flux
        
        # Needed for both scattering calculations
        zd_deg = np.degrees(zd)

        # Add scattering of moonlight
        if moon:
            moon_flux = calc_body_scattering(
                calc_moon_brightness(mjd),
                moon_zd_deg, cos_zd, moon_ra, moon_decl, ra, decl, twi1, twi2, k, airmass, moon_airmass,
                self.rayl_m[band], self.mie_m[band], self.g[band])

            sky_flux += moon_flux

        # Add scattering of sunlight
        if sun:
            sun_flux = calc_body_scattering(
                10**(-0.4*(self.sun_dm[band])),
                sun_zd_deg, cos_zd, sun_ra, sun_decl, ra, decl, twi1, twi2, k, airmass, sun_airmass,
                self.rayl_m[band], self.mie_m[band], self.g[band])

            sky_flux += sun_flux
        
        m = MAG0 - 2.5*np.log10(sky_flux)

        if len(np.shape(m)) > 0 and self.twilight_nan:
            m[sun_zd_deg < 98] = np.nan

        return m
#
# Included for backword compatibility with previous implementation
#
def skymag(m_inf, m_zen, h, g, mie_m, rayl_m, ra, decl, mjd, k, latitude, longitude, offset=0.0,
           sun_dm=-14.0, twi1=-2.52333, twi2=0.01111):
    config = ConfigParser()

    sect = "Observatory Position"
    config.add_section(sect)
    config.set(sect, 'longitude', longitude)
    config.set(sect, 'latitude', latitude)

    sect = "sky"
    config.add_section(sect)
    config.set(sect, 'filters', 'x')
    config.set(sect, 'k', k)
    config.set(sect, 'm_inf', m_inf)
    config.set(sect, 'm_zen', m_zen)
    config.set(sect, 'h', h)
    config.set(sect, 'rayl_m', rayl_m)
    config.set(sect, 'g', g)
    config.set(sect, 'mie_m', mie_m)
    config.set(sect, 'sun_dm', sun_dm)
    config.set(sect, 'twi1', twi1)
    config.set(sect, 'twi2', twi2)

    calc_sky = MoonSkyModel(config)
    sky = calc_sky(mjd, ra, decl, 'x')
    return sky

    
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

    lst = gmst(args.mjd) + np.radians(longitude)
    sun_ra, sun_decl, diam = rdplan(args.mjd, 0, np.radians(longitude), np.radians(latitude))
    sun_ha = lst - sun_ra
    sun_zd = np.degrees(calc_zd(np.radians(latitude), sun_ha, sun_decl))
    print("Sun zenith distance: %f" % sun_zd)

    moon_ra, moon_decl, diam = rdplan(args.mjd, 3, longitude, latitude)
    moon_ha = lst - moon_ra
    moon_zd = np.degrees(calc_zd(np.radians(latitude), moon_ha, moon_decl))
    print("Moon zenith distance: %f" % moon_zd)

    print("Elongation of the moon: %f" % elongation(args.mjd))
    print("Moon brightness: %f" % calc_moon_brightness(args.mjd))


    sep = ang_sep(moon_ra, moon_decl, np.radians(args.ra), np.radians(args.dec))
    print("Pointing angle with moon: %f" % sep)

    ha = lst - np.radians(args.ra)
    z = calc_zd(np.radians(latitude), ha, np.radians(args.dec))
    print("Pointing zenith distance: %f" % np.degrees(z))
    print("Airmass: %f" % calc_airmass(np.cos(z)))

    sky_model = MoonSkyModel(model_config)
    
    print("Sky brightness at pointing: %f" % sky_model(args.mjd, args.ra, args.dec, args.filter))
