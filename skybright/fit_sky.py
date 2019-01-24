import csv
import sys
from argparse import ArgumentParser
from ConfigParser import ConfigParser
from collections import namedtuple
import readline
import code
import numpy as np
import pandas as pd
from math import acos, degrees
from lmfit import minimize, Parameters, Parameter, report_errors
from skybright import calc_zd, rdplan, gmst, ang_sep, calc_moon_brightness, MoonSkyModel, MAG0

latitude = -30.16527778
longitude = -70.815
    
def residuals(params, data):
    dummy_band = 'x'
    config = ConfigParser()

    sect = "Observatory Position"
    config.add_section(sect)
    config.set(sect, 'longitude', str(longitude))
    config.set(sect, 'latitude', str(latitude))

    sect = "sky"
    config.add_section(sect)
    config.set(sect, 'filters', str(dummy_band))
    config.set(sect, 'k', str(float(params['k'])))
    config.set(sect, 'm_inf', str(float(params['m_inf'])))
    config.set(sect, 'm_zen', str(float(params['m_zen'])))
    config.set(sect, 'h', str(float(params['h'])))
    config.set(sect, 'rayl_m', str(float(params['rayl_m'])))
    config.set(sect, 'g', str(float(params['g'])))
    config.set(sect, 'mie_m', str(float(params['mie_m'])))
    config.set(sect, 'sun_dm', str(float(params['sun_dm'])))
    config.set(sect, 'twi1', str(float(params['twi1'])))
    config.set(sect, 'twi2', str(float(params['twi2'])))

    calc_sky = MoonSkyModel(config)
    calc_sky.twilight_nan = False
    calc_sky.k[dummy_band] = data.k.values
    
    modmag = calc_sky(data.mjd.values,
                      data.telra.values,
                      data.teldec.values,
                      dummy_band)
    datamag = data.skymag.values

    resid = modmag-datamag
    mad = np.median(abs(resid))
    stdev = 1.4826*mad
    clip_lim = 3*stdev
    resid[resid>clip_lim] = clip_lim
    resid[resid<(-1*clip_lim)] = -1*clip_lim

    return (modmag-datamag)


def init_params(filter_name, cfg):
    i = cfg.get("sky","filters").split().index(filter_name)

    params = Parameters()
    params.add('k', 
               value = float(cfg.get("sky","k").split()[i]),
               min=0.0, max=5.0)
    params.add('m_inf', 
               value = float(cfg.get("sky","m_inf").split()[i]),
               min=15.0, max=30.0)
    params.add('m_zen', 
               value = float(cfg.get("sky","m_zen").split()[i]),
               min=15.0, max=30.0)
    params.add('h',
               value = float(cfg.get("sky","h").split()[i]),
               min=60, max=3000)
    params.add('g', 
               value = float(cfg.get("sky","g").split()[i]),
               min=0.0, max=1.0)
    params.add('mie_m', 
               value = float(cfg.get("sky","mie_m").split()[i]),
               min=10.0, max=30.0)
    params.add('rayl_m', value = float(cfg.get("sky","rayl_m").split()[i]),
               min=10.0, max=30.0)
    params.add('sun_dm', value = float(cfg.get("sky","sun_dm").split()[i]),
               min=-20.0, max=-10.0)
    params.add('twi1', value = float(cfg.get("sky","twi1").split()[i]),
               min=-100.0, max=100.0)
    params.add('twi2', value = float(cfg.get("sky","twi2").split()[i]),
               min=-100.0, max=100.0)
    return params

def fit_dark_sky_filter(all_data, in_params, filter_name):
    params = Parameters()
    params.add('k', 
               value = in_params['k'].value,
               min=0.0, max=5.0, vary=False)
    params.add('m_inf', 
               value = in_params['m_inf'].value,
               min=20.0, max=30.0, vary=True)
    params.add('m_zen', 
               value =  in_params['m_zen'].value,
               min=15.0, max=30.0)
    params.add('h',
               value =  in_params['h'].value,
               min=60, max=3000, vary=False)
    params.add('g', 
               value =  in_params['g'].value,
               min=0.0, max=1.0, vary=False)
    params.add('mie_m', 
               value =  in_params['mie_m'].value,
               min=10.0, max=30.0, vary=False)
    params.add('rayl_m',  in_params['rayl_m'].value,
               min=10.0, max=30.0, vary=False)
    params.add('sun_dm',  in_params['sun_dm'].value,
               min=-20.0, max=-10.0, vary=False)
    params.add('twi1', value = in_params['twi1'],
               min=-100.0, max=100.0, vary=False)
    params.add('twi2', value = in_params['twi2'],
               min=-100.0, max=100.0, vary=False)

    data = all_data.query("band == '%s' and airmass<2.0 and moon_zd>108.0 and sun_zd>108.0" % filter_name)

    fit = minimize(residuals, params, args=(data,))

    report_errors(fit.params)
    return fit.params

def fit_bright_sky_filter(all_data, in_params, filter_name):
    params = Parameters()
    params.add('k', 
               value = in_params['k'].value,
               min=0.0, max=5.0, vary=False)
    params.add('m_inf', 
               value = in_params['m_inf'].value,
               min=15.0, max=30.0, vary=False)
    params.add('m_zen', 
               value =  in_params['m_zen'].value,
               min=15.0, max=30.0, vary=False)
    params.add('h',
               value =  in_params['h'].value,
               min=60, max=3000, vary=False)
    params.add('g', 
               value =  in_params['g'].value,
               min=0.0, max=1.0, vary=True)
    params.add('mie_m', 
               value =  in_params['mie_m'].value,
               min=10.0, max=30.0, vary=True)
    params.add('rayl_m',  in_params['rayl_m'].value,
               min=10.0, max=30.0, vary=True)
    params.add('sun_dm',  in_params['sun_dm'].value,
               min=-20.0, max=-10.0, vary=False)
    params.add('twi1', value = in_params['twi1'],
               min=-100.0, max=100.0, vary=False)
    params.add('twi2', value = in_params['twi2'],
               min=-100.0, max=100.0, vary=False)
    
    data = all_data.query("band == '%s' and airmass<2.0 and moon_brightness>0.0 and moon_zd<80.0 and sun_zd>108.0" % filter_name)

    fit = minimize(residuals, params, args=(data,))

    report_errors(fit.params)
    return fit.params

def fit_rayl_sky_filter(all_data, in_params, filter_name, rayl_angle):
    params = Parameters()
    params.add('k', 
               value = in_params['k'].value,
               min=0.0, max=5.0, vary=False)
    params.add('m_inf', 
               value = in_params['m_inf'].value,
               min=15.0, max=30.0, vary=False)
    params.add('m_zen', 
               value =  in_params['m_zen'].value,
               min=15.0, max=30.0, vary=False)
    params.add('h',
               value =  in_params['h'].value,
               min=60, max=3000, vary=False)
    params.add('g', 
               value =  in_params['g'].value,
               min=0.0, max=1.0, vary=False)
    params.add('mie_m', 
               value =  in_params['mie_m'].value,
               min=10.0, max=30.0, vary=False)
    params.add('rayl_m',  in_params['rayl_m'].value,
               min=10.0, max=30.0, vary=True)
    params.add('sun_dm',  in_params['sun_dm'].value,
               min=-20.0, max=-10.0, vary=False)
    params.add('twi1', value = in_params['twi1'],
               min=-100.0, max=100.0, vary=False)
    params.add('twi2', value = in_params['twi2'],
               min=-100.0, max=100.0, vary=False)

    data = all_data.query("band == '%s' and airmass<2.0 and moon_brightness>0.0 and moon_angle>%f and moon_zd<80.0 and sun_zd>108.0" % (filter_name, rayl_angle))

    print "Fitting to %d points" % len(data)
    fit = minimize(residuals, params, args=(data,))

    report_errors(fit.params)
    return fit.params

def fit_mie_sky_filter(all_data, in_params, filter_name, mie_angle):
    params = Parameters()
    params.add('k', 
               value = in_params['k'].value,
               min=0.0, max=5.0, vary=False)
    params.add('m_inf', 
               value = in_params['m_inf'].value,
               min=15.0, max=30.0, vary=False)
    params.add('m_zen', 
               value =  in_params['m_zen'].value,
               min=15.0, max=30.0, vary=False)
    params.add('h',
               value =  in_params['h'].value,
               min=80, max=3000, vary=False)
    params.add('g', 
               value =  in_params['g'].value,
               min=0.0, max=1.0, vary=True)
    params.add('mie_m', 
               value =  in_params['mie_m'].value,
               min=10.0, max=30.0, vary=True)
    params.add('rayl_m',  in_params['rayl_m'].value,
               min=10.0, max=30.0, vary=False)
    params.add('sun_dm',  in_params['sun_dm'].value,
               min=-20.0, max=-10.0, vary=False)
    params.add('twi1', value = in_params['twi1'],
               min=-100.0, max=100.0, vary=False)
    params.add('twi2', value = in_params['twi2'],
               min=-100.0, max=100.0, vary=False)

    data = all_data.query("band == '%s' and airmass<2.0 and moon_brightness>0.0 and moon_angle<%f and moon_zd<80.0 and sun_zd>108.0" % (filter_name, mie_angle))

    fit = minimize(residuals, params, args=(data,))
        
    report_errors(fit.params)
    return fit.params

def fit_twilight(all_data, in_params, filter_name):
    params = Parameters()
    params.add('k', 
               value = in_params['k'].value,
               min=0.0, max=5.0, vary=False)
    params.add('m_inf', 
               value = in_params['m_inf'].value,
               min=15.0, max=30.0, vary=False)
    params.add('m_zen', 
               value =  in_params['m_zen'].value,
               min=15.0, max=30.0, vary=False)
    params.add('h',
               value =  in_params['h'].value,
               min=80, max=3000, vary=False)
    params.add('g', 
               value =  in_params['g'].value,
               min=0.0, max=1.0, vary=False)
    params.add('mie_m', 
               value =  in_params['mie_m'].value,
               min=10.0, max=30.0, vary=False)
    params.add('rayl_m',  in_params['rayl_m'].value,
               min=10.0, max=30.0, vary=False)
    params.add('sun_dm',  in_params['sun_dm'].value,
               min=-20.0, max=-10.0, vary=True)
    params.add('twi1', value = in_params['twi1'],
               min=-100.0, max=100.0, vary=True)
    params.add('twi2', value = in_params['twi2'],
               min=-100.0, max=100.0, vary=True)

    data = all_data.query("band == '%s' and airmass < 2.0 and moon_zd > 108.0 and sun_zd < 106.0" % filter_name) 

    fit = minimize(residuals, params, args=(data,))

    report_errors(fit.params)
    return fit.params

if __name__=='__main__':
    parser = ArgumentParser('Fit sky brightness model parames to data')
    parser.add_argument("-c", "--config",
                        help="the configuration file")

    args = parser.parse_args()

    cfg = ConfigParser()
    cfg.read(args.config)

    latitude = cfg.getfloat('Observatory Position', 'latitude')
    longitude = cfg.getfloat('Observatory Position', 'longitude')

    data_fname = cfg.get('Measured data', 'fname')
    d = pd.read_csv(data_fname)

    d['ra_rad'] = np.radians(d.telra)
    d['decl_rad'] = np.radians(d.teldec)
    
    lst = gmst(d.mjd) + np.radians(longitude)
    moon_ra, moon_decl, diam = rdplan(d.mjd, 3, np.radians(longitude), np.radians(latitude))
    moon_ha = lst - moon_ra
    moon_zd = calc_zd(np.radians(latitude), moon_ha, moon_decl)
    d['moon_zd'] = np.degrees(moon_zd)
    d['moon_angle'] = np.degrees(ang_sep(d.ra_rad, d.decl_rad, moon_ra, moon_decl))
    d['moon_brightness'] = calc_moon_brightness(d.mjd)
    
    sun_ra, sun_decl, diam = rdplan(d.mjd, 0, np.radians(longitude), np.radians(latitude))
    sun_ha = lst - sun_ra
    sun_zd = calc_zd(np.radians(latitude), sun_ha, sun_decl)
    d['sun_zd'] = np.degrees(sun_zd)
    d['sun_angle'] = np.degrees(ang_sep(d.ra_rad, d.decl_rad, sun_ra, sun_decl))

    filters_spec = cfg.get('todo', 'filters')
    if len(filters_spec) == 1:
        filter_names = [filters_spec]
    else:
        filter_names = filters_spec.split()

    mie_dominant = cfg.get('fit','mie_dominant').split()
    rayl_dominant = cfg.get('fit','rayl_dominant').split()
    mie_angle = cfg.getfloat('fit','mie_angle')
    rayl_angle = cfg.getfloat('fit','rayl_angle')

    fit_params = {'filters' : [],
                  'k': [],
                  'm_inf': [],
                  'm_zen': [],
                  'h': [],
                  'rayl_m': [],
                  'g': [],
                  'mie_m': [],
                  'sun_dm': [],
                  'twi1': [],
                  'twi2': []}
    
    param_names = ['k','m_inf','m_zen','h','rayl_m','g','mie_m', 'sun_dm', 'twi1', 'twi2']

    for filter_name in filter_names:
        print '*******',filter_name,'*******'
        sys.stdout.flush()
        params = init_params(filter_name, cfg)
        dark_fit = fit_dark_sky_filter(d, params, filter_name)
        print dark_fit
        sys.stdout.flush()
        
        if filter_name in mie_dominant:
            mie_fit = fit_mie_sky_filter(d, dark_fit, filter_name, mie_angle)
            night_fit = fit_rayl_sky_filter(d, mie_fit, 
                                             filter_name, rayl_angle)
        elif filter_name in rayl_dominant:
            rayl_fit = fit_rayl_sky_filter(d, dark_fit, filter_name, rayl_angle)
            night_fit = fit_mie_sky_filter(d, rayl_fit, 
                                             filter_name, mie_angle)
        else:
            rayl_fit = fit_rayl_sky_filter(d, dark_fit, filter_name, rayl_angle)
            night_fit = fit_bright_sky_filter(d, rayl_fit, filter_name)

        bright_fit = fit_twilight(d, night_fit, filter_name)
        
        print bright_fit 
        print
        print 'k: %f' % bright_fit['k'].value
        print 'm_inf: %f' % bright_fit['m_inf'].value
        print 'm_zen: %f' % bright_fit['m_zen'].value
        print 'h: %f' % bright_fit['h'].value
        print 'rayl_m: %f' % bright_fit['rayl_m'].value
        print 'g: %f' % bright_fit['g'].value
        print 'mie_m: %f' % bright_fit['mie_m'].value
        print 'sun_dm: %f' % bright_fit['sun_dm'].value
        print 'twi1: %f' % bright_fit['twi1'].value
        print 'twi2: %f' % bright_fit['twi2'].value
        print

        fit_params['filters'] += [filter_name]
        for p in param_names:
            fit_params[p] += [bright_fit[p].value]

        sys.stdout.flush()

    print '[sky]'
    print 'filters = ' + ' '.join("%s" % v for v in tuple(fit_params['filters']))
    print 'k       = ' + ' '.join("%9.6f" % v for v in tuple(fit_params['k']))
    print 'm_inf   = ' + ' '.join("%9.6f" % v for v in tuple(fit_params['m_inf']))
    print 'm_zen   = ' + ' '.join("%9.6f" % v for v in tuple(fit_params['m_zen']))
    print 'h       = ' + ' '.join("%9.6f" % v for v in tuple(fit_params['h']))
    print 'rayl_m  = ' + ' '.join("%9.6f" % v for v in tuple(fit_params['rayl_m']))
    print 'g       = ' + ' '.join("%9.6f" % v for v in tuple(fit_params['g']))
    print 'mie_m   = ' + ' '.join("%9.6f" % v for v in tuple(fit_params['mie_m']))
    print 'sun_dm  = ' + ' '.join("%9.6f" % v for v in tuple(fit_params['sun_dm']))
    print 'twi1    = ' + ' '.join("%9.6f" % v for v in tuple(fit_params['twi1']))
    print 'twi2    = ' + ' '.join("%9.6f" % v for v in tuple(fit_params['twi2'])) 
