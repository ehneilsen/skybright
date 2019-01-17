import csv
import sys
from argparse import ArgumentParser
from ConfigParser import ConfigParser
from collections import namedtuple
import readline
import code
import numpy as np
from math import acos, degrees
from lmfit import minimize, Parameters, Parameter, report_errors
from skybright import body_zd, skymag, moon_brightness, cosrho

latitude = -30.16527778
longitude = -70.815

def residuals(params, data):
    def mmag(d):
        return skymag(params['m_inf'].value, 
                      params['m_zen'].value, 
                      params['h'].value, 
                      params['g'].value, 
                      params['mie_c'].value, 
                      params['rayl_m'].value, 
                      d.telra, d.teldec, d.mjd, 
                      d.k, latitude, longitude, 0.0,
                      params['sun_m'].value,
                      params['twi1'].value,
                      params['twi2'].value)
    modmag = np.array([mmag(d) for d in data])
    datamag = np.array([d.skymag for d in data])

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
               min=0.5, max=1.5)
    params.add('mie_c', 
               value = float(cfg.get("sky","mie_c").split()[i]),
               min=2.0, max=2000.0)
    params.add('rayl_m', value = float(cfg.get("sky","rayl_m").split()[i]),
               min=-10.0, max=100.0)
    params.add('sun_m', value = float(cfg.get("sky","sun_m").split()[i]),
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
               min=0.5, max=1.5, vary=False)
    params.add('mie_c', 
               value =  in_params['mie_c'].value,
               min=2.0, max=2000.0, vary=False)
    params.add('rayl_m',  in_params['rayl_m'].value,
               min=-10.0, max=100.0, vary=False)
    params.add('sun_m',  in_params['sun_m'].value,
               min=-20.0, max=-10.0, vary=False)
    params.add('twi1', value = in_params['twi1'],
               min=-100.0, max=100.0, vary=False)
    params.add('twi2', value = in_params['twi2'],
               min=-100.0, max=100.0, vary=False)
    
    data = [d for d in all_data 
            if d.band==filter_name 
               and d.airmass < 2.0
               and body_zd('moon', latitude, longitude, d.mjd) > 108.0
               and body_zd('sun', latitude, longitude, d.mjd) > 108.0]

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
               min=0.5, max=1.5, vary=True)
    params.add('mie_c', 
               value =  in_params['mie_c'].value,
               min=2.0, max=20000.0, vary=True)
    params.add('rayl_m',  in_params['rayl_m'].value,
               min=-10.0, max=100.0, vary=True)
    params.add('sun_m',  in_params['sun_m'].value,
               min=-20.0, max=-10.0, vary=False)
    params.add('twi1', value = in_params['twi1'],
               min=-100.0, max=100.0, vary=False)
    params.add('twi2', value = in_params['twi2'],
               min=-100.0, max=100.0, vary=False)
    
    data = [d for d in all_data 
            if d.band==filter_name 
               and d.airmass < 2.0
               and moon_brightness > 0.5
               and body_zd('moon', latitude, longitude, d.mjd) < 80.0
               and body_zd('sun', latitude, longitude, d.mjd) > 108.0]

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
               min=0.5, max=1.5, vary=False)
    params.add('mie_c', 
               value =  in_params['mie_c'].value,
               min=2.0, max=20000.0, vary=False)
    params.add('rayl_m',  in_params['rayl_m'].value,
               min=-10.0, max=100.0, vary=True)
    params.add('sun_m',  in_params['sun_m'].value,
               min=-20.0, max=-10.0, vary=False)
    params.add('twi1', value = in_params['twi1'],
               min=-100.0, max=100.0, vary=False)
    params.add('twi2', value = in_params['twi2'],
               min=-100.0, max=100.0, vary=False)

    data = [d for d in all_data 
            if d.band==filter_name 
               and d.airmass < 2.0
               and moon_brightness > 0.5
               and degrees(acos(cosrho(d.mjd, d.telra, d.teldec, 
                               latitude, longitude,
                               'moon'))) > rayl_angle
               and body_zd('moon', latitude, longitude, d.mjd) < 80.0
               and body_zd('sun', latitude, longitude, d.mjd) > 108.0]

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
               min=0.5, max=1.5, vary=True)
    params.add('mie_c', 
               value =  in_params['mie_c'].value,
               min=2.0, max=20000.0, vary=True)
    params.add('rayl_m',  in_params['rayl_m'].value,
               min=-10.0, max=100.0, vary=False)
    params.add('sun_m',  in_params['sun_m'].value,
               min=-20.0, max=-10.0, vary=False)
    params.add('twi1', value = in_params['twi1'],
               min=-100.0, max=100.0, vary=False)
    params.add('twi2', value = in_params['twi2'],
               min=-100.0, max=100.0, vary=False)
    
    data = [d for d in all_data 
            if d.band==filter_name 
               and d.airmass < 2.0
               and moon_brightness > 0.5
               and degrees(acos(cosrho(d.mjd, d.telra, d.teldec, 
                               latitude, longitude,
                               'moon'))) < mie_angle
               and body_zd('moon', latitude, longitude, d.mjd) < 80.0
               and body_zd('sun', latitude, longitude, d.mjd) > 108.0]

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
               min=0.5, max=1.5, vary=False)
    params.add('mie_c', 
               value =  in_params['mie_c'].value,
               min=2.0, max=20000.0, vary=False)
    params.add('rayl_m',  in_params['rayl_m'].value,
               min=-10.0, max=100.0, vary=False)
    params.add('sun_m',  in_params['sun_m'].value,
               min=-20.0, max=-10.0, vary=True)
    params.add('twi1', value = in_params['twi1'],
               min=-100.0, max=100.0, vary=True)
    params.add('twi2', value = in_params['twi2'],
               min=-100.0, max=100.0, vary=True)
    
    data = [d for d in all_data 
            if d.band==filter_name 
               and d.airmass < 2.0
               and body_zd('moon', latitude, longitude, d.mjd) > 108.0
               and body_zd('sun', latitude, longitude, d.mjd) < 106.0]

    fit = minimize(residuals, params, args=(data,))

    report_errors(fit.params)
    return fit.params

def read_sky_data(filename):
    def tofloat(x):
        try:
            fx = float(x)
            return fx
        except:
            return x

    with open(filename, 'r') as f:
        dialect = csv.Sniffer().sniff(f.read(4096))
        f.seek(0)
        rdr = csv.reader(f)
        rows = list(rdr)
        Exposure = namedtuple('Exposure',rows[0])
        data = [Exposure(*[tofloat(v) for v in r]) for r in rows[1:]]
    return data

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
    d = read_sky_data(data_fname)

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
                  'mie_c': [],
                  'sun_m': [],
                  'twi1': [],
                  'twi2': []}
    
    param_names = ['k','m_inf','m_zen','h','rayl_m','g','mie_c', 'sun_m', 'twi1', 'twi2']

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
        print 'mie_c: %f' % bright_fit['mie_c'].value
        print 'sun_m: %f' % bright_fit['sun_m'].value
        print 'twi1: %f' % bright_fit['twi1'].value
        print 'twi2: %f' % bright_fit['twi2'].value
        print

        fit_params['filters'] += [filter_name]
        for p in param_names:
            fit_params[p] += [bright_fit[p].value]

        sys.stdout.flush()

    print '[sky]'
    print 'filters = %s     %s     %s     %s     %s  %s' % tuple(fit_params['filters'])
    print 'k       = %4.2f %4.2f %4.2f %4.2f %4.2f  %4.2f' % tuple(fit_params['k'])
    print 'm_inf   = %4.2f %4.2f %4.2f %4.2f %4.2f  %4.2f' % tuple(fit_params['m_inf'])
    print 'm_zen   = %4.2f %4.2f %4.2f %4.2f %4.2f  %4.2f' % tuple(fit_params['m_zen'])
    print 'h       = %4.0f %4.0f %4.0f %4.0f %4.0f  %4.2f' % tuple(fit_params['h'])
    print 'rayl_m  = %4.2f %4.2f %4.2f %4.2f %4.2f  %4.2f' % tuple(fit_params['rayl_m'])
    print 'g       = %4.2f %4.2f %4.2f %4.2f %f %f' % tuple(fit_params['g'])
    print 'mie_c   = %4.2f %4.2f %4.2f %4.2f %f %f' % tuple(fit_params['mie_c'])
    print 'sun_m   = %4.2f %4.2f %4.2f %4.2f %f %f' % tuple(fit_params['sun_m'])
    print 'twi1    = %9.6f %9.6f %9.6f %9.6f %f %f' % tuple(fit_params['twi1'])
    print 'twi2    = %9.6f %9.6f %9.6f %9.6f %f %f' % tuple(fit_params['twi2'])

