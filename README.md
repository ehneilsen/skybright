# skybright

Sky light, sky bright,</br>
Only sky I'll see tonight,</br>
I wish I may, I wish I might,</br>
Know the sky I'll see tonight.</br>


`skybright` is a tool for esitamting the sky brightness (in
magnitudes/asec^2) of a given set of celestial coordinates, from a
given site, at a given time, in a given filter.

It use a simple model with four components:

1. An airglow shell in the Earth's atmosphere
2. Rayleigh scattering of moonlight
3. Mie scattering of moonlight
4. A very simple algebraic estimation of twilight.

From a shell, it can be executed thus:

```
$ skybright --help
usage: Estimate the sky brightness [-h] [-m MJD] [-r RA] [-d DEC] [-f FILTER]
                                   [-c CONFIG]

optional arguments:
  -h, --help            show this help message and exit
  -m MJD, --mjd MJD     Modified Julian Date (float) (UTC)
  -r RA, --ra RA        the RA (decimal degrees)
  -d DEC, --dec DEC     the declination (decimal degrees)
  -f FILTER, --filter FILTER
                        the filter
  -c CONFIG, --config CONFIG
                        the configuration file
$ skybright -m 58008.4 -r 40.0 -d -5.0 -f i -c etc/skybright.conf 
Moon zenith distance: 47.645061
Sun zenith distance: 105.734680
Elongation of the moon: 101.444917
Moon brightness: 0.132464
Pointing angle with moon: 35.179739
Pointing zenith distance: 34.326929
Airmass: 1.210294
Sky brightness at pointing: 19.249708
```

The `jupyter` notebook in
[`doc/skybright.ipynb`](doc/skybright.ipynb) shows the use of
the `python` API.