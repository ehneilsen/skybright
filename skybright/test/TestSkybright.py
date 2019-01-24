import unittest
import numpy as np
from skybright import ang_sep
from skybright import calc_zd
from skybright import calc_airmass
from skybright import calc_airglow
from skybright import calc_scat_extinction
from skybright import elongation
from skybright import calc_moon_brightness
from skybright import calc_twilight_fract
from skybright import mag0

class TestSkybright(unittest.TestCase):

    def test_ang_sep(self):
        r1, r2 = (2.7110343929140726, 1.2643281542455602)
        d1, d2 = (0.22769050527178125, -1.0034614875329488)
        s = ang_sep(r1, d1, r2, d2)
        self.assertAlmostEqual(s, 1.696696840884825)

    def test_calc_zd(self):
        zd1 = calc_zd(np.radians(-41.1), 0.0, np.radians(-51.1))
        self.assertAlmostEqual(zd1, np.radians(10))
        

    def test_calc_airmass(self):
        self.assertAlmostEqual(calc_airmass(np.cos(0)), 1.0)
        self.assertAlmostEqual(calc_airmass(np.cos(np.radians(20.0))), 1/np.cos(np.radians(20.0)), 3)
        # Snell's airmass values
        zd = np.asarray([60, 65, 70, 75, 80])
        snell_x = np.asarray([1.993, 2.353, 2.898, 3.802, 5.555])
        x = calc_airmass(np.cos(np.radians(zd)))
        max_diff = np.abs((x-snell_x)/snell_x).max()
        self.assertLess(max_diff, 0.01)

    def test_calc_airglow(self):
        airglow = calc_airglow(6375.0, 90.0, 20.26, 0.08, 0.6014162262210697, 1.2509062370431252)
        self.assertAlmostEqual(mag0 - 2.5*np.log10(airglow), 20.040617969240394)

    def test_calc_scat_extinction(self):
        scat_ext = calc_scat_extinction(0.08, 1.4823902499099404, 1.2509062370431252)
        self.assertAlmostEqual(scat_ext, 2.082035878907766)

    def test_elongation(self):
        elong = elongation(51778.47)
        self.assertAlmostEqual(elong, 94.0, 1)

    def test_calc_moon_brightness(self):
        mb = calc_moon_brightness(51778.47)
        self.assertAlmostEqual(mb, 0.10, 2)

    def test_calc_twilight_fract(self):
        tf = calc_twilight_fract(88.0)
        self.assertEqual(tf, 1.0)
        
        tf = calc_twilight_fract(110.0)
        self.assertEqual(tf, 0.0)

        ft = calc_twilight_fract(105.73468038142708, -0.78734, 0.002107)
        self.assertAlmostEqual(ft, 1.2607908292895948e-06)
        
if __name__ == '__main__':
    unittest.main()


