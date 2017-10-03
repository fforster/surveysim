import numpy as np
import ephem # needs pyephem 
from astropy.time import Time
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# class used to predict the sky magnitude
class sky(object):

    def __init__(self, **kwargs):
        
        self.band = kwargs["band"]
        self.MJDs = kwargs["MJDs"]

        # sky magnitudes
        if self.band == 'u' or self.band == 'B':
            magsky = np.array([22.8, 22.3, 20.7, 19.2, 17.7])
        elif self.band == 'g' or self.band == 'V':
            magsky = np.array([22.1, 21.8, 21.2, 20.4, 19.4])
        elif self.band == 'r' or self.band == 'R':
            magsky = np.array([21.1, 20.9, 20.6, 20.2, 19.7])
        elif self.band == 'i' or self.band == 'I':
            magsky = np.array([20.1, 20.1, 19.9, 19.7, 19.4])
        elif self.band == 'z':
            magsky = np.array([18.7, 18.7, 18.6, 18.5, 18.2])
        elif self.band == 'Y':
            magsky = np.array([18.0, 18.0, 18.0, 17.9, 17.7])

        # moon phase interpolation
        phases = np.array([0, 3, 7, 10, 14])
        self.magskyf = interp1d(phases, magsky, fill_value = "extrapolate")

    def moonphase(self, MJD):

        ephemdate = ephem.Date((Time(MJD, format = 'mjd', scale = 'utc').isot).replace("T", " "))
        ephemnextnewmoon = ephem.next_new_moon(ephemdate)
        ephempreviousnewmoon = ephem.previous_new_moon(ephemdate)
        moonphase = min(ephemnextnewmoon - ephemdate, ephemdate - ephempreviousnewmoon)
        
        return moonphase

    def skymags(self):
        
        moonphases = map(self.moonphase, self.MJDs)
        skymags = map(lambda x: self.magskyf(x), moonphases)
        return np.array(moonphases), np.array(skymags)

if __name__ == "__main__":

    MJDs = np.linspace(57023, 57023 + 365, 3650)

    fig, ax = plt.subplots()

    for band in ['u', 'g', 'r', 'i']:
        skymodel = sky(band = band, MJDs = MJDs)

        (moonphase, skymag) = skymodel.skymags()

        ax.scatter(moonphase, skymag, lw = 0, label = band)

    plt.legend()
    plt.savefig("plots/skymag.png")


