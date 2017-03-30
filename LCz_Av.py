import sys
import numpy as np
import matplotlib.pyplot as plt
# Cosmology stuff
sys.path.append("../cos_calc")
import cos_calc
from LCz import *

# class that combines LCz with different AVs
class LCz_Av(object):

    def __init__(self, **kwargs):
        
        self.LCz = kwargs["LCz"]
        self.Av = kwargs["Av"]
        self.Rv = kwargs["Rv"]
        self.zs = kwargs["zs"]
        self.DL = kwargs["DL"]
        self.Dm = kwargs["Dm"]
        self.filtername = kwargs["filtername"]

        self.nAv = len(self.Av)
        self.nz = len(self.zs)
        self.ntimes = self.LCz.ntimes
        
    def compute_mags(self, **kwargs):

        plotmodel = False
        if "plotmodel" in kwargs.keys():
            plotmodel = kwargs["plotmodel"]
            
        self.magAvf = []
        test = []

        for i in range(self.nAv):

            #print "Av: %f" % self.Av[i]
            #self.LCz.luminosity()
            self.LCz.attenuate(Av = self.Av[i], Rv = self.Rv)
            self.LCz.redshift(zs = self.zs, DL = self.DL)
            self.LCz.dofilter(filtername = self.filtername)
            if i == 0 and plotmodel:
                self.LCz.doplot = True
            else:
                self.LCz.doplot = False
            self.LCz.mags(Dm = self.Dm)

            for j in range(self.nz):

                # add interpolating function
                self.magAvf.append(interp1d(self.LCz.timesz[j], self.LCz.bandmag[j], bounds_error = False, fill_value = 40))

        # reshape list of interpolating functions
        self.magAvf = zip(*[iter(self.magAvf)]*self.nz)

    # define characteristic Av distribution scale
    def set_Avdistribution(self, **kwargs):

        self.lAv = 0.187
        if "lAv" in kwargs.keys():
            self.lAv = kwargs["lAv"]
        
        if self.nAv != 1:
            probs = np.exp(-self.Av / self.lAv)
            cumsum = np.cumsum(probs)
            cumsum = cumsum / np.sum(np.exp(-np.hstack([self.Av, np.arange(max(self.Av) + self.Av[1] - self.Av[0], 10. * max(self.Av), self.Av[1] - self.Av[0])]) / self.lAv))
            print "lAvs:", self.lAv, ", Avs:", self.Av, ", cumulative sum for Avs:", cumsum
            self.random2iAv = interp1d(cumsum, np.array(range(len(self.Av)), dtype = int) + 1, kind = 'zero', bounds_error = False, fill_value = (0, self.nAv - 1))
        
    # generate random light curve at a given redshift (given by zs[iz]) and times in MJD
    def simulate_randomLC(self, **kwargs):
        
        nsim = kwargs["nsim"]
        iz = kwargs["iz"]
        MJDs = kwargs["MJDs"]
        maxrestframeage = kwargs["maxrestframeage"] # days

        #print np.shape(self.LCz.timesz), np.argmin(np.abs(maxrestframeage - self.LCz.times))
        maxobserverage = self.LCz.timesz[iz][np.argmin(np.abs(maxrestframeage - self.LCz.times))]
        tmin = np.min(MJDs) - maxobserverage
        tmax = np.max(MJDs)

        texp = tmin + np.random.random(int(nsim)) * (tmax - tmin)
        if self.nAv != 1:
            iAv = map(lambda i: int(self.random2iAv(i)), np.random.random(int(nsim)))
        else:
            iAv = np.zeros(nsim, dtype = int)

        return (tmax - tmin, texp, iAv, np.array(map(lambda iAv, texp: self.magAvf[iAv][iz](MJDs - texp), iAv, texp)))
        
if __name__  == "__main__":

    # physical constants
    
    h = 6.6260755e-27       # erg s
    G = 6.67259e-8          # cm3 g-1 s-2
    k = 1.380658e-16        # erg K-1
    sigmaSB = 5.6704e-5     # erg cm-2 s-1 K-4
    eV = 1.6021772e-12      # erg
    Rsun = 6.96e10          # cm
    Msun = 1.99e33          # g
    yr2sec = 31556926       # seconds
    yr2days = 365.242199    # days
    pc2cm = 3.08567758e18   # cm
    cm2AA = 1e8
    cspeed = 2.99792458e10  # cm s-1
    cspeedAAs = cspeed * cm2AA  # A s-1
    Hnot = 71               # km s-1 Mpc-1
    Hnotcgs = Hnot * 1e5 / (1e6 * pc2cm)
    OmegaM = 0.27
    OmegaL = 0.73

    # redshifts
    nz = 2
    zmax = 0.44
    zs = np.linspace(zmax / nz, zmax, nz)

    # cosmology
    DL = np.zeros(nz)
    Dc = np.zeros(nz)
    Dm = np.zeros(nz)
    dVdzdOmega = np.zeros(nz)
    h100, omega_m, omega_k, omega_lambda = Hnot / 100., OmegaM, 1. - (OmegaM + OmegaL), OmegaL
    for i in range(nz):
        cosmo =  cos_calc.fn_cos_calc(h100, omega_m, omega_k, omega_lambda, zs[i])
        Dc[i] = cosmo[1] # Mpc
        DL[i] = cosmo[4] # Mpc
        Dm[i] = cosmo[5] # Mpc
        dVdzdOmega[i] = cspeed / (Hnot * 1e5 * np.sqrt((1. + zs[i])**3. * OmegaM + OmegaL)) * Dc[i]**2

    # light curves given model and cosmology
    SN = StellaModel(dir = "models", modelfile = "15z002E1.dat", doplot = False)

    # attenuations
    lAv = 0.187
    Avs = np.linspace(0, 4. * lAv, 10)# np.hstack([0, np.logspace(-1.5, 0, 10)])

    # start new object with model light curves at different redshifts and Avs
    SN_Av = LCz_Av(LCz = SN, Av = Avs, Rv = 4.5, zs = zs, DL = DL, Dm = Dm, filtername = 'g')
    SN_Av.compute_mags()

    # plot magnitudes after Av
    fig, ax = plt.subplots(figsize = (14, 8))
    jet = cm = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin = 0, vmax = SN_Av.nAv)
    scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = jet)
            
    for i in range(SN_Av.nAv):

        colorVal = scalarMap.to_rgba(i)
        
        for j in range(SN_Av.nz):

            ax.plot(SN_Av.LCz.timesz[j] - SN_Av.LCz.tstartpeak[j], SN_Av.magAvf[i][j](SN_Av.LCz.timesz[j]), c = colorVal, lw = 3. * (SN_Av.nz - j) / SN_Av.nz, alpha = 0.5)

    ax.set_ylim(26, 20)
    ax.set_xlim(-2, 15)
    ax.set_xlabel("Time since SBO [days]")
    ax.set_xlabel("mag g")
    plt.grid()
    plt.savefig("plots/LCz_Av_test.png")
    
    # generate Av probability distribution
    SN_Av.set_Avdistribution(lAv = lAv)

    fig, ax = plt.subplots(ncols = 2)
    values = np.random.random(10000)
    ax[0].hist(map(lambda i: Avs[int(i)], SN_Av.random2iAv(values)), normed = True, bins = 100, cumulative = True, lw = 0)
    ax[1].hist(map(lambda i: Avs[int(i)], SN_Av.random2iAv(values)), normed = True, bins = 100, lw = 0)
    xs = np.linspace(0, 10., 1000)
    pdf = np.exp(-xs / lAv)
    cdf = np.cumsum(pdf)
    ax[0].plot(xs, cdf / max(cdf))
    ax[1].plot(xs, 45. * pdf)
    ax[0].set_xlim(0, 1)
    ax[1].set_xlim(0, 1)
    plt.savefig("plots/Avs.png")

    # generate random light curves
    simtimes = np.linspace(10, 20, 1000)
    nsim = 100
    simdt, simtexp, simiAv, simmags = SN_Av.simulate_randomLC(nsim = nsim, iz = 0, MJDs = simtimes, maxrestframeage = 10)
    
    fig, ax = plt.subplots()
    jet = cm = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin = 0, vmax = SN_Av.nAv)
    scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = jet)

    for i in range(nsim):
        colorVal = scalarMap.to_rgba(simiAv[i])
        ax.plot(simtimes, simmags[i], c = colorVal)
        ax.axvline(simtexp[i], c = 'gray')

    ax.set_xlim(min(simtimes), max(simtimes))
    ax.set_ylim(28, 23)
    plt.savefig("plots/sim.png")
    

    
