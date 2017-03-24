import numpy as np
import matplotlib.pyplot as plt

import constants
from obsplan import *
from LCz import *
from LCz_Av import *
from SFHs import *

# Cosmology stuff
sys.path.append("../cos_calc")
import cos_calc

# class that defines a survey
class survey(object):

    # initialize: needs observation plan, model, Avs, Rv, filtername, number of redshift bins
    def __init__(self, **kwargs):

        self.obsplan = kwargs["obsplan"]
        self.LCz = kwargs["LCz"]
        self.SFH = kwargs["SFH"]
        self.efficiency = kwargs["efficiency"]
        self.Avs = kwargs["Avs"]
        self.Rv = kwargs["Rv"]
        self.lAv = kwargs["lAv"]
        self.filtername = kwargs["filtername"]
        self.nz = kwargs["nz"]

        # maximum time for an object to have exploded before the beginning of the simulation to be considered detected
        self.maxrestframeage = 10

    def compute_zs(self):

        self.zs = np.linspace(0, self.maxz, self.nz + 1)[1:]
        self.zbin = self.zs[1] - self.zs[0]
        self.zedges = np.linspace(self.maxz * 0.5 / self.nz, self.maxz * (1. + 0.5 / self.nz), self.nz + 1)

        
    def do_cosmology(self):

        # cosmology
        self.DL = np.zeros_like(self.zs)
        self.Dc = np.zeros_like(self.zs)
        self.Dm = np.zeros_like(self.zs)
        self.dVdzdOmega = np.zeros_like(self.zs)
        h100, omega_m, omega_k, omega_lambda = Hnot / 100., OmegaM, 1. - (OmegaM + OmegaL), OmegaL
        cosmo =  np.array(map(lambda z: cos_calc.fn_cos_calc(h100, omega_m, omega_k, omega_lambda, z), self.zs))
        self.Dc = cosmo[:, 1] # Mpc
        self.DL = cosmo[:, 4] # Mpc
        self.Dm = cosmo[:, 5] # Mpc
        self.dVdzdOmega = cspeed / (Hnot * 1e5 * np.sqrt((1. + self.zs)**3. * OmegaM + OmegaL)) * self.Dc**2

        # compute SFR
        self.SFH.doSFR(self.zs)
            
    def estimate_maxredshift(self, **kwargs):

        print "   Looking for maximum redshift..."

        zguess = kwargs["zguess"]
        self.minprobdetection = kwargs["minprobdetection"]
        self.minndetections = kwargs["minndetections"]

        probdet = [0, 0, 0]
        delta = 0.01
        while True:

            # create redshift array
            self.zs = np.array([(1. - delta) *  zguess, zguess, (1. + delta) * zguess])

            # do cosmology
            self.do_cosmology()
            
            # create LCz_Av object (use minimum Av)
            self.LCz_Av = LCz_Av(LCz = self.LCz, Av = np.array([min(self.Avs)]), Rv = 4.5, lAv = self.lAv, zs = self.zs, DL = self.DL, Dm = self.Dm, filtername = self.filtername)
            self.LCz_Av.compute_mags()
            
            # generate 1/prob SNe and check whether none of them are detected, otherwise increase redshift guess. If no detections, decrease redshift guess
            nsim = int(1. / self.minprobdetection)
            simmags = np.array(map(lambda iz: self.LCz_Av.simulate_randomLC(nsim = nsim, iz = iz, MJDs = self.obsplan.MJDs, maxrestframeage = self.maxrestframeage)[3], range(len(self.zs))))
            
            # compute detection probabilities
            probdet = map(lambda iz: 1. * np.sum(np.sum(simmags[iz, :, :] <= self.obsplan.limmag, axis = 1) >= self.minndetections) / nsim, range(len(self.zs)))

            # correct delta
            if probdet[2] <= self.minprobdetection:
                if probdet[1] <= self.minprobdetection:
                    if probdet[0] <= self.minprobdetection: # below 1st
                        zguess = min(self.zs)
                        delta = 0.5
                    else: # between 1st and 2nd
                        zguess = np.average(self.zs[0:2])
                        delta = delta / 2.
                else: # between 2nd and 3rd
                    delta = delta / 2.
                    if delta <= 0.01:
                        self.maxz = max(self.zs)

                        # compute maximum length of simulation by doing one simulation
                        self.ndayssim = self.LCz_Av.simulate_randomLC(nsim = 1, iz = 2, MJDs = self.obsplan.MJDs, maxrestframeage = self.maxrestframeage)[0]

                        # EXIT here only
                        return 
                    
            else: # above 3rd
                zguess = max(self.zs)
                delta = min(2. * delta, 0.5)

            print "      Correction applied. Current range is", self.zs

            
    # compute magnitudes and set Av distribution
    def compute_mags(self):
        
        print "      Computing magnitudes..."

        # create LCz_Av object (use minimum Av)
        self.LCz_Av = LCz_Av(LCz = self.LCz, Av = self.Avs, Rv = self.Rv, zs = self.zs, DL = self.DL, Dm = self.Dm, filtername = self.filtername)
        self.LCz_Av.compute_mags()

        # generate Av probability distribution
        self.LCz_Av.set_Avdistribution(self.lAv)

    # sample light curves from the survey    
    def sample_events(self, **kwargs):

        print "      Sampling light curves..."

        # save number of simulated LCs
        self.nsim = kwargs["nsim"]
        
        # think how to sample more efficiently from this distribution...
        self.totalSNe = self.efficiency * self.SFH.SFR / (1. + self.zs) * (self.ndayssim / yr2days) * self.dVdzdOmega * (self.obsplan.obs.FoV / ster_sqdeg) * self.obsplan.nfields  # true number of SNe

        # cumulative distribution over which to sample
        self.cumtotalSNe = np.cumsum(self.totalSNe * self.zbin)

        # generate random numbers for redshift of SNe
        rands = np.random.random(size = self.nsim) * self.cumtotalSNe[-1]
        izs = np.array(map(lambda r: np.sum(self.cumtotalSNe < r), rands), dtype = int)
        self.simzs = self.zs[izs]

        # reset 
        if hasattr(self, "simLCs"):
            del self.simLCs, self.simdts, self.simtexps, self.simiAvs, self.nsimz, self.ndetections

        # initialize arrays to store simulated light curves
        self.nsimz = np.zeros(self.nz) # number of simulated LCs at different redshifts
        self.simdts = np.zeros(self.nz)
        self.ndetections = np.zeros(self.nz)
        self.detprob = np.zeros(self.nz)
        self.simtexps = np.zeros(self.nsim)
        self.simiAvs = np.zeros(self.nsim, dtype = int)
        self.simLCs = np.ones((self.nsim, len(self.obsplan.MJDs))) * 40.
        
        # simulate light curves and store them
        l1 = 0
        for iz in range(self.nz):

            self.nsimz[iz] = int(np.sum(izs == iz))

            if self.nsimz[iz] > 0:

                # simulate and count detections
                simdt, simtexp, simiAv, simmags = self.LCz_Av.simulate_randomLC(nsim = self.nsimz[iz], iz = iz, MJDs = self.obsplan.MJDs, maxrestframeage = self.maxrestframeage)
                ndetections = np.sum(map(lambda mags: np.sum(mags <= self.obsplan.limmag) >= self.minndetections, simmags))

                # fill arrays with light curve information and statistics
                self.simdts[iz] = simdt
                self.ndetections[iz] = ndetections
                self.detprob[iz] = 1. * ndetections / self.nsimz[iz]
                l2 = int(l1 + self.nsimz[iz])
                self.simtexps[l1: l2] = simtexp
                self.simiAvs[l1: l2] = simiAv
                self.simLCs[l1: l2, :] = simmags
                l1 = int(l1 + self.nsimz[iz])

    # plot the simulated light curves and some statistics
    def plot_LCs(self, **kwargs):

        save = True
        if "save" in kwargs.keys():
            save = kwargs["save"]

        print "      Plotting light curves and statistics..."
        
        fig, ax = plt.subplots() 
        ax.plot(self.zs, self.cumtotalSNe, label = r"$\int_0^z \eta ~\frac{SFR(z)}{1+z} ~T_{\rm sim} ~\frac{dV(z)}{dz d\Omega} ~\Delta\Omega ~n_{\rm fields} ~dz $", c = 'k', zorder = 100)
        scale = self.cumtotalSNe[-1] / self.nsim
        hist, edges = np.histogram(self.simzs, bins = self.zedges)
        ax.plot((edges[:-1] + edges[1:]) / 2., np.cumsum(hist) * scale, label = "Explosions", c = 'r', lw = 5, alpha = 0.5, drawstyle = 'steps-post')
        ax.plot(self.zs, np.cumsum(self.ndetections) * scale, label = "Detections", c = 'b', lw = 5, drawstyle = 'steps-post')
        ax.legend(loc = 2, fontsize = 14, framealpha = 0.7)
        ax.set_title("Total detections: %.1f" % (1. * np.sum(self.ndetections) / self.nsim * self.cumtotalSNe[-1]))
        ax.set_xlabel("z")
        ax.set_ylabel("Event cumulative distribution")
        if save:
            plt.savefig("plots/simLCs_expvsdet_%s.png" % (self.obsplan.planname))

        fig, ax = plt.subplots()
        map(lambda LC: ax.plot(self.obsplan.MJDs, LC, alpha = 0.1), self.simLCs[np.random.choice(range(self.nsim), size = 5000, replace = True)])
        ax.plot(self.obsplan.MJDs, self.obsplan.limmag, lw = 5, c = 'r', label = "lim. mag.")
        ax.set_ylim(max(self.obsplan.limmag) + 0.5, min(self.simLCs.flatten()) - 0.5)
        ax.set_xlabel("MJD [days]")
        ax.set_ylabel("%s mag" % self.filtername)
        ax.legend()

        ## function used for plot ticks
        def YYYYMMDD(date, pos):
            return Time(date, format = 'mjd', scale = 'utc').iso[:11]
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(YYYYMMDD))
        plt.xticks(rotation = 45, ha = 'left')

        if save:
            plt.savefig("plots/simLCs_LCs_%s.png" % (self.obsplan.planname))
            np.save("npy/simLCs_LCs_%s.npy" % (self.obsplan.planname), self.simLCs)
            np.save("npy/simLCs_MJDs_%s.npy" % (self.obsplan.planname), self.obsplan.MJDs)
            np.save("npy/simLCs_limmag_%s.npy" % (self.obsplan.planname), self.obsplan.limmag)

        fig, ax = plt.subplots()
        ax.hist(self.simtexps)
        ax.set_xlabel("Explosion time MJD [days]")
        ax.set_ylabel("N")
        ax.legend()
        if save:
            plt.savefig("plots/simLCs_texp_%s.png" % (self.obsplan.planname))
            np.save("npy/simLCs_texp_%s.npy" % (self.obsplan.planname), self.simtexps)

        fig, ax = plt.subplots()
        ax.plot(self.zs, self.detprob, drawstyle = "steps", lw = 5)
        ax.set_xlabel("z")
        ax.set_ylabel("Detection probability")
        ax.legend()
        if save:
            plt.savefig("plots/simLCs_prob_%s.png" % (self.obsplan.planname))
            np.save("npy/simLCs_prob_%s.npy" % self.obsplan.planname, self.detprob)

        if not save:
            plt.show()
        

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


    # filtername
    filtername = 'g'
    modelfile = "13z002E1.0.dat"
    obsname = sys.argv[1] #"Blanco-DECam" #"KMTNet"

    # start an observational plan
    customplan = obsplan(obsname = obsname, band = filtername, mode = 'custom', nfields = 1, nepochspernight = 1, ncontnights = 150, nnights = 150, nightfraction = 1. / 100., nread = 1, startmoonphase = -3, maxmoonphase = 15, doplot = False)
    
    # light curve model
    SN = StellaModel(dir = "/home/fforster/Work/Model_LCs/models", modelfile = modelfile, doplot = False)

    # extinction
    lAv = 0.187
    nAv = 10
    Avs = np.linspace(0, 4. * lAv, nAv)# np.hstack([0, np.logspace(-1.5, 0, 10)])
    Rv = 3.1

    # star formation history and efficienciy
    SFH = SFHs(SFH = "MD14")

    # number of redshift bins
    nz = 20 #20 #20

    # conversion efficiency between star formation and explosions
    knorm = 0.0091
    IIPfrac = 0.54
    efficiency = knorm * IIPfrac

    # start survey
    newsurvey = survey(obsplan = customplan, LCz = SN, Avs = Avs, Rv = Rv, lAv = lAv, SFH = SFH, efficiency = knorm * IIPfrac, filtername = 'g', nz = nz)

    # estimate maximum survey redshift
    newsurvey.estimate_maxredshift(zguess = 0.334, minprobdetection = 1e-4, minndetections = 2)

    # compute redshifts
    newsurvey.compute_zs()

    # update cosmology
    newsurvey.do_cosmology()

    # compute magnitudes
    newsurvey.compute_mags()
    
    # sample from distribution
    newsurvey.sample_events(nsim = 10000)
    
    # plot light curves
    newsurvey.plot_LCs(save = True)
    
