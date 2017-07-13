import numpy as np
import matplotlib.pyplot as plt

from constants import *
from obsplan import *
from LCz import *
from LCz_Av import *
from LCz_Av_params import *
from SFHs import *

# Cosmology stuff
sys.path.append("../cos_calc")
import cos_calc

# class that defines a survey
class survey_singlemodel(object):

    # initialize: needs observation plan, model, Avs, Rv, filtername, number of redshift bins
    def __init__(self, **kwargs):

        self.obsplan = kwargs["obsplan"]
        self.SFH = kwargs["SFH"]
        self.efficiency = kwargs["efficiency"]

        self.LCz = kwargs["LCz"]
        self.Avs = kwargs["Avs"]
        self.Rv = kwargs["Rv"]
        if "lAv" in kwargs.keys():
            self.lAv = kwargs["lAv"]

        self.filtername = kwargs["filtername"]
        self.nz = kwargs["nz"]

        # maximum time for an object to have exploded before the beginning of the simulation to be considered detected
        self.maxrestframeage = 10
        if "maxrestframeage" in kwargs.keys():
            self.maxrestframeage = kwargs["maxrestframeage"]
            

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

        self.LCz.doplot = False
        
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
    def compute_mags(self, **kwargs):

        plotmodel = False
        if "plotmodel" in kwargs.keys():
            plotmodel = kwargs["plotmodel"]
        
        print "      Computing magnitudes..."

        # create LCz_Av object (use minimum Av)
        self.LCz_Av = LCz_Av(LCz = self.LCz, Av = self.Avs, Rv = self.Rv, zs = self.zs, DL = self.DL, Dm = self.Dm, filtername = self.filtername)
        self.LCz_Av.compute_mags(plotmodel = plotmodel)

        # generate Av probability distribution
        self.LCz_Av.set_Avdistribution(lAv = self.lAv)

    # sample light curves from the survey    
    def sample_events(self, **kwargs):

        #print "      Sampling light curves..."

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
        self.ndetectionsdiff = np.zeros(self.nz)
        self.detprob = np.zeros(self.nz)
        self.detprobdiff = np.zeros(self.nz)
        self.simtexps = np.zeros(self.nsim)
        self.simiAvs = np.zeros(self.nsim, dtype = int)
        self.simLCs = np.ones((self.nsim, len(self.obsplan.MJDs))) * 40. # assume 40 mag pre-explosion
        
        # simulate light curves and store them
        l1 = 0
        for iz in range(self.nz):

            self.nsimz[iz] = int(np.sum(izs == iz))

            if self.nsimz[iz] > 0:

                # simulate and count detections
                simdt, simtexp, simiAv, simmags = self.LCz_Av.simulate_randomLC(nsim = self.nsimz[iz], iz = iz, MJDs = self.obsplan.MJDs, maxrestframeage = self.maxrestframeage)
                ndetections = np.sum(map(lambda mags: np.sum(mags <= self.obsplan.limmag) >= self.minndetections, simmags))
                ndetectionsdiff = np.sum(map(lambda mags: np.sum(mags <= self.obsplan.limmag - 2.5 * np.log10(np.sqrt(2.))) >= self.minndetections, simmags))

                # fill arrays with light curve information and statistics
                self.simdts[iz] = simdt
                self.ndetections[iz] = ndetections
                self.ndetectionsdiff[iz] = ndetectionsdiff
                self.detprob[iz] = 1. * ndetections / self.nsimz[iz]
                self.detprobdiff[iz] = 1. * ndetectionsdiff / self.nsimz[iz]
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

        if save:
            np.save("npy/simLCs_zs-CDF_%s_%s.npy" % (self.obsplan.planname, self.LCz.modelname), {"zs": self.zs, "cumtotalSNe": self.cumtotalSNe})
        
        fig, ax = plt.subplots() 
        ax.plot(self.zs, self.cumtotalSNe, label = r"$\int_0^z \eta ~\frac{SFR(z)}{1+z} ~T_{\rm sim} ~\frac{dV(z)}{dz d\Omega} ~\Delta\Omega ~n_{\rm fields} ~dz $", c = 'k', zorder = 100)
        scale = self.cumtotalSNe[-1] / self.nsim
        hist, edges = np.histogram(self.simzs, bins = self.zedges)
        ax.plot((edges[:-1] + edges[1:]) / 2., np.cumsum(hist) * scale, label = "Explosions", c = 'r', lw = 5, alpha = 0.5, drawstyle = 'steps-post')
        ax.plot(self.zs, np.cumsum(self.ndetections) * scale, label = "Detections", c = 'b', lw = 5, drawstyle = 'steps-post')
        ax.plot(self.zs, np.cumsum(self.ndetectionsdiff) * scale, label = "Detections (diff.)", c = 'g', lw = 5, drawstyle = 'steps-post')
        ax.legend(loc = 2, fontsize = 14, framealpha = 0.7)
        ax.set_title("Total detections: %.1f, %.1f (diff.)" % (1. * np.sum(self.ndetections) / self.nsim * self.cumtotalSNe[-1], 1. * np.sum(self.ndetectionsdiff) / self.nsim * self.cumtotalSNe[-1]))
        ax.set_xlabel("z")
        ax.set_ylabel("Event cumulative distribution")
        ax.set_xlim(0, self.maxz)
        if save:
            plt.savefig("plots/simLCs_expvsdet_%s_%s.png" % (self.obsplan.planname, self.LCz.modelname))

        fig, ax = plt.subplots(figsize = (13, 7))
        map(lambda LC: ax.plot(self.obsplan.MJDs, LC, alpha = 0.1), self.simLCs[np.random.choice(range(self.nsim), size = 5000, replace = True)])
        ax.plot(self.obsplan.MJDs, self.obsplan.limmag, c = 'r', label = "lim. mag.")
        ax.plot(self.obsplan.MJDs, self.obsplan.limmag - 2.5 * np.log10(np.sqrt(2.)), c = 'r', ls = ':', label = "lim. mag. (diff.)")
        ax.set_xlim(min(self.obsplan.MJDs), max(self.obsplan.MJDs))
        ax.set_ylim(max(self.obsplan.limmag) + 0.5, min(self.simLCs.flatten()) - 0.5)
        ax.set_xlabel("MJD [days]")
        ax.set_ylabel("%s mag" % self.filtername)
        ax.legend()

        plt.xticks(rotation = 90, ha = 'left')
        plt.xticks(np.arange(min(self.obsplan.MJDs) - 2, max(self.obsplan.MJDs) + 2, max(1, int((max(self.obsplan.MJDs) - min(self.obsplan.MJDs)) / 70.))))
        plt.tick_params(pad = 0)
        plt.tick_params(axis='x', which='major', labelsize=10)
        plt.tight_layout()
        
        ## function used for plot ticks
        def YYYYMMDD(date, pos):
            return Time(date, format = 'mjd', scale = 'utc').iso[:11]
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(YYYYMMDD))

        if save:
            plt.savefig("plots/simLCs_LCs_%s_%s.png" % (self.obsplan.planname, self.LCz.modelname))
            np.save("npy/simLCs_LCs_%s_%s.npy" % (self.obsplan.planname, self.LCz.modelname), self.simLCs)
            np.save("npy/simLCs_MJDs_%s_%s.npy" % (self.obsplan.planname, self.LCz.modelname), self.obsplan.MJDs)
            np.save("npy/simLCs_limmag_%s_%s.npy" % (self.obsplan.planname, self.LCz.modelname), self.obsplan.limmag)

        fig, ax = plt.subplots()
        ax.hist(self.simtexps)
        ax.set_xlabel("Explosion time MJD [days]")
        ax.set_ylabel("N")
        if save:
            plt.savefig("plots/simLCs_texp_%s_%s.png" % (self.obsplan.planname, self.LCz.modelname))
            np.save("npy/simLCs_texp_%s_%s.npy" % (self.obsplan.planname, self.LCz.modelname), self.simtexps)

        fig, ax = plt.subplots()
        ax.plot(self.zs, self.detprob, drawstyle = "steps", lw = 5, label = "Det. prob.")
        ax.plot(self.zs, self.detprobdiff, drawstyle = "steps", lw = 5, label = "Det. prob. (diff.)")
        ax.legend(loc = 3)
        ax.set_xlabel("z")
        ax.set_ylabel("Detection probability")
        if save:
            plt.savefig("plots/simLCs_prob_%s_%s.png" % (self.obsplan.planname, self.LCz.modelname))
            np.save("npy/simLCs_prob_%s_%s.npy" % (self.obsplan.planname, self.LCz.modelname), self.detprob)

        if not save:
            plt.show()
        

if __name__  == "__main__":

    # matplotlib
    from matplotlib import rc
    rc('text', usetex=True)
    rc('font', family='sans-serif')
    
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tick_params(axis='both', which='minor', labelsize=10)

    # filtername
    modelfile = "13z002E1.0.dat"
    modelfile = "yoon1209e10.fr"
    modeldir = "/home/fforster/Work/Model_LCs/models/yoon12msun"

    # survey telescope, filter and object type
    obsname = sys.argv[1] #"Blanco-DECam" #"KMTNet"
    filtername = sys.argv[2]
    objtype = sys.argv[3] # Ia II

    # maximum age of object at the start of the survey
    maxrestframeage = 3
    
    # start an observational plan
    if obsname == "HiTS14A":
        plan = obsplan(obsname = "Blanco-DECam", band = filtername, mode = 'custom', nfields = 40, nepochspernight = 4, ncontnights = 5, nnights = 6, nightfraction = 1., nread = 1, startmoonphase = -2, maxmoonphase = 15, doplot = True)
    elif obsname == "HiTS15A":
        plan = obsplan(obsname = "Blanco-DECam", band = filtername, mode = 'custom', nfields = 50, nepochspernight = 5, ncontnights = 6, nnights = 6, nightfraction = 1., nread = 1, startmoonphase = -2, maxmoonphase = 15, doplot = True)
    elif obsname == "VST-OmegaCam":
        plan = obsplan(obsname = obsname, band = filtername, mode = 'custom', nfields = 4, nepochspernight = 1, ncontnights = 120, nnights = 120, nightfraction = 5.32 / 100., nread = 1, startmoonphase = 0, maxmoonphase = 11.5, doplot = True)
    elif obsname == "KMTNet":
        plan = obsplan(obsname = obsname, band = filtername, mode = 'custom', nfields = 1, nepochspernight = 1, ncontnights = 120, nnights = 120, nightfraction = 0.043, nread = 3, startmoonphase = 0, maxmoonphase = 11.5, doplot = True)
    elif obsname == "KMTNetSNsurvey":
        plan = obsplan(obsname = "KMTNet", band = filtername, mode = 'custom', nfields = 5, nepochspernight = 3, nightfraction = 0.045, nread = 1, ncontnights = 180, nnights = 180, startmoonphase = 3, maxmoonphase = 15, doplot = True)
    elif obsname == "KMTNet17B":
        plan = obsplan(obsname = "KMTNet", band = filtername, mode = 'file', inputfile = "KMTNet17B.dat", nfields = 3, nepochspernight = 1, nightfraction = 0.5 / 4., nread = 3, doplot = True)
    elif obsname == "Clay-MegaCam17B":
        plan = obsplan(obsname = "Clay-MegaCam", band = filtername, mode = 'file', inputfile = "Clay-MegaCam17B.dat", nfields = 200, nepochspernight = 1, nightfraction = 0.5, nread = 1, doplot = True)
    elif obsname == "SNLS":
        plan = obsplan(obsname = "CFHT-MegaCam", band = filtername, mode = 'file', inputfile = "SNLS_%s.dat" % filtername, nfields = 1, nepochspernight = 1, nightfraction = 0.045, nread = 5, doplot = True)
    else:
        print "WARNING: undefined observatory"
        sys.exit()

    # extinction
    lAv = 0.187
    nAv = 10
    Avs = np.linspace(0, 4. * lAv, nAv)# np.hstack([0, np.logspace(-1.5, 0, 10)])
    Rv = 3.1

    # light curve model, star formation history and efficiency
    if objtype == 'II':
        SN = StellaModel(dir = modeldir, modelfile = modelfile, doplot = True)
        SFH = SFHs(SFH = "MD14")
        knorm = 0.0091
        IIPfrac = 0.54
        efficiency = knorm * IIPfrac
    elif objtype == 'Ia':
        SN = SNDDz(Mc = 1.44, M56Ni = 0.84, Ae = 0.15, tdays = np.linspace(0, 100, 500))
        SFH = SFHs(SFH = "Ia-P12")
        efficiency = 1.0
    else:
        print "WARNING: SN type"
        sys.exit()

    # number of redshift bins
    nz = 20 #20 #20

    # start survey
    newsurvey = survey_singlemodel(obsplan = plan, LCz = SN, Avs = Avs, Rv = Rv, lAv = lAv, SFH = SFH, efficiency = efficiency, filtername = filtername, nz = nz, maxrestframeage = maxrestframeage)

    # estimate maximum survey redshift
    newsurvey.estimate_maxredshift(zguess = 0.334, minprobdetection = 1e-4, minndetections = 5)

    # compute redshifts
    newsurvey.compute_zs()

    # update cosmology
    newsurvey.do_cosmology()

    # compute magnitudes
    newsurvey.compute_mags(plotmodel = True)
    
    # sample from distribution
    newsurvey.sample_events(nsim = 10000)
    
    # plot light curves
    newsurvey.plot_LCs(save = True)
    
