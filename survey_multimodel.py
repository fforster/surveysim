import numpy as np
import matplotlib.pyplot as plt


from constants import *
from obsplan import *
from LCz import *
from LCz_Av import *
from LCz_Av_params import *
from SFHs import *
from scipy.interpolate import interp1d
from scipy.stats import lognorm, norm, uniform  # leave this here, otherwise it fails!


# Cosmology stuff
sys.path.append("../cos_calc")
import cos_calc

# under construction!

# class that defines a survey, must be multiwavelength and multimodel!
class survey_multimodel(object):

    # initialize: needs observation plan, model, Avs, Rv, filtername, number of redshift bins
    def __init__(self, **kwargs):

        self.obsplan = kwargs["obsplan"]
        self.SFH = kwargs["SFH"]
        self.efficiency = kwargs["efficiency"]
        self.LCs = kwargs["LCs"]
        
        # maximum time for an object to have exploded before the beginning of the simulation to be considered detected
        self.maxrestframeage = 10
        if "maxrestframeage" in kwargs.keys():
            self.maxrestframeage = kwargs["maxrestframeage"]

    # set maximum redshift of simulation
    def set_maxz(self, maxz):

        self.maxz = maxz

    
    # do cosmology for dense redshift grid
    def do_cosmology(self):

        # cosmology
        nz = 100
        self.zs = np.linspace(0, self.maxz, nz)[1:]
        self.zbin = self.zs[1] - self.zs[0]
        self.zedges = np.linspace(self.maxz * 0.5 / nz, self.maxz * (1. + 0.5 / nz), nz + 1)

        h100, omega_m, omega_k, omega_lambda = Hnot / 100., OmegaM, 1. - (OmegaM + OmegaL), OmegaL
        cosmo =  np.array(map(lambda z: cos_calc.fn_cos_calc(h100, omega_m, omega_k, omega_lambda, z), self.zs))

        # cosmology interpolation functions
        self.Dcf = interp1d(self.zs, cosmo[:, 1]) # Mpc
        self.DLf = interp1d(self.zs, cosmo[:, 4]) # Mpc
        self.Dmf = interp1d(self.zs, cosmo[:, 5]) # Mpc
        self.dVdzdOmegaf = interp1d(self.zs, cspeed / (Hnot * 1e5 * np.sqrt((1. + self.zs)**3. * OmegaM + OmegaL)) * self.Dcf(self.zs)**2)

        # compute SFR given SFH at zs grid
        self.SFH.doSFR(self.zs)

        # compute cumulative number of SNe
        self.ndayssim = max(self.obsplan.MJDs) - min(self.obsplan.MJDs) + self.maxrestframeage * (1. + self.zs) # note that simulation length is a function of redshift
        self.totalSNe = self.efficiency * self.SFH.SFR / (1. + self.zs) * (self.ndayssim / yr2days) * self.dVdzdOmegaf(self.zs) * (self.obsplan.obs.FoV / ster_sqdeg) * self.obsplan.nfields  # true number of SNe

        # cumulative distribution over which to sample
        self.cumtotalSNe = np.cumsum(self.totalSNe * self.zbin)
        self.random2z = interp1d(self.cumtotalSNe / self.cumtotalSNe[-1], self.zs, bounds_error = False, fill_value = 'extrapolate')

        
    # sample events
    def sample_events(self, **kwargs):

        self.nsim = kwargs['nsim']  # number of events
        rvs = kwargs['rvs'] # function which return variable given random number
        bounds = kwargs['bounds'] # bounds for all variables
        pars = kwargs['pars'] # initial values for all variables (None for those which will be randomly sampled)

        doplot = False
        if 'doplot' in kwargs.keys():
            doplot = kwargs['doplot']
        doload = False
        if 'doload' in kwargs.keys():
            doload = kwargs['doload']
        dosave = False
        if 'dosave' in kwargs.keys():
            dosave = kwargs['dosave']
        if doload:
            dosave = False

        if not doload:
            # draw redshifts
            rands = np.random.random(size = self.nsim)
            self.logzs = np.log(self.random2z(rands))
            
            # draw explosion times
            tmin = bounds['texp'][0]
            tmax = bounds['texp'][1]
            rands = np.random.random(size = self.nsim)
            self.texps = tmin + 1. * rands * (tmax - tmin)
    
            # draw physical parameters
            params = {}
            for key in rvs.keys():
                rands = rvs[key](self.nsim)
                mask = (rands >= bounds[key][0]) & (rands <= bounds[key][1])
                rands = np.random.choice(rands[mask], size = self.nsim, replace = True)
                if key == 'logAv':
                    self.logAvs = rands
                else:
                    params[key] = rands
                if doplot:
                    fig, ax = plt.subplots()
                    if key[:3] == 'log':
                        rands = np.exp(rands)
                        key = key[3:]
                    ax.hist(rands, bins = 50)
                    ax.set_xlabel(key)
            
            # store variables
            for idx, key1 in enumerate(self.LCs.paramnames):
                for key2 in params.keys():
                    if key1 == key2:
                        pars[idx] = np.array(params[key2])
            for idx, key in enumerate(self.LCs.paramnames):
                if key not in params.keys():
                    pars[idx] = np.ones(self.nsim) * pars[idx]
    
            # create large array
            self.parsarray = np.zeros((self.nsim, size(pars)))
            for idx, par in enumerate(pars):
                self.parsarray[:, idx] = par


        if doplot:
            fig, ax = plt.subplots(figsize = (20, 10))

        if doload:
            import pickle
            LCsfile = "pickles/%s_%s_LCs_%i.pkl" % (self.LCs.modelname, self.obsplan.planname, self.nsim)
            self.LCsamples = pickle.load(open(LCsfile, 'rb'))
            parsfile = "pickles/%s_%s_params_%i.pkl" % (self.LCs.modelname, self.obsplan.planname, self.nsim)
            self.parsamples = np.array(pickle.load(open(parsfile, 'rb'))).transpose()
            self.logzs = self.parsamples[0]
            self.texps = self.parsamples[1]
            self.logAvs = self.parsamples[2]
            self.parsarray = self.parsamples[3:]
            
        else:
            self.LCsamples = []
            self.parsamples = []
        
        for i in range(self.nsim):
            
            if not doload:
                if np.mod(i, 100) == 0:
                    print i

            if not doload:
                self.LCsamples.append(self.LCs.evalmodel(1., self.texps[i], self.logzs[i], self.logAvs[i], self.parsarray[i]))
                self.parsamples.append(np.hstack([self.logzs[i], self.texps[i], self.logAvs[i], self.parsarray[i]]))

            if doplot and i < 1000: # avoid plotting more than 1000 LCs
                for band in self.LCs.uniquefilters:
                    maskband = self.LCs.maskband[band]
                    if i == 0:
                        ax.plot(self.LCs.mjd[maskband], self.obsplan.limmag[maskband], lw = 4, c = self.LCs.bandcolors[band])
                    masklim = self.LCsamples[i][band] < self.obsplan.limmag[maskband] + 2
                    ax.plot(self.LCs.mjd[maskband][masklim], self.LCsamples[i][band][masklim], c = self.LCs.bandcolors[band], alpha = 0.1)

        self.parsamples = np.array(self.parsamples).transpose()

        if doplot:
            ax.set_ylim(ax.get_ylim()[::-1])
            
        if dosave:
            import pickle
            LCsfile = "pickles/%s_%s_LCs_%i.pkl" % (self.LCs.modelname, self.obsplan.planname, self.nsim)
            pickle.dump(self.LCsamples, open(LCsfile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)
            parsfile = "pickles/%s_%s_params_%i.pkl" % (self.LCs.modelname, self.obsplan.planname, self.nsim)
            pickle.dump(self.parsamples, open(parsfile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)
        
    def do_efficiency(self, **kwargs):

        doplot = False
        if 'doplot' in kwargs.keys():
            doplot = kwargs['doplot']
        verbose = False
        if 'verbose' in kwargs.keys():
            verbose = kwargs['verbose']

        if doplot:
            fig, ax = plt.subplots(figsize = (20, 10))

        # count number of detections in every band
        matches = np.zeros((len(self.LCsamples), len(self.obsplan.uniquebands)))
        for idx, LCsample in enumerate(self.LCsamples):

            match = {}
            
            if verbose and np.mod(idx, 100) == 0:
                print(idx)

            for idxb, band in enumerate(self.obsplan.uniquebands):

                maskband = self.LCs.maskband[band]
                masklim = LCsample[band] < self.obsplan.limmag[maskband] - np.sqrt(2.)
                matches[idx, idxb] = np.sum(masklim)

                #if doplot:
                #    if idx == 0:
                #        ax.plot(self.obsplan.MJDs[maskband], self.obsplan.limmag[maskband], lw = 4, zorder = 1000, c = self.LCs.bandcolors[band])
                #        
                #    ax.plot(self.obsplan.MJDs[maskband][masklim], LCsample[band][masklim], alpha = 0.1, c = self.LCs.bandcolors[band])

        # sort number of detections by redshift or other properties
        nmatches = np.sum(matches, axis = 1)
        detections = (nmatches >= 2)

        labels = np.concatenate([np.array(['logz', 'texp', 'logAv'], dtype = str), np.array(self.LCs.paramnames, dtype = str)])
        self.x_effs = {}
        self.effs = {}
        for vallabel, valin in zip(labels, self.parsamples):
            if min(valin) == max(valin):
                continue
            if vallabel == 'logz':
                vallabel = 'z'
                valin = np.exp(valin)
            if vallabel == 'mdot':
                vallabel = 'log10mdot'
                valin = np.log10(valin)
            if vallabel == 'logAv':
                vallabel = 'log10Av'
                valin = valin / np.log(10.)
            idxsort = np.argsort(valin)
            valin = valin[idxsort]
            valout = valin[detections[idxsort]]
            nn = np.cumsum(np.ones_like(valin))
            nin, bin_edgesin = np.histogram(valin, bins = 25)
            nout, bin_edgesout = np.histogram(valout, bins = bin_edgesin)
            self.x_effs[vallabel] = (bin_edgesin[1:] + bin_edgesin[:-1]) / 2.
            self.effs[vallabel] = interp1d(self.x_effs[vallabel], 1. * nout / nin, bounds_error = False, fill_value = 0)
            
            if doplot:
                fig, ax = plt.subplots()
                ax.plot(valin, range(len(valin)), label = 'Explosions')
                ax.plot(valout, range(len(valout)), label = 'Detections')
                #ax.plot((self.x_effs[vallabel], self.effs[vallabel](self.x_effs[vallabel]), linestyle = 'steps-mid')
                ax.set_xlabel(vallabel)
                ax.set_ylabel('N')
                ax.set_ylim(0, ax.get_ylim()[1])
                plt.legend(loc = 2, fontsize = 10)
        
        
        if doplot:
            ax.set_ylim(ax.get_ylim()[::-1])



        

        # count number of detecions in any filters
                   


        # explosion time

        # Av

        # physical parameters

        
#    # compute
#    def compute_zs(self):
#
#        self.zs = np.linspace(0, self.maxz, self.nz + 1)[1:]
#        self.zbin = self.zs[1] - self.zs[0]
#        self.zedges = np.linspace(self.maxz * 0.5 / self.nz, self.maxz * (1. + 0.5 / self.nz), self.nz + 1)
#
#        
#    def do_cosmology(self):
#
#        # cosmology
#        self.DL = np.zeros_like(self.zs)
#        self.Dc = np.zeros_like(self.zs)
#        self.Dm = np.zeros_like(self.zs)
#        self.dVdzdOmega = np.zeros_like(self.zs)
#        h100, omega_m, omega_k, omega_lambda = Hnot / 100., OmegaM, 1. - (OmegaM + OmegaL), OmegaL
#        cosmo =  np.array(map(lambda z: cos_calc.fn_cos_calc(h100, omega_m, omega_k, omega_lambda, z), self.zs))
#        self.Dc = cosmo[:, 1] # Mpc
#        self.DL = cosmo[:, 4] # Mpc
#        self.Dm = cosmo[:, 5] # Mpc
#        self.dVdzdOmega = cspeed / (Hnot * 1e5 * np.sqrt((1. + self.zs)**3. * OmegaM + OmegaL)) * self.Dc**2
#
#        # compute SFR
#        self.SFH.doSFR(self.zs)
#            
#    def estimate_maxredshift(self, **kwargs):
#
#        print "   Looking for maximum redshift..."
#
#        zguess = kwargs["zguess"]
#        self.minprobdetection = kwargs["minprobdetection"]
#        self.minndetections = kwargs["minndetections"]
#
#        probdet = [0, 0, 0]
#        delta = 0.01
#        while True:
#
#            # create redshift array
#            self.zs = np.array([(1. - delta) *  zguess, zguess, (1. + delta) * zguess])
#
#            # do cosmology
#            self.do_cosmology()
#            
#            # create LCz_Av object (use minimum Av)
#            self.LCz_Av = LCz_Av(LCz = self.LCz, Av = np.array([min(self.Avs)]), Rv = 4.5, lAv = self.lAv, zs = self.zs, DL = self.DL, Dm = self.Dm, filtername = self.filtername)
#            self.LCz_Av.compute_mags()
#            
#            # generate 1/prob SNe and check whether none of them are detected, otherwise increase redshift guess. If no detections, decrease redshift guess
#            nsim = int(1. / self.minprobdetection)
#            simmags = np.array(map(lambda iz: self.LCz_Av.simulate_randomLC(nsim = nsim, iz = iz, MJDs = self.obsplan.MJDs, maxrestframeage = self.maxrestframeage)[3], range(len(self.zs))))
#            
#            # compute detection probabilities
#            probdet = map(lambda iz: 1. * np.sum(np.sum(simmags[iz, :, :] <= self.obsplan.limmag, axis = 1) >= self.minndetections) / nsim, range(len(self.zs)))
#
#            # correct delta
#            if probdet[2] <= self.minprobdetection:
#                if probdet[1] <= self.minprobdetection:
#                    if probdet[0] <= self.minprobdetection: # below 1st
#                        zguess = min(self.zs)
#                        delta = 0.5
#                    else: # between 1st and 2nd
#                        zguess = np.average(self.zs[0:2])
#                        delta = delta / 2.
#                else: # between 2nd and 3rd
#                    delta = delta / 2.
#                    if delta <= 0.01:
#                        self.maxz = max(self.zs)
#
#                        # compute maximum length of simulation by doing one simulation
#                        self.ndayssim = self.LCz_Av.simulate_randomLC(nsim = 1, iz = 2, MJDs = self.obsplan.MJDs, maxrestframeage = self.maxrestframeage)[0]
#
#                        # EXIT here only
#                        return 
#                    
#            else: # above 3rd
#                zguess = max(self.zs)
#                delta = min(2. * delta, 0.5)
#
#            print "      Correction applied. Current range is", self.zs
#
#    # compute magnitudes and set Av distribution
#    def compute_mags(self, **kwargs):
#
#        plotmodel = False
#        if "plotmodel" in kwargs.keys():
#            plotmodel = kwargs["plotmodel"]
#        
#        print "      Computing magnitudes..."
#
#        # create LCz_Av object (use minimum Av)
#        self.LCz_Av = LCz_Av(LCz = self.LCz, Av = self.Avs, Rv = self.Rv, zs = self.zs, DL = self.DL, Dm = self.Dm, filtername = self.filtername)
#        self.LCz_Av.compute_mags(plotmodel = plotmodel)
#
#        # generate Av probability distribution
#        self.LCz_Av.set_Avdistribution(lAv = self.lAv)
#
#    # sample light curves from the survey    
#    def sample_events(self, **kwargs):
#
#        #print "      Sampling light curves..."
#
#        # save number of simulated LCs
#        self.nsim = kwargs["nsim"]
#        
#        # think how to sample more efficiently from this distribution...
#        self.totalSNe = self.efficiency * self.SFH.SFR / (1. + self.zs) * (self.ndayssim / yr2days) * self.dVdzdOmega * (self.obsplan.obs.FoV / ster_sqdeg) * self.obsplan.nfields  # true number of SNe
#
#        # cumulative distribution over which to sample
#        self.cumtotalSNe = np.cumsum(self.totalSNe * self.zbin)
#
#        # generate random numbers for redshift of SNe
#        rands = np.random.random(size = self.nsim) * self.cumtotalSNe[-1]
#        izs = np.array(map(lambda r: np.sum(self.cumtotalSNe < r), rands), dtype = int)
#        self.simzs = self.zs[izs]
#
#        # reset 
#        if hasattr(self, "simLCs"):
#            del self.simLCs, self.simdts, self.simtexps, self.simiAvs, self.nsimz, self.ndetections
#
#        # initialize arrays to store simulated light curves
#        self.nsimz = np.zeros(self.nz) # number of simulated LCs at different redshifts
#        self.simdts = np.zeros(self.nz)
#        self.ndetections = np.zeros(self.nz)
#        self.ndetectionsdiff = np.zeros(self.nz)
#        self.detprob = np.zeros(self.nz)
#        self.detprobdiff = np.zeros(self.nz)
#        self.simtexps = np.zeros(self.nsim)
#        self.simiAvs = np.zeros(self.nsim, dtype = int)
#        self.simLCs = np.ones((self.nsim, len(self.obsplan.MJDs))) * 40. # assume 40 mag pre-explosion
#        
#        # simulate light curves and store them
#        l1 = 0
#        for iz in range(self.nz):
#
#            self.nsimz[iz] = int(np.sum(izs == iz))
#
#            if self.nsimz[iz] > 0:
#
#                # simulate and count detections
#                simdt, simtexp, simiAv, simmags = self.LCz_Av.simulate_randomLC(nsim = self.nsimz[iz], iz = iz, MJDs = self.obsplan.MJDs, maxrestframeage = self.maxrestframeage)
#                ndetections = np.sum(map(lambda mags: np.sum(mags <= self.obsplan.limmag) >= self.minndetections, simmags))
#                ndetectionsdiff = np.sum(map(lambda mags: np.sum(mags <= self.obsplan.limmag - 2.5 * np.log10(np.sqrt(2.))) >= self.minndetections, simmags))
#
#                # fill arrays with light curve information and statistics
#                self.simdts[iz] = simdt
#                self.ndetections[iz] = ndetections
#                self.ndetectionsdiff[iz] = ndetectionsdiff
#                self.detprob[iz] = 1. * ndetections / self.nsimz[iz]
#                self.detprobdiff[iz] = 1. * ndetectionsdiff / self.nsimz[iz]
#                l2 = int(l1 + self.nsimz[iz])
#                self.simtexps[l1: l2] = simtexp
#                self.simiAvs[l1: l2] = simiAv
#                self.simLCs[l1: l2, :] = simmags
#                l1 = int(l1 + self.nsimz[iz])
#
#    # plot the simulated light curves and some statistics
#    def plot_LCs(self, **kwargs):
#
#        save = True
#        if "save" in kwargs.keys():
#            save = kwargs["save"]
#
#        print "      Plotting light curves and statistics..."
#
#        if save:
#            np.save("npy/simLCs_zs-CDF_%s_%s.npy" % (self.obsplan.planname, self.LCz.modelname), {"zs": self.zs, "cumtotalSNe": self.cumtotalSNe})
#        
#        fig, ax = plt.subplots() 
#        ax.plot(self.zs, self.cumtotalSNe, label = r"$\int_0^z \eta ~\frac{SFR(z)}{1+z} ~T_{\rm sim} ~\frac{dV(z)}{dz d\Omega} ~\Delta\Omega ~n_{\rm fields} ~dz $", c = 'k', zorder = 100)
#        scale = self.cumtotalSNe[-1] / self.nsim
#        hist, edges = np.histogram(self.simzs, bins = self.zedges)
#        ax.plot((edges[:-1] + edges[1:]) / 2., np.cumsum(hist) * scale, label = "Explosions", c = 'r', lw = 5, alpha = 0.5, drawstyle = 'steps-post')
#        ax.plot(self.zs, np.cumsum(self.ndetections) * scale, label = "Detections", c = 'b', lw = 5, drawstyle = 'steps-post')
#        ax.plot(self.zs, np.cumsum(self.ndetectionsdiff) * scale, label = "Detections (diff.)", c = 'g', lw = 5, drawstyle = 'steps-post')
#        ax.legend(loc = 2, fontsize = 14, framealpha = 0.7)
#        ax.set_title("Total detections: %.1f, %.1f (diff.)" % (1. * np.sum(self.ndetections) / self.nsim * self.cumtotalSNe[-1], 1. * np.sum(self.ndetectionsdiff) / self.nsim * self.cumtotalSNe[-1]))
#        ax.set_xlabel("z")
#        ax.set_ylabel("Event cumulative distribution")
#        ax.set_xlim(0, self.maxz)
#        if save:
#            plt.savefig("plots/simLCs_expvsdet_%s_%s.png" % (self.obsplan.planname, self.LCz.modelname))
#
#        fig, ax = plt.subplots(figsize = (13, 7))
#        map(lambda LC: ax.plot(self.obsplan.MJDs, LC, alpha = 0.1), self.simLCs[np.random.choice(range(self.nsim), size = 5000, replace = True)])
#        ax.plot(self.obsplan.MJDs, self.obsplan.limmag, c = 'r', label = "lim. mag.")
#        ax.plot(self.obsplan.MJDs, self.obsplan.limmag - 2.5 * np.log10(np.sqrt(2.)), c = 'r', ls = ':', label = "lim. mag. (diff.)")
#        ax.set_xlim(min(self.obsplan.MJDs), max(self.obsplan.MJDs))
#        ax.set_ylim(max(self.obsplan.limmag) + 0.5, min(self.simLCs.flatten()) - 0.5)
#        ax.set_xlabel("MJD [days]")
#        ax.set_ylabel("%s mag" % self.filtername)
#        ax.legend()
#
#        plt.xticks(rotation = 90, ha = 'left')
#        plt.xticks(np.arange(min(self.obsplan.MJDs) - 2, max(self.obsplan.MJDs) + 2, max(1, int((max(self.obsplan.MJDs) - min(self.obsplan.MJDs)) / 70.))))
#        plt.tick_params(pad = 0)
#        plt.tick_params(axis='x', which='major', labelsize=10)
#        plt.tight_layout()
#        
#        ## function used for plot ticks
#        def YYYYMMDD(date, pos):
#            return Time(date, format = 'mjd', scale = 'utc').iso[:11]
#        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(YYYYMMDD))
#
#        if save:
#            plt.savefig("plots/simLCs_LCs_%s_%s.png" % (self.obsplan.planname, self.LCz.modelname))
#            np.save("npy/simLCs_LCs_%s_%s.npy" % (self.obsplan.planname, self.LCz.modelname), self.simLCs)
#            np.save("npy/simLCs_MJDs_%s_%s.npy" % (self.obsplan.planname, self.LCz.modelname), self.obsplan.MJDs)
#            np.save("npy/simLCs_limmag_%s_%s.npy" % (self.obsplan.planname, self.LCz.modelname), self.obsplan.limmag)
#
#        fig, ax = plt.subplots()
#        ax.hist(self.simtexps)
#        ax.set_xlabel("Explosion time MJD [days]")
#        ax.set_ylabel("N")
#        if save:
#            plt.savefig("plots/simLCs_texp_%s_%s.png" % (self.obsplan.planname, self.LCz.modelname))
#            np.save("npy/simLCs_texp_%s_%s.npy" % (self.obsplan.planname, self.LCz.modelname), self.simtexps)
#
#        fig, ax = plt.subplots()
#        ax.plot(self.zs, self.detprob, drawstyle = "steps", lw = 5, label = "Det. prob.")
#        ax.plot(self.zs, self.detprobdiff, drawstyle = "steps", lw = 5, label = "Det. prob. (diff.)")
#        ax.legend(loc = 3)
#        ax.set_xlabel("z")
#        ax.set_ylabel("Detection probability")
#        if save:
#            plt.savefig("plots/simLCs_prob_%s_%s.png" % (self.obsplan.planname, self.LCz.modelname))
#            np.save("npy/simLCs_prob_%s_%s.npy" % (self.obsplan.planname, self.LCz.modelname), self.detprob)
#
#        if not save:
#            plt.show()
        

if __name__  == "__main__":

    # matplotlib
    from matplotlib import rc
    rc('text', usetex=True)
    rc('font', family='sans-serif')
    
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tick_params(axis='both', which='minor', labelsize=10)

    # survey telescope, filter and object type
    obsname = sys.argv[1] #"Blanco-DECam" #"KMTNet"
    modelname = sys.argv[2] # MoriyaWindAcc

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
        plan = obsplan(obsname = "CFHT-MegaCam", mode = 'file-cols', inputfile = "SNLS_bands.dat", nfields = 1, nepochspernight = 1, nightfraction = 0.045, nread = 5, doplot = True, doload = True, bandcolors = {'g': 'g', 'r': 'r', 'i': 'brown', 'z': 'k'})
    else:
        print "WARNING: undefined observatory"
        sys.exit()

    # extinction
    lAv = 0.187

    # models
    modelsdir = "/home/fforster/Work/surveysim/models"
    data = np.genfromtxt("%s/%s/modellist.txt" % (modelsdir, modelname), dtype = str, usecols = (0, 1, 3, 5, 7, 9, 10, 11)).transpose()
    data[data == 'no'] = 0
    modelfile, modelmsun, modele51, modelmdot, modelrcsm, modelvwind0, modelvwindinf, modelbeta = data

    # parameters, specific to MoriyaWindAcc models
    modelfile = np.array(modelfile, dtype = str)
    modelmsun = np.array(modelmsun, dtype = float)
    modelfoe = np.array(modele51, dtype = float) / 1e51
    modelmdot = np.array(modelmdot, dtype = float)
    modelrcsm = np.array(modelrcsm, dtype = float) / 1e15
    modelvwind0 = np.array(modelvwind0, dtype = float)  # do not use this
    modelvwindinf = np.array(modelvwindinf, dtype = float)
    modelbeta = np.array(modelbeta, dtype = float)
    params = np.vstack([modelmsun, modelfoe, modelmdot, modelrcsm, modelvwindinf, modelbeta]).transpose()
    try:
        files = np.array(map(lambda name: "%s.fr" % name, modelfile))
    except:
        files = "%s.fr" % modelfile

    # Redshift, Avs and time
    nz = 30
    ntimes = 100
    nAvs = 10
    zs = np.logspace(-3, 0, nz)
    times = np.logspace(-3, 3, ntimes)
    Avs = np.logspace(-4, 1, nAvs)
    Rv = 3.25

    # initialize LCz_Av_params models
    paramnames = ["mass", "energy", "mdot", "rcsm", "vwindinf", "beta"]
    paramunits = ["Msun", "B", "Msun/yr", "1e15 cm", "km/s", ""] 
    LCs = LCz_Av_params(modelsdir = modelsdir, modelname = modelname, files = files, paramnames = paramnames, paramunits = paramunits, params = params, zs = zs, Avs = Avs, Rv = Rv, times = times)

    # do cosmology
    LCs.docosmo()

    # compute models in given bands
    LCs.compute_models(bands = ['u', 'g', 'r', 'i', 'z'], load = True)#, save = True)#, 'r'])#, 'i', 'z'])
    
    # set metric
    LCs.setmetric(metric = np.array([1., 1., 1e-6, 1., 10., 1.]), logscale = np.array([False, False, True, False, False, True], dtype = bool))
    
    # set observations
    LCs.set_observations(mjd = plan.MJDs, flux = None, e_flux = None, filters = plan.bands, objname = None, plot = False, bandcolors = {'g': 'g', 'r': 'r', 'i': 'brown', 'z': 'k'})

    # star formation
    SFH = SFHs(SFH = "MD14")
    knorm = 0.0091
    IIPfrac = 0.54
    efficiency = knorm * IIPfrac
    
    # start survey
    newsurvey = survey_multimodel(obsplan = plan, SFH = SFH, efficiency = efficiency, LCs = LCs, maxrestframeage = maxrestframeage)

    # set maximum redshift
    newsurvey.set_maxz(0.9)
    
    # do cosmology with dense grid
    newsurvey.do_cosmology()

    # set rvs
    minMJD, maxMJD = min(newsurvey.obsplan.MJDs) - newsurvey.maxrestframeage * (1. + max(newsurvey.zs)), max(newsurvey.obsplan.MJDs)
    rvs = {'texp': lambda nsim: uniform.rvs(loc = minMJD, scale = maxMJD - minMJD, size = nsim), \
           'logAv': lambda nsim: norm.rvs(loc = np.log(0.075), scale = 1., size = nsim), \
           #'mass': lambda nsim: norm.rvs(loc = 14, scale = 3, size = nsim), \
           'mass': lambda nsim: 10. * lognorm.rvs(0.7, size = nsim), \
           'energy': lambda nsim: norm.rvs(loc = 1., scale = 1., size = nsim), \
           'mdot': lambda nsim: 5e-4 * lognorm.rvs(1., size = nsim), \
           'beta': lambda nsim: 7. * lognorm.rvs(1., size = nsim)}
    bounds = {'texp': [minMJD, maxMJD], \
              'logAv': [np.log(1e-4), np.log(10.)], \
              'mass': [12, 16], \
              'energy': [0.5, 2.], \
              'mdot': [1e-6, 1e-2], \
              'beta': [1., 5.]}

    # default physical values
    mass = None
    energy = None
    mdot = None
    rcsm = 1. # 1e15
    vwindinf = 10.
    beta = None
    pars = np.array([mass, energy, mdot, rcsm, vwindinf, beta]) # must be in same order as paramnames

    # sample events    
    newsurvey.sample_events(nsim = 50000, doload = True, doplot = False, rvs = rvs, bounds = bounds, pars = pars)

    # measure detections and efficiency
    newsurvey.do_efficiency(doplot = True, verbose = False)
    
    ## estimate maximum survey redshift
    #newsurvey.estimate_maxredshift(zguess = 0.334, minprobdetection = 1e-4, minndetections = 5)
    #
    ## compute redshifts
    #newsurvey.compute_zs()
    #
    ## update cosmology
    #newsurvey.do_cosmology()
    #
    ## compute magnitudes
    #newsurvey.compute_mags(plotmodel = True)
    #
    ## sample from distribution
    #newsurvey.sample_events(nsim = 10000)
    #
    ## plot light curves
    #newsurvey.plot_LCs(save = True)
    
    plt.show()