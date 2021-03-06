import numpy as np
import matplotlib.pyplot as plt


from constants import *
from obsplan import *
from LCz import *
from LCz_Av import *
from LCz_Av_params import *
from SFHs import *
from scipy.interpolate import interp1d
from scipy.stats import lognorm, norm, uniform, multivariate_normal  # leave this here, otherwise it fails!


# Cosmology stuff
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

        self.surveyname = "%s_%s_%s" % (self.obsplan.planname, self.SFH.label, self.LCs.modelname)
        print("Survey name: %s" % self.surveyname)
        
        # maximum time for an object to have exploded before the beginning of the simulation to be considered detected
        self.maxrestframeage = 5. #10
        if "maxrestframeage" in kwargs.keys():
            self.maxrestframeage = float(kwargs["maxrestframeage"])
        # minimum time for an object to be detected (ensure long light curves are detected)
        self.minrestframeage = 30. #10
        if "minrestframeage" in kwargs.keys():
            self.minrestframeage = float(kwargs["minrestframeage"])
            
        if self.obsplan.mode == 'maf':
            self.mafcounter = 0

    # set maximum redshift of simulation
    def set_maxz(self, maxz):

        self.maxz = maxz

    
    # do cosmology for dense redshift grid
    def do_cosmology(self):

        if not hasattr(self, "zs") or self.obsplan.mode != "maf":
            # cosmology
            nz = 100
            self.zs = np.linspace(0, self.maxz, nz)[1:]
            self.zbin = self.zs[1] - self.zs[0]
            self.zedges = np.linspace(self.maxz * 0.5 / nz, self.maxz * (1. + 0.5 / nz), nz + 1)

            h100, omega_m, omega_k, omega_lambda = Hnot / 100., OmegaM, 1. - (OmegaM + OmegaL), OmegaL
            cosmo =  np.array(list(map(lambda z: cos_calc.fn_cos_calc(h100, omega_m, omega_k, omega_lambda, z), self.zs)))
            
            # cosmology interpolation functions
            self.Dcf = interp1d(self.zs, cosmo[:, 1]) # Mpc
            self.DLf = interp1d(self.zs, cosmo[:, 4]) # Mpc
            self.Dmf = interp1d(self.zs, cosmo[:, 5])
            self.dVdzdOmegaf = interp1d(self.zs, cspeed / (Hnot * 1e5 * np.sqrt((1. + self.zs)**3. * OmegaM + OmegaL)) * self.Dcf(self.zs)**2)

            # compute SFR given SFH at zs grid
            self.SFH.doSFR(self.zs)

            self.totalSNe_t = self.efficiency * self.SFH.SFR / (1. + self.zs) * self.dVdzdOmegaf(self.zs) * (self.obsplan.obs.FoV / ster_sqdeg) * self.obsplan.nfields  # true number of SNe

            # cumulative distribution over which to sample
            self.cumtotalSNe_t = np.cumsum(self.totalSNe_t * self.zbin)
            self.random2z = interp1d(self.cumtotalSNe_t / self.cumtotalSNe_t[-1], self.zs, bounds_error = False, fill_value = 'extrapolate')

            
        # compute time and multiply rates (when running maf this gets updated)
        self.ndayssim = max(self.obsplan.MJDs) - min(self.obsplan.MJDs) + (20 + self.maxrestframeage) * (1. + self.zs) # note that simulation length is a function of redshift
        self.totalSNe = self.totalSNe_t * (self.ndayssim / yr2days)
        self.cumtotalSNe = self.cumtotalSNe_t * (self.ndayssim / yr2days)
        
        #print("Total number of events expected to occur within observed volume and simulation time is %i" % self.cumtotalSNe[-1])


        
    # sample events
    def sample_events(self, **kwargs):

        # code
        
        self.nsim = kwargs['nsim']  # number of events
        rvs = kwargs['rvs'] # function which return variable given random number
        bounds = kwargs['bounds'] # bounds for all variables
        pars = kwargs['pars'] # initial values for all variables (None for those which will be randomly sampled)

        doplot = False
        if 'doplot' in kwargs.keys():
            doplot = bool(kwargs['doplot'])
        doload = False
        if 'doload' in kwargs.keys():
            doload = bool(kwargs['doload'])
        dosave = False
        if 'dosave' in kwargs.keys():
            dosave = bool(kwargs['dosave'])
        if doload:
            dosave = False

        # whether to check detection time after emergence or not (if not, after explosion)
        self.doemergence = False
        if "doemergence" in kwargs.keys():
            self.doemergence = kwargs['doemergence']

        if not doload:

            # special variables are redshift, explosion time, Av
            # other variables are interpolated in regular grid
            # objetive of this section is fill self.parsarray array
            
            # sample redshifts first according to SFH, cosmology and maximum redshift
            rands = np.random.random(size = self.nsim)
            self.logzs = np.log(self.random2z(rands))
            
            # sample all parameters, instrinsic (explosion properties) parameters are stored in params
            params = {}
            for key in rvs.keys():
                rands = rvs[key](self.nsim)
                mask = (rands >= bounds[key][0]) & (rands <= bounds[key][1])
                rands = np.random.choice(rands[mask], size = self.nsim, replace = True)
                # extrinsic parameters first
                if key == 'texp': # explosion time
                    self.texps = rands
                elif key == 'logAv': # Av extinction
                    self.logAvs = rands
                else:
                    params[key] = rands
                if doplot:
                    fig, ax = plt.subplots()
                    #if key[:3] == 'log':
                    #    rands = np.exp(rands)
                    #    key = key[3:]
                    ax.hist(rands, bins = 50)
                    ax.set_xlabel(key)
                    plt.savefig("plots/%s_%s.png" % (self.surveyname, key))

            
            # loop among paramnames and store variables in params.keys (if not mdot) in pars array
            for idx, key1 in enumerate(self.LCs.paramnames):
                # keys in params.keys (or mdot)
                for key2 in params.keys():
                    if key1 == key2 or (key1 == "mdot" and key2 == "log10mdot"):
                        pars[idx] = np.array(params[key2])

            # loop among paramnames and store variables not in params.keys (if not mdot) in pars array as constant array
            for idx, key in enumerate(self.LCs.paramnames):
                # keys not in params.keys
                if key not in params.keys() and (key != 'mdot'):
                    pars[idx] = np.ones(self.nsim) * pars[idx]
    
            # create large array of variables in parsarray array
            self.parsarray = np.zeros((self.nsim, size(pars)))
            # this will be filled with pars
            for idx, par in enumerate(pars):
                self.parsarray[:, idx] = par


        if doplot:
            fig, ax = plt.subplots(figsize = (20, 10))


        # load LCs and physical parameters for the given MJDs
        if doload:

            import pickle

            # light curves
            LCsfile = "%s/pickles/%s_%s_LCs_%i.pkl" % (os.environ["SURVEYSIM_PATH"], self.LCs.modelname, self.obsplan.planname, self.nsim)
            self.LCsamples = pickle.load(open(LCsfile, 'rb'))

            # physical parameters
            parsfile = "%s/pickles/%s_%s_params_%i.pkl" % (os.environ["SURVEYSIM_PATH"], self.LCs.modelname, self.obsplan.planname, self.nsim)
            self.parsamples = np.array(pickle.load(open(parsfile, 'rb'))).transpose()
            self.logzs = self.parsamples[0]
            self.texps = self.parsamples[1]
            self.logAvs = self.parsamples[2]
            self.parsarray = self.parsamples[3:]

            # emergence time
            if self.doemergence:
                temergencefile = "%s/pickles/%s_%s_temergence_%i.pkl" % (os.environ["SURVEYSIM_PATH"], self.LCs.modelname, self.obsplan.planname, self.nsim)
                self.temergence = np.array(pickle.load(open(temergencefile, 'rb'))).transpose()
            
        else:
            self.LCsamples = []
            self.parsamples = []
            self.temergence = []
            
        #if not doload:
        #    print("Simulating light curves...")

        if self.doemergence:
            mag_emergence = -13 # absolute magnitude of emergence

        # get LCs and plot them if necessary
        for i in range(self.nsim):

            # print status
            if not doload and self.obsplan.mode != 'maf':
                if np.mod(i, 10) == 0:
                    print("\rSample %i" % i, end = "")

            # sample light curves
            if not doload:
                
                # save only first list with light curve at given time (no reference values)
                scale = 1.
                stretch = 1.
                if hasattr(self, "scale"):
                    scale = self.scales[i]
                if hasattr(self, "stretch"):
                    scale = self.stretchs[i]
                self.LCsamples.append(self.LCs.evalmodel(scale, self.texps[i], self.logzs[i], self.logAvs[i], self.parsarray[i])[0])

                # find approximate time of emergence (when abs g < mag_emergence)
                if self.doemergence:
                    masknice = self.LCs.times < 30
                    try:
                        niceLC = self.LCs.evalmodel(1., self.texps[i], self.logzs[i], self.logAvs[i], self.parsarray[i], nice = True)[0]['g'][masknice]
                    except:
                        niceLC = self.LCs.evalmodel(1., self.texps[i], self.logzs[i], self.logAvs[i], self.parsarray[i], nice = True)[0]['B'][masknice]
                        
                    #niceLC = niceLC
                    mask = (niceLC - self.Dmf(np.exp(self.logzs[i])) <= mag_emergence)
                    if np.sum(mask) > 0:
                        self.temergence.append(np.min(self.LCs.times[masknice][mask]))
                    else:
                        self.temergence.append(-2)

                # append physical parameters to array
                self.parsamples.append(np.hstack([self.logzs[i], self.texps[i], self.logAvs[i], self.parsarray[i]]))

            # plot first 1000 LCs
            if doplot and i < 1000: # avoid plotting more than 1000 LCs
                for band in self.LCs.uniquefilters:
                    maskband = self.LCs.maskband[band]
                    if i == 0:
                        ax.plot(self.LCs.mjd[maskband], self.obsplan.limmag[maskband], lw = 4, c = self.LCs.bandcolors[band])
                        ax.plot(self.LCs.mjd[maskband], self.obsplan.limmag[maskband] - 2.5 * np.log10(np.sqrt(2.)), lw = 4, c = self.LCs.bandcolors[band], ls = ':')
                    masklim = self.LCsamples[i][band] < self.obsplan.limmag[maskband] + 2
                    ax.plot(self.LCs.mjd[maskband][masklim], self.LCsamples[i][band][masklim], c = self.LCs.bandcolors[band], alpha = 0.1)

        # numpify physical parameters
        self.parsamples = np.array(self.parsamples).transpose()
        if self.doemergence:
            self.temergence = np.array(self.temergence).transpose()

        # invert axis
        if doplot:
            ax.set_ylim(ax.get_ylim()[::-1])
            ax.set_ylim(25, ax.get_ylim()[1])
            plt.savefig("plots/%s_LCs.png" % (self.surveyname))


        # save LCs and physical parameters
        if dosave:
            import pickle

            # LCs
            LCsfile = "%s/pickles/%s_%s_LCs_%i.pkl" % (os.environ["SURVEYSIM_PATH"], self.LCs.modelname, self.obsplan.planname, self.nsim)
            pickle.dump(self.LCsamples, open(LCsfile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)

            # physical parameters
            parsfile = "%s/pickles/%s_%s_params_%i.pkl" % (os.environ["SURVEYSIM_PATH"], self.LCs.modelname, self.obsplan.planname, self.nsim)
            pickle.dump(self.parsamples, open(parsfile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)
            if self.doemergence:
                temergencefile = "%s/pickles/%s_%s_temergence_%i.pkl" % (os.environ["SURVEYSIM_PATH"], self.LCs.modelname, self.obsplan.planname, self.nsim)
                pickle.dump(self.temergence, open(temergencefile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)

        # plot histogram of emergence times
        if doplot and self.doemergence:
            fig, ax = plt.subplots()
            ax.hist(self.temergence)
            ax.set_xlabel("Time of emergence (abs. mag g $< %.1f$)" % mag_emergence)
            plt.savefig("plots/%s_timeemergence.png" % self.surveyname)
            
        
    # sample events assuming PCA representation
    def sample_events_PCA(self, **kwargs):

        self.nsim = kwargs['nsim']  # number of events
        rvs = kwargs['rvs'] # function which return variable given random number

        doplot = False
        if 'doplot' in kwargs.keys():
            doplot = bool(kwargs['doplot'])
        doload = False
        if 'doload' in kwargs.keys():
            doload = bool(kwargs['doload'])
        dosave = False
        if 'dosave' in kwargs.keys():
            dosave = bool(kwargs['dosave'])
        keepLCs = False
        if 'keepLCs' in kwargs.keys():
            keepLCs = bool(kwargs['keepLCs'])
        if doload:
            dosave = False
    
        # sample variables
        if not doload:

            # sample redshifts
            rands = np.random.random(size = self.nsim)
            self.logzs = np.log(self.random2z(rands))
            
            # sample explosion times
            tmin = min(self.obsplan.MJDs)
            tmax = max(self.obsplan.MJDs)
            rands = np.random.random(size = self.nsim)
            self.texps = tmin + 1. * rands * (tmax - tmin)
    
            # sample physical parameters
            for key in rvs.keys():
                rands = rvs[key](self.nsim)
                if key == 'logAv':
                    self.logAvs = rands
                if key == "alphas":
                    # use random numbers plus column of ones to account for constant (PCA0)
                    self.parsarray = np.column_stack((np.ones(self.nsim), rands))

        if doplot:
            fig, ax = plt.subplots(figsize = (20, 10))
    
        # load LCs and physical parameters for the given MJDs
        if doload:
            print("Load LCs, TBD")
        else:
            self.LCsamples = []
            self.parsamples = []
            
        # get LCs and plot them if necessary
        for i in range(self.nsim):

            # print status
            if not doload and self.obsplan.mode != 'maf':
                if np.mod(i, 10) == 0:
                    print("\r%i" % i, end = "")
    
            # sample light curves
            if not doload:
                
                # save only first list with light curve at given time (no reference values)
                self.LCsamples.append(self.LCs.evalmodel(1., self.texps[i], self.logzs[i], self.logAvs[i], self.parsarray[i])[0])
    
                # append physical parameters to array
                self.parsamples.append(np.hstack([self.logzs[i], self.texps[i], self.logAvs[i], self.parsarray[i]]))
    
            # plot first 1000 LCs
            if doplot and i < 1000: # avoid plotting more than 1000 LCs
                for band in self.LCs.uniquefilters:
                    maskband = self.LCs.maskband[band]
                    if i == 0:
                        ax.plot(self.LCs.mjd[maskband], self.obsplan.limmag[maskband], lw = 4, c = self.LCs.bandcolors[band])
                        ax.plot(self.LCs.mjd[maskband], self.obsplan.limmag[maskband] - 2.5 * np.log10(np.sqrt(2.)), lw = 4, c = self.LCs.bandcolors[band], ls = ':')
                    masklim = self.LCsamples[i][band] < self.obsplan.limmag[maskband] + 2
                    ax.plot(self.LCs.mjd[maskband][masklim], self.LCsamples[i][band][masklim], c = self.LCs.bandcolors[band], alpha = 0.1)
    
        # numpify and transpose physical parameters
        self.parsamples = np.array(self.parsamples).transpose()
    
        # invert axis
        if doplot:
            ax.set_ylim(ax.get_ylim()[::-1])
            plt.savefig("plots/%s_LCs2.png" % (self.surveyname, key))
    
        # save LCs and physical parameters
        if dosave:
            import pickle
    
            # LCs
            LCsfile = "%s/pickles/%s_%s_LCs_%i.pkl" % (os.environ["SURVEYSIM_PATH"], self.LCs.modelname, self.obsplan.planname, self.nsim)
            pickle.dump(self.LCsamples, open(LCsfile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)
    
            # physical parameters
            parsfile = "%s/pickles/%s_%s_params_%i.pkl" % (os.environ["SURVEYSIM_PATH"], self.LCs.modelname, self.obsplan.planname, self.nsim)
            pickle.dump(self.parsamples, open(parsfile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)

        # save plot
        if doplot:
            plt.savefig("plots/%s_%s.png" % (self.obsplan.planname, self.LCs.modelname))

        # keep LCs in memory
        if keepLCs:
            for isim in range(self.nsim):
                for band in self.LCsamples[isim].keys():
                    maskband = self.LCs.maskband[band]
                    masklim = self.LCsamples[isim][band] < self.obsplan.limmag[maskband]
                    if np.sum(masklim) > 0:
                        mindet, maxdet = min(self.LCs.mjd[maskband][masklim]), max(self.LCs.mjd[maskband][masklim])
                        masktime = (self.LCs.mjd[maskband] > mindet - 30) & (self.LCs.mjd[maskband] < maxdet + 30)
                        if np.sum(masktime) > 0:
                            lenmask = np.sum(masktime)
                            dfLCs = pd.DataFrame({"IDSN": [isim for i in range(lenmask)],\
                                           "MJD": self.LCs.mjd[maskband][masktime], \
                                           "limmag": self.obsplan.limmag[maskband][masktime], \
                                           "filter": [band for i in range(lenmask)], \
                                           "magSN": self.LCsamples[isim][band][masktime]})
                            dfpars = pd.DataFrame({"IDSN": [isim], \
                                                   "logz": [self.parsamples[0][isim]], \
                                                   "texp": [self.parsamples[1][isim]], \
                                                   "logAv": [self.parsamples[2][isim]], \
                                                   "alpha1": [self.parsamples[4][isim]], \
                                                   "alpha2": [self.parsamples[5][isim]]})  # check how to fix this for general number of parameters
                            if "simLCs" in locals():
                                simLCs = simLCs.append(dfLCs)
                                if not isim in list(simpars["IDSN"]):
                                    simpars = simpars.append(dfpars)
                            else:
                                simLCs = dfLCs
                                simpars = dfpars
            if "simLCs" in locals():
                return True, simLCs, simpars
            else:
                return False, None, None

            
    def do_efficiency(self, **kwargs):

        doplot = False
        if 'doplot' in kwargs.keys():
            doplot = kwargs['doplot']

        dosave = False
        if 'dosave' in kwargs.keys():
            dosave = kwargs['dosave']
            
        verbose = False
        if 'verbose' in kwargs.keys():
            verbose = kwargs['verbose']

        restrstring = ""
        check1stdetection = False
        if 'check1stdetection' in kwargs.keys():
            check1stdetection = bool(kwargs["check1stdetection"])
            # time of 1st detection in rest frame per object (after emergence)
            rftime1stdet = 1e99 * np.ones(len(self.LCsamples))
            restrstring = "%s_maxage%i" % (restrstring, self.maxrestframeage)

        checklastdetection = False
        if 'checklastdetection' in kwargs.keys():
            checklastdetection = bool(kwargs["checklastdetection"])
            # time of last detection in rest frame per object (after emergence)
            rftimelastdet = np.zeros(len(self.LCsamples))
            restrstring = "%s_minage%i" % (restrstring, self.minrestframeage)

        mindetections = 2
        if 'mindetections' in kwargs.keys():
            mindetections = int(kwargs["mindetections"])
            restrstring = "%s_mindet%i" % (restrstring, mindetections)

        print(restrstring)

        showLC = False
        if 'showLC' in kwargs.keys():
            showLC = bool(kwargs["showLC"])

        # number of detections per object per band
        matches = np.zeros((len(self.LCsamples), len(self.obsplan.uniquebands)))
        
        # brightest magnitude detected per object per band
        minmags = np.zeros((len(self.LCsamples), len(self.obsplan.uniquebands)))

        # check all light curves
        for idx, LCsample in enumerate(self.LCsamples):

            redshift = np.exp(self.parsamples[0, idx])
            texp = self.parsamples[1, idx]

            # loop among bands
            for idxb, band in enumerate(self.obsplan.uniquebands):

                maskband = self.LCs.maskband[band]

                dodiff = True
                if dodiff:
                    diffeffect = -2.5 * np.log10(np.sqrt(2.))
                else:
                    diffeffect = 0
                    
                masklim = LCsample[band] < self.obsplan.limmag[maskband] + diffeffect # assume worst case for difference imaging

                if check1stdetection or checklastdetection:
                    masklim2 = LCsample[band] < self.obsplan.limmag[maskband] + diffeffect + 1. # this is equivalent to 2 sigma

                matches[idx, idxb] = np.sum(masklim)

                #rftimedetections = (self.obsplan.MJDs[maskband][masklim] - (texp + self.temergence[idx])) / (1. + redshift)

                if matches[idx, idxb] > 0:
                    if check1stdetection or checklastdetection:
                        # best match to actual filters
                        rftimedetections = (self.obsplan.MJDs[maskband][masklim2] - texp) # measure time from explosion in observer time
                        if self.doemergence:
                            rftimedetections -= self.temergence[idx] # measure time from emergence instead

                    if doplot:
                        if matches[idx, idxb] > 0:
                            minmags[idx, idxb] = min(LCsample[band][masklim])
                    if check1stdetection:
                        rftime1stdet[idx] = min(rftime1stdet[idx], min(rftimedetections))
                    if checklastdetection:
                        rftimelastdet[idx] = max(rftimelastdet[idx], max(rftimedetections))


        ## show 10 random light curves
        #if showLC:
        #    # loop among bands
        #    for idxb, band in enumerate(self.obsplan.uniquebands):
            

        # count only detections with at least two detections [and with early detections if check1stdetection]
        detections = (np.sum(matches, axis = 1) >= mindetections)
        if check1stdetection:
            detections = np.array(detections & (rftime1stdet <= self.maxrestframeage))
        if checklastdetection:
            detections = np.array(detections & (rftimelastdet >= self.minrestframeage))

        # filter by time of first detection
        labels = np.concatenate([np.array(['logz', 'texp', 'logAv'], dtype = str), np.array(self.LCs.paramnames, dtype = str)])
        self.x_effs = {}
        self.y_effs = {}
        self.effs = {}
        
        for vallabel, valin in zip(labels, self.parsamples):
            if min(valin) == max(valin):
                continue
            if vallabel == 'logz':
                vallabel = 'z'
                valin = np.exp(valin)
            if vallabel == 'mdot':
                vallabel = 'log10mdot'
            if vallabel == 'logAv':
                vallabel = 'log10Av'
                valin = valin / np.log(10.)

            # values of detected SNe
            valout = np.array(valin[detections])

            # store this for later comparison
            if vallabel == 'log10mdot':
                log10mdot = np.array(valout)
                if self.doemergence:
                    # SBO times
                    tSBO = self.temergence[detections]
            if vallabel == 'z':
                z = np.array(valout)

            # sort values
            idxsort = np.argsort(valout)
            valout = valout[idxsort]

            # create bins for histogram
            nn = np.cumsum(np.ones(self.nsim))
            nin, bin_edgesin = np.histogram(valin, bins = 25)
            nout, bin_edgesout = np.histogram(valout, bins = bin_edgesin)
            self.x_effs[vallabel] = (bin_edgesin[1:] + bin_edgesin[:-1]) / 2.
            self.y_effs[vallabel] = 1. * nout / nin
            self.effs[vallabel] = interp1d(self.x_effs[vallabel], self.y_effs[vallabel], bounds_error = False, fill_value = 0)

            
            # save efficiencies
            if dosave:
                import pickle
                effsfile = "%s/pickles/%s_%s_LCs_%i_effs.pkl" % (os.environ["SURVEYSIM_PATH"], self.LCs.modelname, self.obsplan.planname, self.nsim)
                pickle.dump([self.x_effs, self.y_effs], open(effsfile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)

            # plot histograms
            if doplot:
                fig, ax = plt.subplots()
                #ax.plot(valin, range(len(valin)), label = 'Explosions')
                # plot scaled number of detections
                ax.plot(self.x_effs[vallabel], self.effs[vallabel](self.x_effs[vallabel]), linestyle = 'steps-mid', c = 'k')
                ax.set_ylim(0, ax.get_ylim()[1])
                ax.set_ylabel('Efficiency', fontsize = 14)
                ax.set_xlabel(vallabel, fontsize = 14)
                ax2 = ax.twinx()
                ax2.plot(valout, np.array(range(len(valout)), dtype = float) / len(valin) * self.cumtotalSNe[-1], label = 'Detections', c = 'r')
                ax2.set_ylabel('CDF', color='r', fontsize = 14)
                ax2.set_ylim(0, ax2.get_ylim()[1])
                ax2.tick_params('y', colors='r')
                if vallabel == 'log10mdot':
                    ax.set_xlabel(r'$\log_{10} \dot M\ [M_\odot yr^{-1}]$')
                elif  vallabel == 'beta':
                    ax.set_xlabel(r'$\beta$')
                elif vallabel == "texp":
                    ax.set_xlabel("Explosion time")
                plt.savefig("plots/%s%s_%s_efficiency.png" % (self.surveyname, restrstring, vallabel))
                
        if doplot:

            # mass loss rate vs emergence times
            if self.doemergence:
                fig, ax = plt.subplots()
                ax.scatter(log10mdot, tSBO)
                ax.set_xlabel("log10mdot")
                ax.set_ylabel("temergence - texp [days]")

            if "log10mdot" in self.LCs.paramnames:
                np.save("%s/pickles/%s_%s_zlog10mdot.npy" % (os.environ["SURVEYSIM_PATH"], self.LCs.modelname, self.obsplan.planname), [z, log10mdot])
                # mass loss rate vs redshift
                fig, ax = plt.subplots()
                ax.scatter(z, log10mdot, marker  = '.', alpha = 0.5)
                ax.set_xlabel("z")
                ax.set_ylabel("log10mdot")
                plt.savefig("plots/%s%s_log10mdot.png" % (self.surveyname, restrstring))


                H, xedges, yedges = np.histogram2d(z, log10mdot, range = [[0, self.maxz], [-8, -2]], bins = (12, 12))
                x, y = np.meshgrid((xedges[1:] + xedges[:-1]) / 2., (yedges[1:] + yedges[:-1]) / 2.)
                extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
                cset = ax.contour(x, y, H.transpose(), origin = 'lower')#, levels, origin = 'lower')
            #plt.clabel(cset, inline = 1, fontsize = 10, fmt = '%1.0i')
            #for c in cset.collections:
            #    c.set_linestyle('solid')
            
            # apparent magnitudes
            fig, ax = plt.subplots()
            for idxb, band in enumerate(self.obsplan.uniquebands):
                factor = 1. / np.sum(minmags[:, idxb] > 0) * np.sum(detections) / len(detections) * self.cumtotalSNe[-1]
                ax.plot(sorted(minmags[:, idxb][minmags[:, idxb] > 0]), range(np.sum(minmags[:, idxb] > 0)) * factor, color = self.LCs.bandcolors[band], label = band)
                ax.scatter([min(minmags[:, idxb][minmags[:, idxb] > 0]), max(minmags[:, idxb][minmags[:, idxb] > 0])], [0, (np.sum(minmags[:, idxb] > 0) - 1) * factor], color = self.LCs.bandcolors[band])
                #ax.hist(minmags[:, idxb][minmags[:, idxb] > 0], alpha = 0.5, label = band, color = self.LCs.bandcolors[band], cumulative = True)
            ax.legend(loc = 2)
            ax.set_yscale('log')
            ax.set_xlabel("min mag")
            ax.set_ylabel("CDF")
            plt.savefig("plots/%s%s_minmag.png" % (self.surveyname, restrstring))

            # absolute magnitudes
            fig, ax = plt.subplots()
            for idxb, band in enumerate(self.obsplan.uniquebands):
                factor = 1. / np.sum(minmags[:, idxb] > 0) * np.sum(detections) / len(detections) * self.cumtotalSNe[-1]
                mask = (minmags[:, idxb] > 0)
                Dms = np.array(list(map(lambda logz: self.Dmf(np.exp(logz)), self.parsamples[0, :])))
                ax.hist(minmags[:, idxb][mask] - Dms[mask], alpha = 0.5, label = band, weights = factor * np.ones(np.sum(mask)), color = self.LCs.bandcolors[band])
            ax.legend(loc = 2)    
            ax.set_xlabel("min abs mag")
            plt.savefig("plots/%s%s_minabsmag.png" % (self.surveyname, restrstring))
                

    def do_efficiency_PCA(self, **kwargs):

        doplot = False
        if 'doplot' in kwargs.keys():
            doplot = bool(kwargs['doplot'])

        dosave = False
        if 'dosave' in kwargs.keys():
            dosave = bool(kwargs['dosave'])
            
        verbose = False
        if 'verbose' in kwargs.keys():
            verbose = bool(kwargs['verbose'])

        check1stdetection = False
        if 'check1stdetection' in kwargs.keys():
            check1stdetection = bool(kwargs["check1stdetection"])
            # time of 1st detection in rest frame per object (after emergence)
            rftime1stdet = 1e99 * np.ones(len(self.LCsamples))

        checklastdetection = False
        if 'checklastdetection' in kwargs.keys():
            checklastdetection = bool(kwargs["checklastdetection"])
            # time of 1st detection in rest frame per object (after emergence)
            rftimelastdet = 1e99 * np.ones(len(self.LCsamples))

        mindetections = 2
        if 'mindetection' in kwargs.keys():
            mindetections = int(kwargs["mindetections"])

        # number of detections per object per band
        matches = np.zeros((len(self.LCsamples), len(self.obsplan.uniquebands)))

        if doplot:
            # brightest magnitude detected per object per band
            minmags = np.zeros((len(self.LCsamples), len(self.obsplan.uniquebands)))

        # check all light curves
        for idx, LCsample in enumerate(self.LCsamples):

            redshift = np.exp(self.parsamples[0, idx])
            texp = self.parsamples[1, idx]

            # loop among bands
            for idxb, band in enumerate(self.obsplan.uniquebands):

                maskband = self.LCs.maskband[band]
                
                masklim = LCsample[band] < self.obsplan.limmag[maskband] - 2.5 * np.log10(np.sqrt(2.)) # assume worst case for difference imaging

                if check1stdetection or checklastdetection:
                    masklim2 = LCsample[band] < self.obsplan.limmag[maskband] - 2.5 * np.log10(np.sqrt(2.)) + 1. # this is equivalent to 2 sigma
                
                matches[idx, idxb] = np.sum(masklim)
                
                if check1stdetection or checklastdetection:
                    # best match to actual filters
                    rftimedetections = (self.obsplan.MJDs[maskband][masklim2] - texp)

                if doplot:
                    if matches[idx, idxb] > 0:
                        minmags[idx, idxb] = min(LCsample[band][masklim])
                if check1stdetection:
                    if rftime1stdet[idx] == 0:
                        rftime1stdet[idx] = min(rftimedetections)
                    else:
                        rftime1stdet[idx] = min(rftime1stdet[idx], min(rftimedetections))
                if checklastdetection:
                    rftimelastdet[idx] = max(rftimelastdet[idx], max(rftimedetections))


        # count only detections with at least two detections [and with early detections if check1stdetection]
        detections = (np.sum(matches, axis = 1) >= mindetections)
        if check1stdetection:
            detections = np.array(detections & (rftime1stdet <= self.maxrestframeage))
        if checklastdetection:
            detections = np.array(detections & (rftimelastdet >= self.minrestframeage))

        # filter by time of first detection
        labels = np.concatenate([np.array(['logz', 'texp', 'logAv'], dtype = str), np.array(["alpha1", "alpha2"], dtype = str)])
        self.x_effs = {}
        self.y_effs = {}
        self.effs = {}
        
        for vallabel, valin in zip(labels, self.parsamples):
            if min(valin) == max(valin):
                continue
            if vallabel == 'logz':
                vallabel = 'z'
                valin = np.exp(valin)
            if vallabel == 'mdot':
                vallabel = 'log10mdot'
            if vallabel == 'logAv':
                vallabel = 'log10Av'
                valin = valin / np.log(10.)

            # values of detected SNe
            valout = np.array(valin[detections])

            # store this for later comparison
            if vallabel == 'log10mdot':
                log10mdot = np.array(valout)
                if self.doemergence:
                    # SBO times
                    tSBO = self.temergence[detections]
            if vallabel == 'z':
                z = np.array(valout)

            # sort values
            idxsort = np.argsort(valout)
            valout = valout[idxsort]

            # create bins for histogram
            nn = np.cumsum(np.ones(self.nsim))
            nin, bin_edgesin = np.histogram(valin, bins = 25)
            nout, bin_edgesout = np.histogram(valout, bins = bin_edgesin)
            self.x_effs[vallabel] = (bin_edgesin[1:] + bin_edgesin[:-1]) / 2.
            self.y_effs[vallabel] = 1. * nout / nin
            self.effs[vallabel] = interp1d(self.x_effs[vallabel], self.y_effs[vallabel], bounds_error = False, fill_value = 0)

            
            # save efficiencies
            if dosave:
                import pickle
                effsfile = "%s/pickles/%s_%s_LCs_%i_effs.pkl" % (os.environ["SURVEYSIM_PATH"], self.LCs.modelname, self.obsplan.planname, self.nsim)
                pickle.dump([self.x_effs, self.y_effs], open(effsfile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)

            # plot histograms
            if doplot:
                fig, ax = plt.subplots()
                #ax.plot(valin, range(len(valin)), label = 'Explosions')
                # plot scaled number of detections
                ax.plot(self.x_effs[vallabel], self.effs[vallabel](self.x_effs[vallabel]), linestyle = 'steps-mid', c = 'k')
                ax.set_ylim(0, ax.get_ylim()[1])
                ax.set_ylabel('Efficiency', fontsize = 14)
                ax.set_xlabel(vallabel, fontsize = 14)
                ax2 = ax.twinx()
                ax2.plot(valout, np.array(range(len(valout)), dtype = float) / len(valin) * self.cumtotalSNe[-1], label = 'Detections', c = 'r')
                ax2.set_ylabel('CDF', color='r', fontsize = 14)
                ax2.set_ylim(0, ax2.get_ylim()[1])
                ax2.tick_params('y', colors='r')
                if vallabel == 'log10mdot':
                    ax.set_xlabel(r'$\log_{10} \dot M\ [M_\odot yr^{-1}]$')
                    ax.set_ylabel("Efficiency")
                    plt.savefig("plots/log10mdot_efficiency.png")
                elif  vallabel == 'beta':
                    ax.set_xlabel(r'$\beta$')
                    ax.set_ylabel("Efficiency")
                    plt.savefig("plots/beta_efficiency.png")
                else:
                    plt.savefig("plots/%s_efficiency.png" % vallabel)

        if doplot:

            # apparent magnitudes
            fig, ax = plt.subplots()
            for idxb, band in enumerate(self.obsplan.uniquebands):
                ax.hist(minmags[:, idxb][minmags[:, idxb] > 0], alpha = 0.5, label = band, color = self.LCs.bandcolors[band])
            ax.legend(loc = 2)    
            ax.set_xlabel("min mag")

            # absolute magnitudes
            fig, ax = plt.subplots()
            for idxb, band in enumerate(self.obsplan.uniquebands):
                mask = (minmags[:, idxb] > 0)
                Dms = np.array(list(map(lambda logz: self.Dmf(np.exp(logz)), self.parsamples[0, :])))
                ax.hist(minmags[:, idxb][mask] - Dms[mask], alpha = 0.5, label = band, color = self.LCs.bandcolors[band])
            ax.legend(loc = 2)    
            ax.set_xlabel("min abs mag")


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
    nsim = int(sys.argv[3])

    # create observational plan
    if obsname == "SNLS":
        plan = obsplan(obsname = "CFHT-MegaCam", mode = 'file-cols', inputfile = "SNLS_bands.dat", nfields = 1, nepochspernight = 1, nightfraction = 0.045, nread = 5, doplot = True, doload = True, bandcolors = {'g': 'g', 'r': 'r', 'i': 'brown', 'z': 'k'})
    elif obsname == "KMTNet17B":
        #plan = obsplan(obsname = "KMTNet", mode = 'file-cols', inputfile = "KMTNet_17B.dat", nfields = 200, nepochspernight = 1, nightfraction = 1., nread = 1, doplot = True, bandcolors = {'g': 'g', 'r': 'r', 'i': 'brown', 'z': 'k'})
        #plan = obsplan(obsname = "KMTNet", mode = 'file-cols', inputfile = "KMTNet_17B_BVRI.dat", nfields = 67, nepochspernight = 2, nightfraction = 0.65, nread = 1, doplot = True, bandcolors = {'g': 'k', 'B': 'b', 'V': 'g', 'R': 'r', 'I': 'brown'})
        #plan = obsplan(obsname = "KMTNet", mode = 'file-cols', inputfile = "KMTNet_17B_VBRI.dat", nfields = 67, nepochspernight = 2, nightfraction = 0.65, nread = 1, doplot = True, bandcolors = {'g': 'k', 'B': 'b', 'V': 'g', 'R': 'r', 'I': 'brown'})
        plan = obsplan(obsname = "KMTNet", mode = 'file-cols', inputfile = "KMTNet_17B_RBVI.dat", nfields = 67, nepochspernight = 2, nightfraction = 0.65, nread = 1, doplot = True, bandcolors = {'g': 'k', 'B': 'b', 'V': 'g', 'R': 'r', 'I': 'brown'})
    elif obsname == "HiTS15A":
        plan = obsplan(obsname = "Blanco-DECam", mode = 'file-cols', inputfile = "HiTS15A.dat", nfields = 50, nepochspernight = 1., nightfraction = 1., nread = 1, doplot = True, bandcolors = {'g': 'g', 'r': 'r', 'i': 'brown'})        
    else:
        print("WARNING: undefined observatory")
        sys.exit()

    # load models
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
    parammetric = np.array([1., 1., 1e-6, 1., 10., 1.])
    paramlogscale = np.array([False, False, True, False, False, True], dtype = bool)
    LCs = LCz_Av_params(modelsdir = modelsdir, modelname = modelname, files = files, paramnames = paramnames, paramunits = paramunits, params = params, zs = zs, Avs = Avs, Rv = Rv, times = times)

    # do cosmology
    LCs.docosmo()

    # compute models in given bands
    if plan.obsname in ["Blanco-DECam"]:
        LCs.compute_models(bands = ['u', 'g', 'r', 'i', 'z'], load = True)#, save = True)#, 'r'])#, 'i', 'z'])
    elif plan.obsname in ["KMTNet"]:
        LCs.compute_models(bands = ['g', 'B', 'V', 'R', 'I'], load = True)#False, save = True)#, 'r'])#, 'i', 'z'])
    else:
        print("What bands to use?")
        sys.exit()
        
    # set metric
    LCs.setmetric(metric = parammetric, logscale = paramlogscale)
    
    # set observations
    if plan.obsname in ["Blanco-DECam", "CFHT-MegaCam"]:
        LCs.set_observations(mjd = plan.MJDs, flux = None, e_flux = None, filters = plan.bands, objname = None, plot = False, bandcolors = {'g': 'g', 'r': 'r', 'i': 'brown', 'z': 'k'})
    elif plan.obsname in ["KMTNet"]:
        LCs.set_observations(mjd = plan.MJDs, flux = None, e_flux = None, filters = plan.bands, objname = None, plot = False, bandcolors = {'g': 'gray', 'B': 'b', 'V': 'g', 'R': 'r', 'I': 'brown'})
    else:
        print("What bands to use?")
        sys.exit()

    # star formation
    SFH = SFHs(SFH = "MD14")
    knorm = 0.0091
    IIPfrac = 0.54
    efficiency = knorm * IIPfrac
    
    # maximum age of object at the start of the survey
    maxrestframeage = 3.
    
    # start survey
    newsurvey = survey_multimodel(obsplan = plan, SFH = SFH, efficiency = efficiency, LCs = LCs, maxrestframeage = maxrestframeage)

    # set maximum redshift
    #newsurvey.set_maxz(0.3)
    if obsname == "HiTS15A":
        newsurvey.set_maxz(0.6)
    
    # do cosmology with dense grid
    newsurvey.do_cosmology()

    # set distribution of physical parameters
    minMJD, maxMJD = min(newsurvey.obsplan.MJDs) - (20. + newsurvey.maxrestframeage) * (1. + max(newsurvey.zs)), max(newsurvey.obsplan.MJDs)
    rvs = {'texp': lambda nsim: uniform.rvs(loc = minMJD, scale = maxMJD - minMJD, size = nsim), \
           'logAv': lambda nsim: norm.rvs(loc = np.log(0.1), scale = 1., size = nsim), \
           #'mass': lambda nsim: norm.rvs(loc = 14, scale = 3, size = nsim), \
           'mass': lambda nsim: uniform.rvs(loc = 12., scale = 4., size = nsim), \
           'energy': lambda nsim: norm.rvs(loc = 1., scale = 1., size = nsim), \
           'log10mdot': lambda nsim: uniform.rvs(loc = -8, scale = 6, size = nsim), \
           'beta': lambda nsim: uniform.rvs(loc = 1., scale = 4., size = nsim)}
    bounds = {'texp': [minMJD, maxMJD], \
              'logAv': [np.log(1e-4), np.log(10.)], \
              'mass': [12, 16], \
              'energy': [0.5, 2.], \
              'log10mdot': [-8, -2], \
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
    #newsurvey.sample_events(nsim = nsim, doload = False, dosave = True, doplot = True, rvs = rvs, bounds = bounds, pars = pars)
    newsurvey.sample_events(nsim = nsim, doload = True, doplot = True, rvs = rvs, bounds = bounds, pars = pars)

    # measure detections and efficiency
    newsurvey.do_efficiency(doplot = True, verbose = False, check1stdetection = True)
    
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
    # plot light curves
    #newsurvey.plot_LCs(save = True)
    
    plt.show()
