import numpy as np
import re, sys, os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from collections import defaultdict
import itertools
from scipy.optimize import minimize

import emcee
import corner
import pickle

import time

from constants import *
from LCz import *
from LCz_Av import *

sys.path.append("../cos_calc")
from cos_calc import *


# class to fit an  MCMC model to data given a grid of models
class MCMCfit(object):

    # initialize grid of models
    def __init__(self, **kwargs):

        # model instrinsic physical paramater grid
        self.paramnames = kwargs["paramnames"]
        self.paramunits = kwargs["paramunits"]
        self.params = kwargs["params"]
        self.modelsdir = kwargs["modelsdir"]
        self.modelname = kwargs["modelname"]
        self.files = kwargs["files"] # dictionary with filenames given by nested keys

        # model additional interpolation parameters
        self.zs = np.atleast_1d(kwargs["zs"]) # must be np.logspace(...)
        self.Avs = np.atleast_1d(kwargs["Avs"]) # must be np.logspace(...)
        self.Rv = kwargs["Rv"]
        self.times = np.atleast_1d(kwargs["times"])

        # values for easier interpolation
        self.nz = len(self.zs)
        self.logzs = np.log(self.zs)
        self.dlogz = self.logzs[-1] - self.logzs[-2]
        self.minlogz = min(self.logzs)
        self.nAv = len(self.Avs)
        self.logAvs = np.log(self.Avs)
        self.dlogAv = self.logAvs[-1] - self.logAvs[-2]
        self.minlogAv = min(self.logAvs)

        # number of model parameters
        self.nvar = len(self.paramnames)


    # do cosmology
    def docosmo(self):

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

    # compute models in the given bands
    def compute_models(self, **kwargs):

        bands = kwargs["bands"]
        dosave = False
        doload = False
        if "save" in kwargs.keys():
            dosave = kwargs["save"]
            doload = False
        if "load" in kwargs.keys():
            doload = kwargs["load"]
            dosave = False

        # save
        if doload:

            print "Loading models..."
            if not np.array_equal(np.load("allmags/%s_%s_Avs.npy" % (self.modelname, "".join(bands))), self.Avs):
                print "Avs array does not match loaded value, using saved values..."
                self.Avs = np.load("allmags/%s_%s_Avs.npy" % (self.modelname, "".join(bands)))
            if not np.array_equal(np.load("allmags/%s_%s_zs.npy" % (self.modelname, "".join(bands))), self.zs):
                print "zs array does not match loaded value, using saved values..."
                self.zs = np.load("allmags/%s_%s_zs.npy" % (self.modelname, "".join(bands)))
            if not np.array_equal(np.load("allmags/%s_%s_times.npy" % (self.modelname, "".join(bands))), self.times):
                print "WARNING: times array does not match loaded value, using saved values..."
                self.times = np.load("allmags/%s_%s_times.npy" % (self.modelname, "".join(bands)))
            if not np.array_equal(np.load("allmags/%s_%s_params.npy" % (self.modelname, "".join(bands))), self.params):
                print "params array does not match loaded value, using saved values..."
                self.params = np.load("allmags/%s_%s_params.npy" % (self.modelname, "".join(bands)))
            self.allmags = {}
            for band in bands:
                self.allmags[band] = np.load("allmags/%s_%s_allmags.npy" % (self.modelname, band))

            return

        nruns = len(self.files) * len(bands)
        print "Computing %i models..." % nruns
        print self.paramnames
        print self.paramunits

        start_time = time.time()
        
        self.allmags = {}

        aux = 0
        for filename, params in zip(self.files, self.params):

            print filename

            for band in bands:

                if "remtime" in locals():
                    print "File: %s, params: %s, band: %s, remaining time: %f min" % (filename, params, band, remtime / 60.)
                else:
                    print "File: %s, params: %s, band: %s" % (filename, params, band)
            
                SN = StellaModel(dir = "%s/%s" % (self.modelsdir, self.modelname), modelname = "%s-%s" % (self.modelname, filename), modelfile = filename, doplot = False)
                SN_Av = LCz_Av(LCz = SN, Av = self.Avs, Rv = self.Rv, zs = self.zs, DL = self.DL, Dm = self.Dm, filtername = band, doplot = False)
                SN_Av.compute_mags()
                mags = np.array([SN_Av.magAvf[iAv][iz](self.times) for iz, iAv in itertools.product(range(len(self.zs)), range(len(self.Avs)))]).reshape((len(self.zs), len(self.Avs), len(self.times)))
                mags = mags[np.newaxis, :]
                
                if band in self.allmags.keys():
                    self.allmags[band] = np.vstack([self.allmags[band], mags])
                else:
                    self.allmags[band] = mags

                aux = aux + 1
                ellapsed_time = time.time() - start_time

                remtime = ellapsed_time / aux * (nruns - aux) # sec
                    
        # save
        if dosave:
            np.save("allmags/%s_%s_Avs.npy" % (self.modelname, "".join(bands)), self.Avs)
            np.save("allmags/%s_%s_zs.npy" % (self.modelname, "".join(bands)), self.zs)
            np.save("allmags/%s_%s_times.npy" % (self.modelname, "".join(bands)), self.times)
            np.save("allmags/%s_%s_params.npy" % (self.modelname, "".join(bands)), self.params)
            for band in bands:
                np.save("allmags/%s_%s_allmags.npy" % (self.modelname, band), self.allmags[band])

            
    # set observational data to interpolate
    def set_observations(self, **kwargs):

        # observations
        self.mjd = kwargs["mjd"]
        self.flux = kwargs["flux"]
        self.e_flux = kwargs["e_flux"]
        self.filters = kwargs["filters"]
        self.objname = kwargs["objname"]
        self.bandcolors = kwargs["bandcolors"]

        self.uniquefilters = np.unique(self.filters)

        self.maskband = {}
        for band in self.uniquefilters:
            self.maskband[band] = self.filters == band

        doplot = False
        if "plot" in kwargs.keys():
            doplot = kwargs["plot"]

        if doplot:
            fig, ax = plt.subplots()
            for band in self.uniquefilters:
                mask = self.maskband[band]
                ax.errorbar(self.mjd[mask], self.flux[mask], yerr = self.e_flux[mask], lw = 0, elinewidth = 1, c = self.bandcolors[band], marker = 'o', label = "%s, %s band" % (name, band))

            ax.legend()
            ax.set_xlabel("mjd")
            ax.set_ylabel("Flux [erg/s/cm2]")
            plt.show()
            
    # defines the metric for interpolation
    def setmetric(self, **kwargs):

        self.metric = kwargs["metric"]
        self.logscale = kwargs["logscale"]

    # defines the distance between two parameter vectors
    def paramdist(self, p1, p2):

            self.notlogscale = np.invert(self.logscale)
            return np.sqrt(np.sum(((p1[self.notlogscale] - p2[self.notlogscale]) / self.metric[self.notlogscale])**2) + \
                           np.sum(np.log10((p1[self.logscale] / self.metric[self.logscale] + 1e-4) / (p2[self.logscale] / self.metric[self.logscale] + 1e-4))**2))

    # radial basis function
    def RBF(self, dist, scale):

        return np.exp(-(dist / scale)**2)
            
    # Function that interpolates the model at given parameter set, z, Av, texp and observed times
    def evalmodel(self, scale, texp, logz, logAv, pars, nice = False):

        if not hasattr(self, "paramdist"):

            print "Please set the metric function for parameter model interpolation"
            sys.exit()

        # find z and Av indices
        idxz = (logz - self.minlogz) / self.dlogz
        idxAv = (logAv - self.minlogAv) / self.dlogAv
        fz = idxz - int(idxz)
        fAv = idxAv - int(idxAv)

        # find best matching parameters and radial basis function weights
        distances = np.array(map(lambda p: self.paramdist(p, pars), self.params))
        npt = 1 # number of nearest neighbours to consider
        idxbest = np.argsort(distances)#[:npt] # best models to consider
        RBFs = self.RBF(distances[idxbest], 2. * np.average(distances[idxbest[:2]])) # set scale
        RBFs = RBFs / np.sum(RBFs) # normalize
        # remove tail of weight distribution
        cumsumRBFs = np.cumsum(RBFs)
        maskRBF = np.invert(np.array(cumsumRBFs > 0.9, dtype = bool))
        idxbest = idxbest[maskRBF]
        RBFs = RBFs[maskRBF]
        RBFs = RBFs / np.sum(RBFs)

        # light curve interpolation
        intLC = {}
        weights = {}
        for band in self.uniquefilters:
            intLC[band] = 0

        status = False
        for iz in range(2):
            if iz + int(idxz) < 0 or iz + int(idxz) >= self.nz:
                continue
            for iAv in range(2):
                if iAv + int(idxAv) < 0 or iAv + int(idxAv) >= self.nAv:
                    continue
                status = True

                for band in self.uniquefilters:
                    mask = self.maskband[band]

                    weights[band] = []
                    for idx, rbf in zip(idxbest, RBFs):

                        if nice:
                            LCint = np.interp(self.times, self.times, self.allmags[band][idx][iz + int(idxz)][iAv + int(idxAv)])
                        else:
                            LCint = np.interp(self.mjd[mask] - texp, self.times, self.allmags[band][idx][iz + int(idxz)][iAv + int(idxAv)])#, left = 30, right = 30)
                            
                        intLC[band] = intLC[band] + rbf * LCint \
                                      * (iz * fz + (1 - iz) * (1. - fz)) \
                                      * (iAv * fAv + (1 - iAv) * (1. - fAv))

        return intLC

    # chi2: sum of differences squared divided by the variance
    def chi2(self, allpars, fixed, vals):

        # separate fixed from variable parameters
        count = 0
        for idx, fix_val in enumerate(zip(fixed, vals)):
            fix, val = fix_val
            if not fix:
                vals[idx] = allpars[count]
                count = count + 1

        scale, texp, logz, logAv = vals[:4]
        pars = vals[4:]

        chi2 = 0
        modelmag = self.evalmodel(scale, texp, logz, logAv, pars)
        for band in self.uniquefilters:
            mask = self.maskband[band]
            chi2 = chi2 + np.sum((self.flux[mask] - mag2flux(modelmag[band]))**2 / self.e_flux[mask]**2)
            #if band == 'g':
            #    ax.plot(self.mjd[mask], scale * mag2flux(modelmag[band]), c = 'gray', alpha = 0.1)
        #print vals[np.invert(fixed)], chi2
        
        return chi2
        
    
    # find best fit to the data
    def findbest(self, **kwargs):

        theta0 = kwargs["theta0"]
        self.fixedvars = kwargs["fixedvars"]
        self.parvals = kwargs["parvals"]
        self.parbounds = kwargs["parbounds"]
        self.parlabels = kwargs["parlabels"]
        skip = False
        if "skip" in kwargs.keys():
            skip = bool(kwargs["skip"])

        print skip
        bounds = self.parbounds[np.invert(self.fixedvars)]

        method = 'L-BFGS-B'
        options = {'disp': False}

        if not skip:
            theta_lsq = minimize(self.chi2, theta0, args = (self.fixedvars, self.parvals), method= method, bounds = bounds, options = options)

        else:
            class sol(object):
                def __init__(self, success, x):
                    self.success = success
                    self.x = x
            theta_lsq = sol(True, theta0)
        
        return theta_lsq
    
    # set a priori distributions
    def set_priors(self, priors):

        print "Setting priors..."
        mask = np.invert(self.fixedvars)
        self.priors = priors[mask]
        bounds = self.parbounds[mask]
        labels = self.parlabels[mask]

        fig, ax = plt.subplots(nrows = len(bounds))
        for idx in range(len(bounds)):

            print "   ", labels[idx]

            xs = np.linspace(bounds[idx][0], bounds[idx][1], 1000)
            ax[idx].plot(xs, self.priors[idx](xs))
            ax[idx].set_title(labels[idx], fontsize = 6)

        plt.tight_layout()
        plt.savefig("plots/priors.png")
                

    # log likelihood
    def lnlike(self, theta):
        
        if not np.isfinite(np.sum(theta)):
            return -np.inf
        
        self.parvals[np.invert(self.fixedvars)] = theta

        scale, texp, logz, logAv = self.parvals[:4]
        pars = self.parvals[4:]

        loglike = 0
        modelmag = self.evalmodel(scale, texp, logz, logAv, pars)
        for band in self.uniquefilters:
            mask = self.maskband[band]
            loglike = loglike - 0.5 * np.sum((self.flux[mask] - mag2flux(modelmag[band]))**2 / self.e_flux[mask]**2)
        
        return loglike

    # prior likelihood of the parameters theta
    def lnprior(self, theta):

        if not np.isfinite(np.sum(theta)):
            return -np.inf
        
        bounds = self.parbounds[np.invert(self.fixedvars)]
        lnprior = 0
        for idx in range(len(theta)):
            if theta[idx] < bounds[idx][0] or theta[idx] > bounds[idx][1]:
                return -np.inf
            else:
                lnprior = lnprior + np.log(self.priors[idx](theta[idx]))

        return lnprior

    # lnlike plus lnprior
    def lnprob(self, theta):
        
        if not np.isfinite(np.sum(theta)):
            return -np.inf

        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf

        lnprob = lp + self.lnlike(theta)
        if not np.isfinite(lnprob):
            return -np.inf

        return lnprob
    
    
    # find MCMC confidence intervals for the model parameters
    def doMCMC(self, **kwargs):

        print "Estimating a posterior parameter distrution using Monte Carlo Markov Chain"
        
        self.bestfit = kwargs["bestfit"]
        self.nsteps = kwargs["nsteps"]
        self.nwalkers = kwargs["nwalkers"]
        nburn = kwargs["nburn"]
        deltabestfit = kwargs["deltabestfit"]
        doload = False
        if "load" in kwargs.keys():
            doload = bool(kwargs["load"])
            
        # select not fixed vars
        self.labels = self.parlabels[np.invert(self.fixedvars)]
        self.fitlabels = "-".join(self.labels)
        
        # initial position
        result = {}
        result["x"] = kwargs["bestfit"]
        
        # initialize random walkers
        self.ndim = len(result["x"])
        walkerscale = np.maximum(np.ones(self.ndim), (np.array(self.parbounds[np.invert(self.fixedvars), 1:] - self.parbounds[np.invert(self.fixedvars), :-1])).flatten())
        print "walkerscale:", walkerscale
        pos = [result["x"] + deltabestfit * np.abs(walkerscale) * np.random.randn(self.ndim) for i in range(self.nwalkers)]

        # initilize emcee sampler
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob)#, threads = 2);

        # filename containing sample chain
        chainname = "samples/chain_%s_%s_%s.dat" % (self.modelname, self.objname, self.fitlabels)
        
        # load samples
        if doload:
            print "Loading previous MCMC chain..."
            self.chain = np.loadtxt(chainname).transpose()
            self.nsteps = int(np.max(self.chain[0]) + 1)
            self.nwalkers = int(np.max(self.chain[1]) + 1)
            self.ndim = int(np.shape(self.chain)[0] - 2)
            print "nsteps: %i, nwalkers: %i, ndim: %i" % (self.nsteps, self.nwalkers, self.ndim)
            print "Original shape:", np.shape(self.chain)
            self.chain = np.reshape(self.chain[2:], (self.ndim, self.nsteps, self.nwalkers))
            self.chain = np.swapaxes(self.chain, 1, 2) # test
            self.chain = np.swapaxes(self.chain, 0, 1)
            self.chain = np.swapaxes(self.chain, 1, 2)
            print "New shape:", np.shape(self.chain)
        
        else:
            f = open(chainname, "w")
            f.write("# %s\n" % self.labels)
            f.close()
            for idx, result in enumerate(sampler.sample(pos, iterations=self.nsteps)):
                position = result[0]
                f = open(chainname, "a")
                for k in range(position.shape[0]):
                    f.write("%i %i %s\n" % (idx, k, " ".join(map(lambda p: str(p), position[k]))))
            f.close()
            self.chain = sampler.chain

            self.acceptance_fraction = np.mean(sampler.acceptance_fraction)
            print "Ensemble Sampler acceptance fraction: ", self.acceptance_fraction

                
    # find MCMC confidence intervals for the model parameters
    def plotMCMC(self, **kwargs):

        nburn = kwargs["nburn"]
        correctlogs = False
        if "correctlogs" in kwargs.keys():
            correctlogs = bool(kwargs["correctlogs"])
        correctedlogs = np.zeros(self.ndim, dtype = bool)

        if correctlogs:
            print "Correcting logarithmic variables..."
            for idx, lab in enumerate(self.labels):
                print idx, lab
                if lab[:3] == 'log':
                    print lab, "->", lab[3:]
                    self.chain[:, :, idx] = np.exp(self.chain[:, :, idx])
                    self.labels[idx] = self.labels[idx][3:]
                    correctedlogs[idx] = True

        # plot parameter evolution
        print "Plotting chain evolution..."
        fig, ax = plt.subplots(nrows = self.ndim, sharex = True, figsize = (16, 10))
        for j in range(self.ndim):
            ax[j].set_ylabel(self.labels[j])
            for i in range(self.nwalkers):
                ax[j].plot(self.chain[i, :, j], alpha = 0.2)
        plt.savefig("plots/MCMC_%s_%s_%s_evol.png" % (self.modelname, self.objname, self.fitlabels))

        # extract samples and convert to normal units
        samples = self.chain[:, nburn:, :].reshape((-1, self.ndim))
        samplescorner = np.array(samples)

        # do corner plot
        print "Doing corner plot..."
        fig = corner.corner(samplescorner, labels = self.labels, truths = self.bestfit)
        plt.savefig("plots/MCMC_%s_%s_%s_corner.png" % (self.modelname, self.objname, self.fitlabels))

        
        # show sample
        print "Plotting model sample"
        fig, ax = plt.subplots(figsize = (14, 7))

        for idx, band in enumerate(self.uniquefilters):
            mask = self.maskband[band]
            ax.errorbar(self.mjd[mask], self.flux[mask], yerr = self.e_flux[mask], marker = 'o', alpha = 0.5, lw = 0, elinewidth = 1, c = self.bandcolors[band], label = "%s" % band)


        # simulations
        nselection = 100
        idxselection = np.random.choice(np.array(range(self.nsteps)[nburn:]), size = nselection, replace = True)

        if correctlogs and np.sum(correctedlogs) > 0:
            samples[:, correctedlogs] = np.log(samples[:, correctedlogs])
        
        for idxsel, i in enumerate(idxselection):

            # recover variables
            self.parvals[np.invert(self.fixedvars)] = samples[i]

            # check best solution
            scale, texp, logz, logAv = self.parvals[:4]
            pars = self.parvals[4:]
            print "Parameters:", scale, texp, logz, logAv, pars

            LCmag = self.evalmodel(scale, texp, logz, logAv, pars, True)

            for idx, band in enumerate(self.uniquefilters):
                #mask = self.maskband[band]
                ax.axvline(texp, c = 'gray', alpha = 0.05)
                #ax.plot(self.mjd[mask], mag2flux(LCmag[band]), label = "%s" % band, c = self.bandcolors[band], alpha = 0.05)
                ax.plot(self.times + texp, mag2flux(LCmag[band]), c = self.bandcolors[band], alpha = 0.05)
            
        ax.set_ylabel(r"f$_{\nu}$ [erg/s/cm$^2$/Hz]", fontsize = 14)
        ax.set_xlabel("MJD [days]", fontsize = 14)
        ax.set_title("%s %s" % (self.modelname, self.objname))
        ax.set_xlim(min(min(self.mjd), min(samples[:, 1])) - 1, max(self.mjd) + 1)
        ax.legend(loc = 2, fontsize = 10)

        plt.savefig("plots/MCMC_%s_%s_%s_models.png" % (self.modelname, self.objname, self.fitlabels))

        
        
if __name__ == "__main__":

    print "Test run...\n"

    # load DES models
    DESdir = "/home/fforster/Work/DES"
    dirs = os.listdir(DESdir)
    SNe = defaultdict(list)
    zSNe = {}

    # read data
    for SN in dirs:
        
        if SN[:3] == 'DES' and SN[-3:] != 'txt':
            
            SNe[SN] = defaultdict(list)

            # read redshift
            info = open("%s/%s/%s.info" % (DESdir, SN, SN), 'r')
            zcmb = float(re.findall("Zcmb:\s(.*)\n", info.read())[0])
            zSNe[SN] = float(zcmb)
            name = SN

            # read photometry
            for f in os.listdir("%s/%s" % (DESdir, SN)):

                # extract band
                band = f[-1]

                # loop among allowed bands
                if band in ['u', 'g', 'r', 'i', 'z']:
                    
                    SNe[SN][band] = defaultdict(list)

                    # extract data
                    mjd, mag, e_mag = np.loadtxt("%s/%s/%s.out_DES%s" % (DESdir, SN, SN, band)).transpose()
                    SNe[SN][band]["MJD"] = mjd
                    SNe[SN][band]["mag"] = mag
                    SNe[SN][band]["e_mag"] = e_mag

    # plot SNe
    SNname = "DES15X2mku"
    for SN in SNe.keys():

        if SN != SNname:
            continue
        
        zcmb = zSNe[SN]
    
        for band in SNe[SN].keys():

            mjd = SNe[SN][band]["MJD"]
            mag = SNe[SN][band]["mag"]
            e_mag = SNe[SN][band]["e_mag"]
            
            mask = (mjd > 57300) & (mjd < 57500)
            mjd = mjd[mask]
            mag = mag[mask]
            e_mag = e_mag[mask]

            flux = mag2flux(mag)
            e_flux = mag2flux(mag + e_mag) - flux
            filters = np.array(map(lambda i: band, mjd))
            
            if "sn_mjd" not in locals():
                sn_mjd = np.array(mjd, dtype = float)
                sn_flux = flux
                sn_e_flux = e_flux
                sn_filters = filters
            else:
                sn_mjd = np.hstack([sn_mjd, mjd])
                sn_flux = np.hstack([sn_flux, flux])
                sn_e_flux = np.hstack([sn_e_flux, e_flux])
                sn_filters = np.hstack([sn_filters, filters])


    # Theoretical  models

    modelname = "yoon12msun"
    modelsdir = "/home/fforster/Work/surveysim/models"
    data = np.loadtxt("%s/%s/modellist.txt" % (modelsdir, modelname), dtype = str, usecols = (0, 1, 3, 5, 7)).transpose()
    data[data == 'no'] = 0
    modelfile, modelmsun, modele51, modelmdot, modelrcsm = data
    
    modelmsun = np.array(modelmsun, dtype = float)
    modele51 = np.array(modele51, dtype = float)
    modelmdot = np.array(modelmdot, dtype = float)
    modelrcsm = np.array(modelrcsm, dtype = float) / 1e15

    params = np.vstack([modelmsun, modele51, modelmdot, modelrcsm]).transpose()
    files = np.array(map(lambda name: "%s.fr" % name, modelfile))

    # Redshift, Avs and time
    nz = 30
    ntimes = 100
    nAvs = 10
    zs = np.logspace(-3, 0, nz)
    times = np.logspace(-3, 3, ntimes)
    Avs = np.logspace(-4, 1, nAvs)
    Rv = 3.1

    # initialize MCMCfit model
    MCMC = MCMCfit(modelsdir = modelsdir, modelname = modelname, files = files, paramnames = ["Mass", "E51", "Mdot", "Rcsm"], paramunits = ["Msun", "B", "Msun/yr", "1e15 cm"], params = params, zs = zs, Avs = Avs, Rv = Rv, times = times)

    # do cosmology
    MCMC.docosmo()

    # compute models in given bands
    MCMC.compute_models(bands = ['u', 'g', 'r', 'i', 'z'], load = True)#save = True)#, 'r'])#, 'i', 'z'])

    # set metric
    MCMC.setmetric(metric = np.array([10., 1e51, 1e-3, 1.]), logscale = np.array([False, False, True, True], dtype = bool))
        
    # set observations
    MCMC.set_observations(mjd = sn_mjd, flux = sn_flux, e_flux = sn_e_flux, filters = sn_filters, objname = SNname, plot = False, bandcolors = {'g': 'g', 'r': 'r', 'i': 'brown', 'z': 'k'})

    # start plotting
    fig, ax = plt.subplots(figsize = (12, 7))

    for band in MCMC.uniquefilters:
        mask = MCMC.maskband[band]
        ax.errorbar(MCMC.mjd[mask], MCMC.flux[mask], yerr = MCMC.e_flux[mask], marker = 'o', c = MCMC.bandcolors[band], lw = 0, elinewidth = 1)

    # actual model
    #filename = files[np.argmin(map(lambda p: MCMC.paramdist(par, p), params))]
    #h100, omega_m, omega_k, omega_lambda = Hnot / 100., OmegaM, 1. - (OmegaM + OmegaL), OmegaL
    #cosmo = cos_calc.fn_cos_calc(h100, omega_m, omega_k, omega_lambda, zcmb)
    #DL = cosmo[4] # Mpc
    #Dm = cosmo[5] # Mpc
    #for band in MCMC.uniquefilters:
    #    SN = StellaModel(dir = "/home/fforster/Work/Model_LCs/models/yoon12msun", modelfile = filename, doplot = False)
    #    SN_Av = LCz_Av(LCz = SN, Av = np.atleast_1d(min(Avs)), Rv = Rv, zs = np.atleast_1d(zcmb), DL = np.atleast_1d(DL), Dm = np.atleast_1d(Dm), filtername = band, doplot = False)
    #    SN_Av.compute_mags()
    #    mags = SN_Av.magAvf[0][0](MCMC.times)
    #    ax.plot(MCMC.times + texp, scale * mag2flux(mags), label = "%s" % band, lw = 1, alpha = 0.8, c = bandcolors[band])

    # find best fit
    scale = 0.8
    texp = sn_mjd[np.argmax(np.abs(sn_flux[1:] - sn_flux[:-1]))]# + 14.
    logz = np.log(zcmb)
    logAv = np.log(min(Avs))
    Msun = 12
    foe = 1e51
    mdot = 1e-3
    rcsm = 0.3
    parvals = np.array([scale, texp, logz, logAv, Msun, foe, mdot, rcsm])
    parbounds = np.array([[0.1, 10.], [texp - 10, texp + 10], [np.log(1e-3), np.log(1.)], [np.log(min(Avs)), np.log(1.)], [12, 12], [1e51, 1e51], [1e-12, 1e-2], [0.3, 3]])
    parlabels = np.array(["scale", "texp", "logz", "logAv", "Msun", "foe", "mdot", "rcsm"])
    fixedvars = np.array([False, False, True, False, True, True, False, False], dtype = bool)
    theta0 = parvals[np.invert(fixedvars)]
    
    sol = MCMC.findbest(theta0 = theta0, parbounds = parbounds, fixedvars = fixedvars, parvals = parvals)
    if not sol.success:
        print "WARNING: initial estimation does not converge"
        sys.exit()
    
    # recover variables
    aux = 0
    for idx, fix in enumerate(fixedvars):
        if not fix:
            val = sol.x[aux]
            aux = aux + 1
        else:
            val = parvals[idx]
        if idx == 0:
            scale = val
        elif idx == 1:
            texp = val
        elif idx == 2:
            logz = val
        elif idx == 3:
            logAv = val
        elif idx == 4:
            Msun = val
        elif idx == 5:
            foe = val
        elif idx == 6:
            mdot = val
        elif idx == 7:
            rcsm = val
    
    # check best solution
    pars = np.array([Msun, foe, mdot, rcsm])
    print "Best fit parameters:", scale, texp, logz, logAv, pars
    LCmag = MCMC.evalmodel(scale, texp, logz, logAv, pars)

    ax.axvline(texp, c = 'gray')
    for band in MCMC.uniquefilters:
        mask = MCMC.maskband[band]
        ax.scatter(MCMC.mjd[mask], scale * mag2flux(LCmag[band]), label = "%s" % band, lw = 0, c = MCMC.bandcolors[band], marker = 'o', alpha = 0.3, s = 200)
        
    ax.set_title("scale: %5.3f, texp: %f, Av: %f, mdot: %3.1e, rcsm: %3.1f" % (scale, texp, np.exp(logAv), mdot, rcsm), fontsize = 8)
    ax.legend(loc = 1, fontsize = 8, framealpha = 0.5)
    ax.set_xlim(min(MCMC.mjd), max(MCMC.mjd) + 100)
    plt.savefig("plots/Bestfit_%s_%s_%s.png" % (MCMC.modelname, MCMC.objname, MCMC.fitlabels))

    sys.exit()

    # set a priori distributions
    from scipy.stats import lognorm, norm, uniform  # leave this here
    priors = np.array([lambda scale: lognorm.pdf(scale, 1.), \
                       lambda texp: norm.pdf(texp, loc = theta0[1], scale = 5), \
                       None, \
                       lambda logAv: norm.pdf(logAv, loc = np.log(1e-2), scale = 3), \
                       None, \
                       None, \
                       lambda mdot: lognorm.pdf(mdot / 1e-4, 1.), \
                       lambda rcsm: norm.pdf(rcsm, loc = (3 + 0.3) / 2., scale = (3 - 0.3) / 2)])
    MCMC.set_priors(priors)

    solfit = np.array(sol.x)
    print solfit
    print MCMC.lnlike(solfit, fixedvars, parvals)
    print MCMC.lnprior(solfit, fixedvars, parbounds)
    print MCMC.lnprob(solfit, fixedvars, parvals, parbounds)

    # start MCMC
    MCMC.doMCMC(bestfit = solfit, fixedvars = fixedvars, parvals = parvals, parbounds = parbounds, nwalkers = 2, deltabestfit = 1e-4, nsteps = 500, nburn = 100, parlabels = parlabels) 

