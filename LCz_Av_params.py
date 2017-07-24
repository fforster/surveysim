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
class LCz_Av_params(object):

    # initialize grid of models
    def __init__(self, **kwargs):

        # model instrinsic physical paramater grid
        self.paramnames = np.array(kwargs["paramnames"], dtype = str)
        self.paramunits = np.array(kwargs["paramunits"], dtype = str)
        self.params = np.array(kwargs["params"])
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

        # check if mdot is in the variable names
        print(self.paramnames)
        self.maskmdot = np.array([self.paramnames == 'mdots'], dtype = bool)
        self.maskmdotvars = np.array([(self.paramnames == 'rcsm') | (self.paramnames == 'vwindinf') | (self.paramnames == 'beta')], dtype = bool)

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

        # start computing or loading specific models
        nruns = len(self.files) * len(bands)

        if not doload:
            print("Computing %i models..." % nruns)
            print(self.paramnames)
            print(self.paramunits)

        start_time = time.time()
        
        self.allmags = {}

        aux = 0
        for filename, params in zip(self.files, self.params):

            if not doload:
                print(filename)

            for band in bands:

                if "remtime" in locals() and not doload:
                    print("File: %s, params: %s, band: %s, remaining time: %f min" % (filename, params, band, remtime / 60.))
                else:
                    if not doload:
                        print("File: %s, params: %s, band: %s" % (filename, params, band))


                npydir = "npy/%s/%s" % (self.modelname, band)
                if doload:
                    mags = np.load("%s/%s.npy" % (npydir, filename))
                else:
                    npyfile = "%s/%s.npy" % (npydir, filename)
                    if not os.path.exists(npyfile):
                        SN = StellaModel(dir = "%s/%s" % (self.modelsdir, self.modelname), modelname = "%s-%s" % (self.modelname, filename), modelfile = filename, doplot = False)
                        SN_Av = LCz_Av(LCz = SN, Av = self.Avs, Rv = self.Rv, zs = self.zs, DL = self.DL, Dm = self.Dm, filtername = band, doplot = False)
                        SN_Av.compute_mags()
                        mags = np.array([SN_Av.magAvf[iAv][iz](self.times) for iz, iAv in itertools.product(range(len(self.zs)), range(len(self.Avs)))]).reshape((len(self.zs), len(self.Avs), len(self.times)))
                        if save:
                            if not os.path.exists(npydir):
                                os.makedirs(npydir)
                            np.save(npyfile, mags)
                    else:
                        mags = np.load("%s/%s.npy" % (npydir, filename))

                mags = mags[np.newaxis, :]
                if band in self.allmags.keys():
                    self.allmags[band] = np.vstack([self.allmags[band], mags])
                else:
                    self.allmags[band] = mags

                if not doload:
                    aux = aux + 1
                    ellapsed_time = time.time() - start_time
                    remtime = ellapsed_time / aux * (nruns - aux) # sec
                    
            
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
            
    # defines a metric for interpolation
    def setmetric(self, **kwargs):

        self.metric = kwargs["metric"]
        self.logscale = kwargs["logscale"]

    # distance between two parameter vectors given the metric
    def paramdist(self, p1, p2):

        # distance between variables
        dist = np.abs(p1 - p2) + 1e-2 * self.metric

        # distance between some variables should be zero if mdot is zero in one of the two vectors
        if np.sum(self.maskmdot) > 0:
            if p1[self.maskmdot] == 0 or p2[self.maskmdot] == 0:
                dist[self.maskmdotvars] = 0
                
        return np.product(dist)
        
    # Function that interpolates the model at given parameter set, z, Av, texp and observed times
    def evalmodel(self, scale, texp, logz, logAv, pars, nice = False, closest = False, verbose = False):

        if not hasattr(self, "paramdist"):
            print("Please set the metric function for parameter model interpolation")
            sys.exit()

        # find z and Av indices
        idxz = (logz - self.minlogz) / self.dlogz
        idxAv = (logAv - self.minlogAv) / self.dlogAv
        fz = idxz - int(idxz)
        fAv = idxAv - int(idxAv)

        if verbose:
            print("   Evalmodel", zip(self.paramnames, pars))

        # find closest values for all variables
        parsearch = {}
        for idx, var in enumerate(self.paramnames):
            # exact match for this variable
            if min(np.abs(pars[idx] - self.params[:, idx])) < 1e-9 :
                parsearch[var] = pars[idx]
            else:
                if min(self.params[:, idx]) > pars[idx]:
                    parsearch[var] = min(self.params[:, idx])
                elif max(self.params[:, idx]) < pars[idx]:
                    parsearch[var] = max(self.params[:, idx])
                else:
                    parsearch[var] = [max(self.params[self.params[:, idx] < pars[idx], idx]), min(self.params[self.params[:, idx] > pars[idx], idx])]

        if verbose:
            print(parsearch)
        
        if verbose:
            print("   parsearch:", parsearch)
                            
        # use only models which contain closest values
        maskclose = np.ones_like(self.params[:, 0], dtype = bool)
        for idx, var in enumerate(self.paramnames):

            if verbose:
                print("   ", parsearch[var])
            if size(parsearch[var]) == 1:
                nomatch = np.abs(self.params[:, idx] - parsearch[var]) > 1e-9
                if var != "mdot":
                    maskclose[nomatch] = False
                else:
                    maskclose[nomatch] = False
            else:
                nomatch0 = np.abs(self.params[:, idx] - parsearch[var][0]) > 1e-9
                nomatch1 = np.abs(self.params[:, idx] - parsearch[var][1]) > 1e-9
                if var != "mdot":
                    maskclose[nomatch0 & nomatch1] = False
                else:
                    maskclose[nomatch0 & nomatch1] = False

            #TEST
            #print(np.sum(maskclose))
            
        # check if interpolation is possible
        if np.sum(maskclose) == 0:
            print("Cannot interpolate", pars)
            print(parsearch)
            sys.exit()

        # indices
        idxbest = np.array(range(len(self.files)))[maskclose]
        if verbose:
            print("   ", self.files[idxbest])

        #TEST
        #self.idxbest = idxbest
            
        # compute distances from close model parameters
        distances = np.array(map(lambda p: self.paramdist(p, pars), self.params[idxbest]))
        
        if closest:
            idxbest = np.array([idxbest[np.argmin(distances)]])
            distances = np.array([min(distances)])
        
        # compute weights
        weights = 1. / (distances + 1e-20)
        weights = weights / np.sum(weights)
        
        if verbose:
            print(weights)

        #TEST
        #self.distances = distances
        
        # light curve interpolation
        intLC = {}
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

                    for idx, ww in zip(idxbest, weights):
                        if nice:
                            LCint = np.interp(self.times, self.times, self.allmags[band][idx][iz + int(idxz)][iAv + int(idxAv)])
                        else:
                            LCint = np.interp(self.mjd[mask] - texp, self.times, self.allmags[band][idx][iz + int(idxz)][iAv + int(idxAv)])#, left = 30, right = 30)
                            
                        intLC[band] = intLC[band] + ww * LCint \
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

        scale, texp, logz, logAv = vals[:self.nvext]
        pars = vals[self.nvext:]

        chi2 = 0
        modelmag = self.evalmodel(scale, texp, logz, logAv, pars)
        for band in self.uniquefilters:
            mask = self.maskband[band]
            chi2 = chi2 + np.sum((self.flux[mask] - mag2flux(modelmag[band]))**2 / self.e_flux[mask]**2)
            #if band == 'g':
            #    ax.plot(self.mjd[mask], scale * mag2flux(modelmag[band]), c = 'gray', alpha = 0.1)
        #print(vals[np.invert(fixed)], chi2)
        
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
            if skip:
                print("Skipping initial solution iteration")
        bounds = self.parbounds[np.invert(self.fixedvars)]

        # number of extrinsic variables, at the start of the variables array (scale, texp, z, Av)
        self.nvext = len(self.parlabels) - len(self.logscale)

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

    # test interpolation
    def test_interpolation(self, var):
        
        # save initial values and get extrinsic value
        startvars = np.array(self.parvals)
        scale, texp, logz, logAv = self.parvals[:self.nvext]
        
        # find index for given variable
        for idx, param in enumerate(self.parlabels):
            if var == param:
                idxvar = idx
                break
        print("\n\nTesting interpolation on %s variable... (%s, %i)" % (self.parlabels[idxvar], var, idxvar))


        # initialize plots and color scale
        import matplotlib.colors as colors
        fig, ax = plt.subplots(figsize = (16, 10), nrows = len(self.uniquefilters), sharex = True)
        jet = cm = plt.get_cmap('jet') 
        nplot = 100
        l1 = self.parbounds[idxvar, 0]
        l2 = self.parbounds[idxvar, 1]
        if self.logscale[idxvar - self.nvext]: # var is logarithmic
            vals = np.logspace(np.log10(l1), np.log10(l2), nplot)
            #TEST
            #vals = [1.1e-4, 2e-4, 2.9e-4]
            cNorm  = colors.Normalize(vmin = np.log10(l1), vmax = np.log10(l2))
            scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = jet)
        else: # var is linear
            vals = np.linspace(l1, l2, nplot)
            cNorm  = colors.Normalize(vmin = l1, vmax = l2)
            scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = jet)

        # vary test variable
        for val in vals:

            # value
            print("%s: %s" % (self.parlabels[idxvar], val))
            self.parvals[idxvar] = val

            # color
            if self.logscale[idxvar - self.nvext]:
                colorVal = scalarMap.to_rgba(np.log10(val))
            else:
                colorVal = scalarMap.to_rgba(val)

            # light curve model evaluation (interpolation happens here)
            LCmag = self.evalmodel(scale, texp, logz, logAv, self.parvals[self.nvext:], True, False) # use dense time and interpolate models
            #LCmagmodel = self.evalmodel(scale, texp, logz, logAv, self.parvals[self.nvext:], True, True) # use dense time and use closest model

            #print(self.parvals[self.nvext:])
            #print(self.idxbest)
            #print(self.params[self.idxbest])
            #print(self.distances)
            
            # loop among bands
            for idxf, band in enumerate(self.uniquefilters):
                ax[idxf].set_ylabel(band)
                ax[idxf].plot(self.times + texp, mag2flux(LCmag[band]), c = colorVal)
                #ax[idxf].plot(self.times + texp, mag2flux(LCmagmodel[band]), alpha = 0.2, lw = 4, c = colorVal)
                ax[idxf].axvline(texp, c = 'gray')

            #TEST
            #for idxb in self.idxbest:
            #    LCmagmodel = self.evalmodel(scale, texp, logz, logAv, self.params[idxb], True, True) # use dense time and use closest model
            #    for idxf, band in enumerate(self.uniquefilters):
            #        colorVal = scalarMap.to_rgba(self.params[idxb, 0])
            #        ax[idxf].plot(self.times + texp, mag2flux(LCmagmodel[band]), alpha = 0.2, lw = 4, c = colorVal)
                
                

        # axis and save
        ax[0].set_xlim(min(texp, min(self.mjd)) - 1, max(self.mjd) + 100)
        ax[0].set_title(" ".join(map(lambda x, y: "%s: %e" % (x, y), self.parlabels, startvars)), fontsize = 6)
        plt.savefig("plots/interpolation/interpolation_%s_%s_%s.png" % (self.modelname, self.objname, self.parlabels[idxvar]))

        # recover original values
        self.parvals = np.array(startvars)
        
    # set a priori distributions
    def set_priors(self, priors):

        print("Setting priors...")
        mask = np.invert(self.fixedvars)
        self.priors = priors[mask]
        bounds = self.parbounds[mask]
        labels = self.parlabels[mask]

        fig, ax = plt.subplots(nrows = len(bounds))
        for idx in range(len(bounds)):

            print("   ", labels[idx])

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

        scale, texp, logz, logAv = self.parvals[:self.nvext]
        pars = self.parvals[self.nvext:]

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

        print("Estimating a posterior parameter distrution using Monte Carlo Markov Chain")
        
        self.bestfit = kwargs["bestfit"]
        self.nsteps = kwargs["nsteps"]
        self.nwalkers = kwargs["nwalkers"]
        nburn = kwargs["nburn"]
        deltabestfit = kwargs["deltabestfit"]
        doload = False
        if "load" in kwargs.keys():
            doload = bool(kwargs["load"])
            
        # select vars which are not fixed
        self.labels = self.parlabels[np.invert(self.fixedvars)]
        self.fitlabels = "-".join(self.labels)
        
        # initial position
        result = {}
        result["x"] = kwargs["bestfit"]
        
        # initialize random walkers
        self.ndim = len(result["x"])
        walkerscale = np.maximum(np.ones(self.ndim), (np.array(self.parbounds[np.invert(self.fixedvars), 1:] - self.parbounds[np.invert(self.fixedvars), :-1])).flatten())
        print("walkerscale:", walkerscale)
        pos = [result["x"] + deltabestfit * np.abs(walkerscale) * np.random.randn(self.ndim) for i in range(self.nwalkers)]

        # initilize emcee sampler
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob)

        # filename containing sample chain
        chainname = "samples/chain_%s_%s_%s.dat" % (self.modelname, self.objname, self.fitlabels)
        
        # load samples
        if doload:
            print("Loading previous MCMC chain...")
            self.chain = np.loadtxt(chainname).transpose()
            self.nsteps = int(np.max(self.chain[0]) + 1)
            self.nwalkers = int(np.max(self.chain[1]) + 1)
            self.ndim = int(np.shape(self.chain)[0] - 2)
            print("nsteps: %i, nwalkers: %i, ndim: %i" % (self.nsteps, self.nwalkers, self.ndim))
            print("Original shape:", np.shape(self.chain))
            self.chain = np.reshape(self.chain[2:], (self.ndim, self.nsteps, self.nwalkers))
            self.chain = np.swapaxes(self.chain, 1, 2) # test
            self.chain = np.swapaxes(self.chain, 0, 1)
            self.chain = np.swapaxes(self.chain, 1, 2)
            print("New shape:", np.shape(self.chain))
        
        else:
            f = open(chainname, "w")
            f.write("# %s\n" % self.labels)
            for idx, result in enumerate(sampler.sample(pos, iterations = self.nsteps)):
                position = result[0]
                for k in range(position.shape[0]):
                    if np.mod(idx, 2) == 0:
                        print("%4i %4i %s" % (idx, k, " ".join(map(lambda p: str(p), position[k]))))
                    f.write("%i %i %s\n" % (idx, k, " ".join(map(lambda p: str(p), position[k]))))
            f.close()
                
            self.chain = sampler.chain
        
            self.acceptance_fraction = np.mean(sampler.acceptance_fraction)
            print("Ensemble Sampler acceptance fraction: ", self.acceptance_fraction)

                
    # find MCMC confidence intervals for the model parameters
    def plotMCMC(self, **kwargs):

        nburn = kwargs["nburn"]
        correctlogs = False
        if "correctlogs" in kwargs.keys():
            correctlogs = bool(kwargs["correctlogs"]) 
        correctmdot = False
        if "correctmdot" in kwargs.keys():
            correctmdot = bool(kwargs["correctmdot"])
        correctedlogs = np.zeros(self.ndim, dtype = bool)

        if correctlogs:
            print("Correcting logarithmic variables...")
            for idx, lab in enumerate(self.labels):
                print(idx, lab)
                if lab[:3] == 'log':
                    print(lab, "->", lab[3:])
                    self.chain[:, :, idx] = np.exp(self.chain[:, :, idx])
                    self.labels[idx] = self.labels[idx][3:]
                    correctedlogs[idx] = True

        correctmdot = True
        if correctmdot:
            for idx, lab in enumerate(self.labels):
                if lab == 'mdot':
                    idxmdot = idx
                    print(lab, "->", "log10mdot")
                    self.chain[:, :, idx] = np.log10(self.chain[:, :, idx])
                    self.labels[idx] = "log10mdot"

        # plot parameter evolution
        print("Plotting chain evolution...")
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
        print("Doing corner plot...")
        fig = corner.corner(samplescorner, labels = self.labels, truths = self.bestfit)
        plt.savefig("plots/MCMC_%s_%s_%s_corner.png" % (self.modelname, self.objname, self.fitlabels))

        
        # show sample
        print("Plotting model sample")
        fig, ax = plt.subplots(figsize = (14, 7))

        for idx, band in enumerate(self.uniquefilters):
            mask = self.maskband[band]
            ax.errorbar(self.mjd[mask], self.flux[mask], yerr = self.e_flux[mask], marker = 'o', alpha = 0.5, lw = 0, elinewidth = 1, c = self.bandcolors[band], label = "%s" % band)


        # simulations
        nselection = 100
        idxselection = np.random.choice(np.array(range(self.nsteps)[nburn:]), size = nselection, replace = True)

        if correctlogs and np.sum(correctedlogs) > 0:
            samples[:, correctedlogs] = np.log(samples[:, correctedlogs])
        if correctmdot:
            samples[:, idxmdot] = 10**(samples[:, idxmdot])
        
        for idxsel, i in enumerate(idxselection):

            # recover variables
            self.parvals[np.invert(self.fixedvars)] = samples[i]

            # check best solution
            scale, texp, logz, logAv = self.parvals[:self.nvext]
            pars = self.parvals[self.nvext:]
            print("Parameters:", scale, texp, logz, logAv, pars)

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

        
        
#if __name__ == "__main__":
#
#    print("Test run...\n")
#
#    # load DES models
#    DESdir = "/home/fforster/Work/DES"
#    dirs = os.listdir(DESdir)
#    SNe = defaultdict(list)
#    zSNe = {}
#
#    # read data
#    for SN in dirs:
#        
#        if SN[:3] == 'DES' and SN[-3:] != 'txt':
#            
#            SNe[SN] = defaultdict(list)
#
#            # read redshift
#            info = open("%s/%s/%s.info" % (DESdir, SN, SN), 'r')
#            zcmb = float(re.findall("Zcmb:\s(.*)\n", info.read())[0])
#            zSNe[SN] = float(zcmb)
#            name = SN
#
#            # read photometry
#            for f in os.listdir("%s/%s" % (DESdir, SN)):
#
#                # extract band
#                band = f[-1]
#
#                # loop among allowed bands
#                if band in ['u', 'g', 'r', 'i', 'z']:
#                    
#                    SNe[SN][band] = defaultdict(list)
#
#                    # extract data
#                    mjd, mag, e_mag = np.loadtxt("%s/%s/%s.out_DES%s" % (DESdir, SN, SN, band)).transpose()
#                    SNe[SN][band]["MJD"] = mjd
#                    SNe[SN][band]["mag"] = mag
#                    SNe[SN][band]["e_mag"] = e_mag
#
#    # plot SNe
#    SNname = "DES15X2mku"
#    for SN in SNe.keys():
#
#        if SN != SNname:
#            continue
#        
#        zcmb = zSNe[SN]
#    
#        for band in SNe[SN].keys():
#
#            mjd = SNe[SN][band]["MJD"]
#            mag = SNe[SN][band]["mag"]
#            e_mag = SNe[SN][band]["e_mag"]
#            
#            mask = (mjd > 57300) & (mjd < 57500)
#            mjd = mjd[mask]
#            mag = mag[mask]
#            e_mag = e_mag[mask]
#
#            flux = mag2flux(mag)
#            e_flux = mag2flux(mag + e_mag) - flux
#            filters = np.array(map(lambda i: band, mjd))
#            
#            if "sn_mjd" not in locals():
#                sn_mjd = np.array(mjd, dtype = float)
#                sn_flux = flux
#                sn_e_flux = e_flux
#                sn_filters = filters
#            else:
#                sn_mjd = np.hstack([sn_mjd, mjd])
#                sn_flux = np.hstack([sn_flux, flux])
#                sn_e_flux = np.hstack([sn_e_flux, e_flux])
#                sn_filters = np.hstack([sn_filters, filters])
#
#
#    # Theoretical  models
#
#    modelname = "yoon12msun"
#    modelsdir = "/home/fforster/Work/surveysim/models"
#    data = np.loadtxt("%s/%s/modellist.txt" % (modelsdir, modelname), dtype = str, usecols = (0, 1, 3, 5, 7)).transpose()
#    data[data == 'no'] = 0
#    modelfile, modelmsun, modele51, modelmdot, modelrcsm = data
#    
#    modelmsun = np.array(modelmsun, dtype = float)
#    modele51 = np.array(modele51, dtype = float)
#    modelmdot = np.array(modelmdot, dtype = float)
#    modelrcsm = np.array(modelrcsm, dtype = float) / 1e15
#
#    params = np.vstack([modelmsun, modele51, modelmdot, modelrcsm]).transpose()
#    files = np.array(map(lambda name: "%s.fr" % name, modelfile))
#
#    # Redshift, Avs and time
#    nz = 30
#    ntimes = 100
#    nAvs = 10
#    zs = np.logspace(-3, 0, nz)
#    times = np.logspace(-3, 3, ntimes)
#    Avs = np.logspace(-4, 1, nAvs)
#    Rv = 3.1
#
#    # initialize MCMCfit model
#    MCMC = MCMCfit(modelsdir = modelsdir, modelname = modelname, files = files, paramnames = ["Mass", "E51", "Mdot", "Rcsm"], paramunits = ["Msun", "B", "Msun/yr", "1e15 cm"], params = params, zs = zs, Avs = Avs, Rv = Rv, times = times)
#
#    # do cosmology
#    MCMC.docosmo()
#
#    # compute models in given bands
#    MCMC.compute_models(bands = ['u', 'g', 'r', 'i', 'z'], load = True)#save = True)#, 'r'])#, 'i', 'z'])
#
#    # set metric
#    MCMC.setmetric(metric = np.array([10., 1e51, 1e-3, 1.]), logscale = np.array([False, False, True, True], dtype = bool))
#        
#    # set observations
#    MCMC.set_observations(mjd = sn_mjd, flux = sn_flux, e_flux = sn_e_flux, filters = sn_filters, objname = SNname, plot = False, bandcolors = {'g': 'g', 'r': 'r', 'i': 'brown', 'z': 'k'})
#
#    # start plotting
#    fig, ax = plt.subplots(figsize = (12, 7))
#
#    for band in MCMC.uniquefilters:
#        mask = MCMC.maskband[band]
#        ax.errorbar(MCMC.mjd[mask], MCMC.flux[mask], yerr = MCMC.e_flux[mask], marker = 'o', c = MCMC.bandcolors[band], lw = 0, elinewidth = 1)
#
#    # actual model
#    #filename = files[np.argmin(map(lambda p: MCMC.paramdist(par, p), params))]
#    #h100, omega_m, omega_k, omega_lambda = Hnot / 100., OmegaM, 1. - (OmegaM + OmegaL), OmegaL
#    #cosmo = cos_calc.fn_cos_calc(h100, omega_m, omega_k, omega_lambda, zcmb)
#    #DL = cosmo[4] # Mpc
#    #Dm = cosmo[5] # Mpc
#    #for band in MCMC.uniquefilters:
#    #    SN = StellaModel(dir = "/home/fforster/Work/Model_LCs/models/yoon12msun", modelfile = filename, doplot = False)
#    #    SN_Av = LCz_Av(LCz = SN, Av = np.atleast_1d(min(Avs)), Rv = Rv, zs = np.atleast_1d(zcmb), DL = np.atleast_1d(DL), Dm = np.atleast_1d(Dm), filtername = band, doplot = False)
#    #    SN_Av.compute_mags()
#    #    mags = SN_Av.magAvf[0][0](MCMC.times)
#    #    ax.plot(MCMC.times + texp, scale * mag2flux(mags), label = "%s" % band, lw = 1, alpha = 0.8, c = bandcolors[band])
#
#    # find best fit
#    scale = 0.8
#    texp = sn_mjd[np.argmax(np.abs(sn_flux[1:] - sn_flux[:-1]))]# + 14.
#    logz = np.log(zcmb)
#    logAv = np.log(min(Avs))
#    Msun = 12
#    foe = 1e51
#    mdot = 1e-3
#    rcsm = 0.3
#    parvals = np.array([scale, texp, logz, logAv, Msun, foe, mdot, rcsm])
#    parbounds = np.array([[0.1, 10.], [texp - 10, texp + 10], [np.log(1e-3), np.log(1.)], [np.log(min(Avs)), np.log(1.)], [12, 12], [1e51, 1e51], [1e-12, 1e-2], [0.3, 3]])
#    parlabels = np.array(["scale", "texp", "logz", "logAv", "Msun", "foe", "mdot", "rcsm"])
#    fixedvars = np.array([False, False, True, False, False, False, False, False], dtype = bool)
#    theta0 = parvals[np.invert(fixedvars)]
#
#    sol = MCMC.findbest(theta0 = theta0, parbounds = parbounds, fixedvars = fixedvars, parvals = parvals)
#    if not sol.success:
#        print("WARNING: initial estimation does not converge")
#        sys.exit()
#    
#    # recover variables
#    aux = 0
#    for idx, fix in enumerate(fixedvars):
#        if not fix:
#            val = sol.x[aux]
#            aux = aux + 1
#        else:
#            val = parvals[idx]
#        if idx == 0:
#            scale = val
#        elif idx == 1:
#            texp = val
#        elif idx == 2:
#            logz = val
#        elif idx == 3:
#            logAv = val
#        elif idx == 4:
#            Msun = val
#        elif idx == 5:
#            foe = val
#        elif idx == 6:
#            mdot = val
#        elif idx == 7:
#            rcsm = val
#    
#    # check best solution
#    pars = np.array([Msun, foe, mdot, rcsm])
#    print("Best fit parameters:", scale, texp, logz, logAv, pars)
#    LCmag = MCMC.evalmodel(scale, texp, logz, logAv, pars)
#
#    ax.axvline(texp, c = 'gray')
#    for band in MCMC.uniquefilters:
#        mask = MCMC.maskband[band]
#        ax.scatter(MCMC.mjd[mask], scale * mag2flux(LCmag[band]), label = "%s" % band, lw = 0, c = MCMC.bandcolors[band], marker = 'o', alpha = 0.3, s = 200)
#        
#    ax.set_title("scale: %5.3f, texp: %f, Av: %f, mdot: %3.1e, rcsm: %3.1f" % (scale, texp, np.exp(logAv), mdot, rcsm), fontsize = 8)
#    ax.legend(loc = 1, fontsize = 8, framealpha = 0.5)
#    ax.set_xlim(min(MCMC.mjd), max(MCMC.mjd) + 100)
#    plt.savefig("plots/Bestfit_%s_%s_%s.png" % (MCMC.modelname, MCMC.objname, MCMC.fitlabels))
#
#    sys.exit()
#
#    # set a priori distributions
#    from scipy.stats import lognorm, norm, uniform  # leave this here
#    priors = np.array([lambda scale: lognorm.pdf(scale, 1.), \
#                       lambda texp: norm.pdf(texp, loc = theta0[1], scale = 5), \
#                       None, \
#                       lambda logAv: norm.pdf(logAv, loc = np.log(1e-2), scale = 3), \
#                       None, \
#                       None, \
#                       lambda mdot: lognorm.pdf(mdot / 1e-4, 1.), \
#                       lambda rcsm: norm.pdf(rcsm, loc = (3 + 0.3) / 2., scale = (3 - 0.3) / 2)])
#    MCMC.set_priors(priors)
#
#    solfit = np.array(sol.x)
#    print(solfit)
#    print(MCMC.lnlike(solfit, fixedvars, parvals))
#    print(MCMC.lnprior(solfit, fixedvars, parbounds))
#    print(MCMC.lnprob(solfit, fixedvars, parvals, parbounds))
#
#    # start MCMC
#    MCMC.doMCMC(bestfit = solfit, fixedvars = fixedvars, parvals = parvals, parbounds = parbounds, nwalkers = 2, deltabestfit = 1e-4, nsteps = 500, nburn = 100, parlabels = parlabels) 
#
