import numpy as np
import re, sys, os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.widgets import Slider

from collections import defaultdict
import itertools
from scipy.optimize import minimize

import emcee
import corner

import time

from constants import *
from LCz import *
from LCz_Av import *

from LCz_Av_params import *

sys.path.append("../cos_calc")
from cos_calc import *

                
if __name__ == "__main__":

    print("Markov chain Monte Carlo model fitting...\n")
    
    dodes = True
    dohits = False

    #########################
    # Observational data
    #########################

    
    # DES SNe
    # -----------------------------

    if dodes:
        
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
    
        SNname = "DES15X2mku"
        SNname = "DES13C2jtx"
        SNname = "DES15S2eaq"
        SNname = "DES15X1lzp"
        
        for SN in SNe.keys():
    
            if SN != SNname:
                continue
            
            zcmb = zSNe[SN]
            fixz = True
        
            for band in SNe[SN].keys():
    
                mjd = SNe[SN][band]["MJD"]
                mag = SNe[SN][band]["mag"]
                e_mag = SNe[SN][band]["e_mag"]

                if SNname == "DES15X2mku":
                    mask = (mjd > 57300) & (mjd < 57500)
                elif SNname == "DES13C2jtx":
                    mask = (mjd > 56450) & (mjd < 56700)
                elif SNname == "DES15S2eaq":
                    mask = (mjd > 57200) & (mjd < 57500)
                elif SNname == "DES15X1lzp":
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

        mask = sn_filters == 'g'
        texp0 = sn_mjd[mask][np.argmax(np.diff(np.abs(sn_flux[mask])))]
        if SNname == "DES15X2mku":
            texp0 = 57325
        elif SNname == "DES13C2jtx":
            texp0 = 56550
        elif SNname == "DES15S2eaq":
            texp0 = 57275
        elif SNname == "DES15Xlzp":
            texp0 = 57320

    # HiTS SNe
    # -----------------------------------------------------------------

    elif dohits:
        SNname = "SNHiTS15aw"
        #SNname = "SNHiTS15K"
        SNname = "SNHiTS14B"
        (MJDs, MJDrefs, ADUs, e_ADUs, mags, e1_mags, e2_mags, sn_filters) \
            = np.loadtxt("/home/fforster/Work/HiTS/LCs/%s.txt" % SNname, usecols = (0, 1, 5, 6, 7, 8, 9, 10), dtype = str).transpose()
        sn_mjd = np.array(MJDs, dtype = float)
        sn_adu = np.array(ADUs, dtype = float)
        sn_e_adu = np.array(e_ADUs, dtype = float)
        sn_mag = np.array(mags, dtype = float)
        sn_flux = np.array(sn_adu)
        sn_e_flux = np.array(sn_e_adu)
        maskg = sn_filters == 'g'
        if np.sum(maskg) > 0:
            factorg = mag2flux(sn_mag[maskg][-1]) / sn_adu[maskg][-1]
            sn_flux[maskg] = sn_flux[maskg] * factorg
            sn_e_flux[maskg] = sn_e_flux[maskg] * factorg
        maskr = sn_filters == 'r'
        if np.sum(maskr) > 0:
            factorr = mag2flux(sn_mag[maskr][-1]) / sn_adu[maskr][-1]
            sn_flux[maskr] = sn_flux[maskr] * factorr
            sn_e_flux[maskr] = sn_e_flux[maskr] * factorr
        texp0 = 57077.

        mask = sn_filters == 'g'
        texp0 = sn_mjd[mask][np.argmax(np.diff(sn_flux[mask]))]
        fixz = False
        if SNname == "SNHiTS15aw":
            texp0 = 57074
            zcmb = 0.0663
            fixz = True
        if SNname == "SNHiTS15K":
            texp0 = 57067
            zcmb = 0.18
            fixz = False
        if SNname == "SNHiTS14B":
            texp0 = 56717
            zcmb = 0.3
            fixz = False

    else:
        print("Define observations...")
        sys.exit()

        
    # Theoretical  models
    # -------------------------------------------------------------

    modelsdir = "/home/fforster/Work/surveysim/models"
    modelname = "MoriyaWindAcc"
    data = np.genfromtxt("%s/%s/modellist.txt" % (modelsdir, modelname), dtype = str, usecols = (0, 1, 3, 5, 7, 9, 10, 11)).transpose()
    data[data == 'no'] = 0
    modelfile, modelmsun, modele51, modelmdot, modelrcsm, modelvwind0, modelvwindinf, modelbeta = data

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
    print(files)

    # Redshift, Avs and time
    nz = 30
    ntimes = 100
    nAvs = 10
    zs = np.logspace(-3, 0, nz)
    times = np.logspace(-3, 3, ntimes)
    Avs = np.logspace(-4, 1, nAvs)
    Rv = 3.25


    # ----------------------------
    # ------- MCMC fitter --------
    # ----------------------------

    
    # initialize MCMCfit model
    MCMC = LCz_Av_params(modelsdir = modelsdir, modelname = modelname, files = files, paramnames = ["mass", "energy", "mdot", "rcsm", "vwindinf", "beta"], paramunits = ["Msun", "B", "Msun/yr", "1e15 cm", "km/s", ""], params = params, zs = zs, Avs = Avs, Rv = Rv, times = times)

    # do cosmology
    LCs.docosmo()

    # compute models in given bands
    LCs.compute_models(bands = ['u', 'g', 'r', 'i', 'z'], load = True)#, save = True)#, 'r'])#, 'i', 'z'])
    
    # set metric
    LCs.setmetric(metric = np.array([1., 1., 1e-6, 1., 10., 1.]), logscale = np.array([False, False, True, False, False, True], dtype = bool))
        
    # set observations
    LCs.set_observations(mjd = sn_mjd, flux = sn_flux, e_flux = sn_e_flux, filters = sn_filters, objname = SNname, plot = False, bandcolors = {'g': 'g', 'r': 'r', 'i': 'brown', 'z': 'k'})
    
    # actual model
    #filename = files[np.argmin(map(lambda p: LCs.paramdist(par, p), params))]
    #h100, omega_m, omega_k, omega_lambda = Hnot / 100., OmegaM, 1. - (OmegaM + OmegaL), OmegaL
    #cosmo = cos_calc.fn_cos_calc(h100, omega_m, omega_k, omega_lambda, zcmb)
    #DL = cosmo[4] # Mpc
    #Dm = cosmo[5] # Mpc
    #for band in LCs.uniquefilters:
    #    SN = StellaModel(dir = "/home/fforster/Work/Model_LCs/models/yoon12msun", modelfile = filename, doplot = False)
    #    SN_Av = LCz_Av(LCz = SN, Av = np.atleast_1d(min(Avs)), Rv = Rv, zs = np.atleast_1d(zcmb), DL = np.atleast_1d(DL), Dm = np.atleast_1d(Dm), filtername = band, doplot = False)
    #    SN_Av.compute_mags()
    #    mags = SN_Av.magAvf[0][0](LCs.times)
    #    ax.plot(LCs.times + texp, scale * mag2flux(mags), label = "%s" % band, lw = 1, alpha = 0.8, c = bandcolors[band])
    #


    # -----------------------------------
    # ----- Initial guess ---------------
    # -----------------------------------
    
    
    # find best fit
    scale = 1.0
    texp = texp0
    logz = np.log(zcmb)
    logAv = np.log(0.1)#min(Avs))
    mass = 14.
    energy = 1. # foe
    mdot = 1e-5
    rcsm = 1. # 1e15
    vwindinf = 10.
    beta = 2.5
    parvals = np.array([scale, texp, logz, logAv, mass, energy, mdot, rcsm, vwindinf, beta])
    #parbounds = np.array([[0.1, 10.], [texp - 5, texp + 5], [np.log(1e-4), np.log(10.)], [np.log(1e-4), np.log(10.)], [12, 16], [0.5, 2.], [3e-5, 1e-2], [1., 1.], [10, 10], [1., 5.]])
    parbounds = np.array([[0.1, 10.], [texp - 5, texp + 5], [np.log(1e-4), np.log(10.)], [np.log(1e-4), np.log(10.)], [12, 16], [0.5, 2.], [1e-6, 1e-2], [1., 1.], [10, 10], [1., 5.]])
    parlabels = np.array(["scale", "texp", "logz", "logAv", "mass", "energy", "mdot", "rcsm", "vwindinf", "beta"])
    fixedvars = np.array([False,     False,  fixz,   False,   False,   False,    False,   True,   True,      False], dtype = bool)  # rcsm and vwinf should be True with current model grid
 
    # initialize with previous parameters
    theta0 = parvals[np.invert(fixedvars)]
    sol = LCs.findbest(theta0 = theta0, parbounds = parbounds, fixedvars = fixedvars, parvals = parvals, parlabels = parlabels, skip = True)
    
    # exit if not convergence
    if not sol.success:
        print(sol)
        print("WARNING: initial estimation does not converge")
        sys.exit()
    
    # recover variables
    LCs.parvals[np.invert(LCs.fixedvars)] = sol.x
    scale, texp, logz, logAv, mass, energy, mdot, rcsm, vwindinf, beta = LCs.parvals

    # check best solution
    print("Best fit parameters:", zip(parlabels, LCs.parvals))
    LCmag = LCs.evalmodel(scale, texp, logz, logAv, LCs.parvals[4:], True, False)
    fig, ax = plt.subplots(figsize = (12, 7))
    modelplot = {}
    texpplot = ax.axvline(texp, c = 'gray', alpha = 1)
    for band in LCs.uniquefilters:
        mask = LCs.maskband[band]
        ax.errorbar(LCs.mjd[mask], LCs.flux[mask], yerr = LCs.e_flux[mask], marker = 'o', c = LCs.bandcolors[band], lw = 0, elinewidth = 1)
        modelplot[band] = ax.plot(LCs.times + texp, scale * mag2flux(LCmag[band]), label = "%s" % band, c = LCs.bandcolors[band])
    #title = ax.set_title("scale: %5.3f, texp: %f, Av: %f, mass: %f, energy: %f, mdot: %3.1e, rcsm: %3.1f, beta: %f" % (scale, texp, np.exp(logAv), mass, energy, mdot, rcsm, beta), fontsize = 8)
    ax.legend(loc = 1, fontsize = 8, framealpha = 0.5)
    ax.set_xlim(min(texp, min(LCs.mjd)) - 1, max(LCs.mjd) + 50)

    dointeractive = False
    if dointeractive:
        # slider axes
        texp_slider_ax =   fig.add_axes([0.15, 0.985, 0.75, 0.015], axisbg='w')
        scale_slider_ax  = fig.add_axes([0.15, 0.97, 0.75, 0.015], axisbg='w')
        z_slider_ax =      fig.add_axes([0.15, 0.955, 0.75, 0.015], axisbg='w')
        av_slider_ax =     fig.add_axes([0.15, 0.94, 0.75, 0.015], axisbg='w')
        
        mass_slider_ax = fig.add_axes([0.15, 0.045, 0.75, 0.015], axisbg='w')
        energy_slider_ax = fig.add_axes([0.15, 0.03, 0.75, 0.015], axisbg='w')
        mdot_slider_ax = fig.add_axes([0.15, 0.015, 0.75, 0.015], axisbg='w')
        beta_slider_ax = fig.add_axes([0.15, 0.0, 0.75, 0.015], axisbg='w')
        
        # slider objects
        texp_slider = Slider(texp_slider_ax, 'texp', texp0 - 20, texp0 + 20, valinit=texp0, dragging = False)
        scale_slider = Slider(scale_slider_ax, 'log scale', -3., 1., valinit=1., dragging = False)
        z_slider = Slider(z_slider_ax, 'logz', LCs.parbounds[LCs.parlabels == 'logz'][0][0], LCs.parbounds[LCs.parlabels == 'logz'][0][1], valinit=LCs.parvals[LCs.parlabels == 'logz'][0], dragging = False)
        av_slider = Slider(av_slider_ax, 'logAv', LCs.parbounds[LCs.parlabels == 'logAv'][0][0], LCs.parbounds[LCs.parlabels == 'logAv'][0][1], valinit=LCs.parvals[LCs.parlabels == 'logAv'][0], dragging = False)
        
        mass_slider = Slider(mass_slider_ax, 'mass', LCs.parbounds[LCs.parlabels == 'mass'][0][0], LCs.parbounds[LCs.parlabels == 'mass'][0][1], valinit=LCs.parvals[LCs.parlabels == 'mass'][0], dragging = False)
        energy_slider = Slider(energy_slider_ax, 'energy', LCs.parbounds[LCs.parlabels == 'energy'][0][0], LCs.parbounds[LCs.parlabels == 'energy'][0][1], valinit=LCs.parvals[LCs.parlabels == 'energy'][0], dragging = False)
        mdot_slider = Slider(mdot_slider_ax, 'log mdot', np.log10(LCs.parbounds[LCs.parlabels == 'mdot'][0][0]), np.log10(LCs.parbounds[LCs.parlabels == 'mdot'][0][1]), valinit=np.log10(LCs.parvals[LCs.parlabels == 'mdot'][0]), dragging = False)
        beta_slider = Slider(beta_slider_ax, 'beta', LCs.parbounds[LCs.parlabels == 'beta'][0][0], LCs.parbounds[LCs.parlabels == 'beta'][0][1], valinit=LCs.parvals[LCs.parlabels == 'beta'][0], dragging = False)
        
        def slider_update(val):
        
            # update values
            scale = 10**scale_slider.val
            texp = texp_slider.val
            logz = z_slider.val
            logAv = av_slider.val
            LCs.parvals[LCs.parlabels == 'mass'] = mass_slider.val
            LCs.parvals[LCs.parlabels == 'mass'] = mass_slider.val
            LCs.parvals[LCs.parlabels == 'energy'] = energy_slider.val
            LCs.parvals[LCs.parlabels == 'mdot'] = 10**mdot_slider.val
            LCs.parvals[LCs.parlabels == 'beta'] = beta_slider.val
            # compute new model
            LCmag = LCs.evalmodel(scale, texp, logz, logAv, LCs.parvals[LCs.nvext:], True, False)
            texpplot.set_xdata(texp)
            fig.canvas.draw()
            # update plots
            for band in LCs.uniquefilters:
                modelplot[band][0].set_ydata(scale * mag2flux(LCmag[band]))
                modelplot[band][0].set_xdata(LCs.times + texp)
                ax.set_title("scale: %5.3f, texp: %f, Av: %f, mass: %f, energy: %f, mdot: %3.1e, rcsm: %3.1f, beta: %f" % (scale, texp, np.exp(logAv), mass, energy, mdot, rcsm, beta), fontsize = 8)

                fig.canvas.draw_idle()
                
            
        texp_slider.on_changed(slider_update)
        scale_slider.on_changed(slider_update)
        z_slider.on_changed(slider_update)
        av_slider.on_changed(slider_update)
        mass_slider.on_changed(slider_update)
        energy_slider.on_changed(slider_update)
        mdot_slider.on_changed(slider_update)
        beta_slider.on_changed(slider_update)
        
        
        plt.show()
        plt.savefig("plots/Bestfit_%s_%s.png" % (LCs.modelname, LCs.objname))
        
        sys.exit()
        
    dotest = False
    
    if  dotest:
        print("Testing interpolation\n\n")
        LCs.test_interpolation("mass")
        LCs.test_interpolation("energy")
        LCs.test_interpolation("beta")
        LCs.test_interpolation("mdot")
        sys.exit()


    # -------------------------
    # ---- MCMC ---------------
    # -------------------------

    
    # set prior distributions
    from scipy.stats import lognorm, norm, uniform  # leave this here, otherwise it fails!

    priors = np.array([lambda scale: norm.pdf(scale, loc = 1., scale = 0.1), \
                       lambda texp: norm.pdf(texp, loc = theta0[1], scale = 3.), \
                       lambda logz: norm.pdf(logz, loc = np.log(0.18), scale = 2), \
                       lambda logAv: norm.pdf(logAv, loc = np.log(0.01), scale = 0.5), \
                       lambda mass: norm.pdf(mass, loc = 14, scale = 3), \
                       lambda energy: norm.pdf(energy, loc = 1., scale = 1.), \
                       lambda mdot: lognorm.pdf(mdot / 1e-4, 1.), \
                       lambda rcsm: norm.pdf(rcsm, loc = 1., scale = 1.), \
                       None, \
                       lambda beta: lognorm.pdf(beta / 7., 1.)])
    
    #def scaleprior(scale):
    #    return norm.pdf(scale, loc = 1., scale = 0.1)
    #def texpprior(texp):
    #    return norm.pdf(texp, loc = theta0[1], scale = 3.)
    #def logzprior(logz):
    #    return norm.pdf(logz, loc = np.log(0.18), scale = 2)
    #def logAvprior(logAv):
    #    return norm.pdf(logAv, loc = np.log(0.01), scale = 0.5)
    #def massprior(mass):
    #    return norm.pdf(mass, loc = 14, scale = 3)
    #def energyprior(energy):
    #    return norm.pdf(energy, loc = 1., scale = 1.)
    #def mdotprior(mdot):
    #    return lognorm.pdf(mdot / 1e-4, 1.)
    #def rcsmprior(rcsm):
    #    return norm.pdf(rcsm, loc = 1., scale = 1.)
    #def betaprior(beta):
    #    return lognorm.pdf(beta / 7., 1.)
    #
    #priors = np.array([scaleprior, texpprior, logzprior, logAvprior, massprior, energyprior, mdotprior, rcsmprior, None, betaprior])
    
    LCs.set_priors(priors)
    
    # start MCMC
    LCs.doMCMC(bestfit = np.array(sol.x), nwalkers = 400, deltabestfit = 1e-5, nsteps = 1000, nburn = 100, parlabels = parlabels, load = False) 

    # plot results
    LCs.plotMCMC(nburn = 500, correctlogs = True)


