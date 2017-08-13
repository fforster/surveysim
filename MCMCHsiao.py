import numpy as np
import re, sys, os, getopt
leftraru = False
if os.getcwd() == "/home/fforster":
    leftraru = True
    import matplotlib # uncomment for using in leftraru
    matplotlib.use('Agg') # uncomment for using in leftraru
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pickle

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

from readSNdata import *

if __name__ == "__main__":

    # default values
    dointeractive = False
    dohits = False
    dodes = False
    verbose = False
    overwrite = False
    dotest = False
    loadMCMC = False
    diffLC = False
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "p:s:n:w:b:dioltvh", ["project=", "supernova=", "nsteps=", "walkers=", "burnin=", "diffLC", "interactive", "overwrite", "loadMCMC", "test", "verbose", "help"])
    except getopt.GetoptError:
        print 'python MCMCwindacc.py --help'
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print("Markov chain Monte Carlo fitting using Moriya wind acceleration models.")
            print("e.g. python ./MCMCHsiao.py --project DES --supernova DES15E2avs --interactive --verbose (interactively choose starting points)")
            print("e.g. python ./MCMCHsiao.py --project DES --supernova DES15E2avs --interactive --overwrite --verbose (interactively choose starting points, overwrite previously defined values)")
            print("e.g. python ./MCMCHsiao.py --project DES --supernova DES15E2avs --nsteps 1000 --walkers 400 --burnin 500 (run MCMC chain with 1000 steps, 400 walkers, burnin of 500)")
            print("e.g. python ./MCMCHsiao.py --project DES --supernova DES15E2avs --nsteps 1000 --walkers 400 --burnin 800 --loadMCMC --verbose (load MCMC chain for plotting, can choose new burnin value)")
        elif opt in ('-p', '--project'):
            project = arg
            print ("Project %s" % project)
        elif opt in ('-s', '--supernova'):
            SNname = arg
            print("Supernova %s" % SNname)
        elif opt in ('-n', '--nsteps'):
            nsteps = int(arg)
        elif opt in ('-w', '--walkers'):
            nwalkers = int(arg)
        elif opt in ('-b', '--burnin'):
            burnin = int(arg)
        elif opt in ('-o', '--overwrite'):
            overwrite = True
        elif opt in ('-l', '--loadMCMC'):
            loadMCMC = True
        elif opt in ('-d', '--diffLC'):
            diffLC = True
        elif opt in ('i', '--interactive'):
            dointeractive = True
        elif opt in ('t', '--test'):
            dotest = True
        elif opt in ('-v', '--verbose'):
            verbose = True

    if dointeractive:
        from matplotlib.widgets import Slider

    if 'SNname' not in locals():
        print("Need to define supernova name")
        sys.exit()
    
    print("Markov chain Monte Carlo model fitting...\n")
    

    # Observational data
    # -------------------------------------------------------------
    
    sn_mjd, sn_mjdref, sn_flux, sn_e_flux, sn_filters, fixz, zcmb, texp0 = readSNdata(project, SNname)

    
    # Theoretical  models
    # -------------------------------------------------------------

    modelsdir = "models"
    modelname = "Hsiao"
    modelfile = "snflux_1a.dat"
    files = np.array([modelfile], dtype = str)
    params = np.array([1.])[:, np.newaxis]

    # if a previous interactive estimation of the parameters existed
    par0 = {}
    if not dointeractive:
        if os.path.exists("initial_pars/%s/%s.pkl" % (modelname, SNname)):
            par0 = pickle.load(open("initial_pars/%s/%s.pkl" % (modelname, SNname), 'rb'))
            print par0
            if 'MJDmin' in par0.keys() and 'MJDmax' in par0.keys():
                mask = (sn_mjd > par0['MJDmin']) & (sn_mjd < par0['MJDmax'])
                sn_mjd = sn_mjd[mask]
                sn_flux = sn_flux[mask]
                sn_e_flux = sn_e_flux[mask]
                sn_filters = sn_filters[mask]
    else:
        if not overwrite and os.path.exists("initial_pars/%s/%s.pkl" % (modelname, SNname)):
            print "Exiting, initial parameter estimation already exists", dotest 
            sys.exit()
    
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
    LCs = LCz_Av_params(modelsdir = modelsdir, modelname = modelname, files = files, paramnames = ['stretch'], paramunits = [''], params = params, dostretch = True, zs = zs, Avs = Avs, Rv = Rv, times = times)

    # do cosmology
    LCs.docosmo()

    # compute models in given bands
    LCs.compute_models(bands = ['u', 'g', 'r', 'i', 'z'], load = True)#, save = True) #, 'r'])#, 'i', 'z'])
    # set metric
    LCs.setmetric(metric = np.array([1.]), logscale = [False])
        
    # set observations
    if not diffLC:
        LCs.set_observations(mjd = sn_mjd, flux = sn_flux, e_flux = sn_e_flux, filters = sn_filters, objname = SNname, plot = False, bandcolors = {'g': 'g', 'r': 'r', 'i': 'brown', 'z': 'k'})
    else:
        LCs.set_observations(mjd = sn_mjd, mjdref = sn_mjdref, flux = sn_flux, e_flux = sn_e_flux, filters = sn_filters, objname = SNname, plot = False, bandcolors = {'g': 'g', 'r': 'r', 'i': 'brown', 'z': 'k'})

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
    if 'texp' in par0.keys():
        texp = par0['texp']
    else:
        texp = texp0
    if not fixz and 'logz' in par0.keys():
        logz = par0['logz']
    else:
        logz = np.log(zcmb)
    logAv = np.log(0.1)#min(Avs))
    stretch = 1.
        
    parvals = np.array([scale, texp, logz, logAv, stretch])
    #parbounds = np.array([[0.1, 10.], [texp - 5, texp + 5], [np.log(1e-4), np.log(10.)], [np.log(1e-4), np.log(10.)], [12, 16], [0.5, 2.], [3e-5, 1e-2], [1., 1.], [10, 10], [1., 5.]])
    parbounds = np.array([[0.1, 10.], [texp - 5, texp + 5], [np.log(1e-4), np.log(10.)], [np.log(1e-4), np.log(10.)], [0.5, 1.5]])
    parlabels = np.array(["scale", "texp", "logz", "logAv", "stretch"])
    fixedvars = np.array([False,     False,  fixz,   False, False], dtype = bool)  # rcsm and vwinf should be True with current model grid
 
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
    scale, texp, logz, logAv, stretch = LCs.parvals
    
    # check best solution
    print("Best fit parameters:", zip(parlabels, LCs.parvals))
    print("...")
    print "Intrinsic variables solution", LCs.parvals[4:]
    LCmag, LCmagref = LCs.evalmodel(scale, texp, logz, logAv, LCs.parvals[4:], True, False)

    fig, ax = plt.subplots(figsize = (12, 7))
    modelplot = {}
    texpplot = ax.axvline(texp, c = 'gray', alpha = 1)
    for band in LCs.uniquefilters:
        if np.size(LCmag[band]) == 1:
            if LCmag[band] == 0:
                print("Error in %s" % SNname)
                sys.exit()
        mask = LCs.maskband[band]
        if np.sum(mask) > 0:
            ax.errorbar(LCs.mjd[mask], LCs.flux[mask], yerr = LCs.e_flux[mask], marker = 'o', c = LCs.bandcolors[band], lw = 0, elinewidth = 1)
            modelplot[band] = ax.plot(LCs.times + texp, (mag2flux(LCmag[band]) - mag2flux(LCmagref[band])), label = "%s" % band, c = LCs.bandcolors[band])
    #title = ax.set_title("scale: %5.3f, texp: %f, Av: %f, mass: %f, energy: %f, mdot: %3.1e, rcsm: %3.1f, beta: %f" % (scale, texp, np.exp(logAv), mass, energy, mdot, rcsm, beta), fontsize = 8)
    ax.legend(loc = 1, fontsize = 8, framealpha = 0.5)
    ax.set_xlim(min(texp, min(LCs.mjd)) - 5, max(LCs.mjd) + 10)

    if dointeractive:

        # TextBox with limits
        MJDminall = 1e99
        MJDmaxall = -1e99
        for band in LCs.uniquefilters:
            MJDminall = min(MJDminall, min(LCs.mjd[LCs.maskband[band]]) - 5)
            MJDmaxall = max(MJDmaxall, max(LCs.mjd[LCs.maskband[band]]) + 5)
        
        # slider axes
        texp_slider_ax =   fig.add_axes([0.15, 0.985, 0.75, 0.015], axisbg='w')
        #scale_slider_ax  = fig.add_axes([0.15, 0.97, 0.75, 0.015], axisbg='w')
        if not fixz: 
            logz_slider_ax =      fig.add_axes([0.15, 0.955, 0.75, 0.015], axisbg='w')
        av_slider_ax =     fig.add_axes([0.15, 0.94, 0.75, 0.015], axisbg='w')
        MJDmin_slider_ax = fig.add_axes([0.15, 0.925, 0.75, 0.015], axisbg='w')
        MJDmax_slider_ax = fig.add_axes([0.15, 0.91, 0.75, 0.015], axisbg='w')
        if LCs.dostretch:
            stretch_slider_ax = fig.add_axes([0.15, 0.045, 0.75, 0.015], axisbg='w')
        
        # slider objects
        MJDmin_slider = Slider(MJDmin_slider_ax, 'MJD min', MJDminall - 20, MJDmaxall, valinit=MJDminall, dragging = True)
        MJDmax_slider = Slider(MJDmax_slider_ax, 'MJD max', MJDminall, MJDmaxall, valinit=MJDmaxall, dragging = True)
        texp_slider = Slider(texp_slider_ax, 'texp', MJDminall - 20, MJDmaxall, valinit=texp0, dragging = False)
        #scale_slider = Slider(scale_slider_ax, 'log10 scale', -3., 1., valinit=1., dragging = False)
        if not fixz:
            logz_slider = Slider(logz_slider_ax, 'logz', LCs.parbounds[LCs.parlabels == 'logz'][0][0], LCs.parbounds[LCs.parlabels == 'logz'][0][1], valinit=LCs.parvals[LCs.parlabels == 'logz'][0], dragging = False)
        else:
            logz = LCs.parvals[LCs.parlabels == 'logz'][0]
        av_slider = Slider(av_slider_ax, 'logAv', LCs.parbounds[LCs.parlabels == 'logAv'][0][0], LCs.parbounds[LCs.parlabels == 'logAv'][0][1], valinit=LCs.parvals[LCs.parlabels == 'logAv'][0], dragging = False)

        if LCs.dostretch:
            stretch_slider = Slider(stretch_slider_ax, 'stretch', 0.5, 1.5, valinit=1., dragging = False)

        
        def slider_update(val):
        
            # update values
            #scale = 10**scale_slider.val
            scale = 1.
            texp = texp_slider.val
            if not fixz:
                logz = logz_slider.val
            else:
                logz = LCs.parvals[LCs.parlabels == 'logz'][0]
            logAv = av_slider.val
            if LCs.dostretch:
                stretch = stretch_slider.val
                LCs.parvals[LCs.nvext] = stretch

            # compute new model
            LCmag, LCmagref = LCs.evalmodel(scale, texp, logz, logAv, LCs.parvals[LCs.nvext:], True, False)
            texpplot.set_xdata(texp)
            fig.canvas.draw()
            # update plots
            for band in LCs.uniquefilters:
                modelplot[band][0].set_ydata((mag2flux(LCmag[band]) - mag2flux(LCmagref[band])))
                modelplot[band][0].set_xdata(LCs.times + texp)
                ax.relim()
                ax.autoscale_view()
                ax.set_title("scale: %5.3f, texp: %f, Av: %f" % (scale, texp, np.exp(logAv)), fontsize = 8)

                fig.canvas.draw_idle()


        def limits_update(val):
            ax.set_xlim(MJDmin_slider.val, MJDmax_slider.val)
            
        MJDmin_slider.on_changed(limits_update)
        MJDmax_slider.on_changed(limits_update)
        texp_slider.on_changed(slider_update)
        #scale_slider.on_changed(slider_update)
        if not fixz:
            logz_slider.on_changed(slider_update)
        av_slider.on_changed(slider_update)
        if LCs.dostretch:
            stretch_slider.on_changed(slider_update)
        
        plt.show()
        plt.savefig("plots/Bestfit_%s_%s.png" % (LCs.modelname, LCs.objname))

    if dotest:
        print("Test")
        sys.exit()

    # recover values
    if dointeractive:

        par0['MJDmin'] = MJDmin_slider.val
        par0['MJDmax'] = MJDmax_slider.val
        par0['texp'] = texp_slider.val
        if not fixz:
            par0['logz'] = logz_slider.val
        pickle.dump(par0, open("initial_pars/%s/%s.pkl" % (modelname, SNname), 'wb'), protocol = pickle.HIGHEST_PROTOCOL)
        sys.exit()
    
    # -------------------------
    # ---- MCMC ---------------
    # -------------------------

    
    # set prior distributions
    from scipy.stats import lognorm, norm, uniform  # leave this here, otherwise it fails!

    priors = np.array([lambda scale: norm.pdf(scale, loc = 1., scale = 0.3), \
                       lambda texp: norm.pdf(texp, loc = theta0[1], scale = 3.), \
                       lambda logz: norm.pdf(logz, loc = np.log(0.18), scale = 2), \
                       lambda logAv: norm.pdf(logAv, loc = np.log(0.05), scale = 1.), \
                       lambda stretch: norm.pdf(stretch, loc = 1., scale = 0.3)])
    
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
    LCs.doMCMC(bestfit = np.array(sol.x), nwalkers = nwalkers, deltabestfit = 1e-5, nsteps = nsteps, nburn = burnin, parlabels = parlabels, load = loadMCMC) 

    # plot results (and save lnlikes)
    LCs.plotMCMC(nburn = burnin, correctlogs = True)

