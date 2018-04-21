import numpy as np
import re, sys, os, getopt
leftraru = False
if os.getcwd() == "/home/fforster/surveysim":
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

if leftraru:
    os.environ["SURVEYSIM_PATH"] = "/home/fforster/surveysim"
else:
    os.environ["SURVEYSIM_PATH"] = "/home/fforster/Work/surveysim"
sys.path.append("%s/lib" % os.environ["SURVEYSIM_PATH"])

from constants import *
from LCz import *
from LCz_Av import *

from LCz_Av_params import *

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
    computemodels = False
    diffLC = False
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "O:B:p:s:n:w:b:dioltcf:vh", ["observatory=", "bands=", "project=", "supernova=", "nsteps=", "walkers=", "burnin=", "diffLC", "interactive", "overwrite", "loadMCMC", "test", "computemodels", "selfiles=", "verbose", "help"])
    except getopt.GetoptError:
        print('python MCMCwindacc.py --help')
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print("Markov chain Monte Carlo fitting using Moriya wind acceleration models.")
            print("e.g. python ./MCMCwindacc.py --observatory Blanco-DECam --bands 'u g r i z' --project DES --supernova DES15E2avs --interactive --verbose (interactively choose starting points)")
            print("e.g. python ./MCMCwindacc.py --project DES --supernova DES15E2avs --computemodels --filenames \"file1 file2\" (compute models in modellist.txt or in selected files)")
            print("e.g. python ./MCMCwindacc.py --project DES --supernova DES15E2avs --interactive --overwrite -`-verbose (interactively choose starting points, overwrite previously defined values)")
            print("e.g. python ./MCMCwindacc.py --project DES --supernova DES15E2avs --nsteps 1000 --walkers 400 --burnin 500 (run MCMC chain with 1000 steps, 400 walkers, burnin of 500)")
            print("e.g. python ./MCMCwindacc.py --project DES --supernova DES15E2avs --nsteps 1000 --walkers 400 --burnin 800 --loadMCMC --verbose (load MCMC chain for plotting, can choose new burnin value)")
        elif opt in ('-O', '--observatory'):
            obsname = arg
            print("Observatory: %s" % obsname)
        elif opt in ('-B', '--bands'):
            bands = arg.split()
            print("Bands: %s" % bands)
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
        elif opt in ('c', '--computemodels'):
            computemodels = True
        elif opt in ('f', '--selfiles'):
            selfiles = arg.split()
            print("Selected files:", selfiles)
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

    modelname = "MoriyaWindAcc"

    # if a previous interactive estimation of the parameters existed
    par0 = {}
    if not dointeractive:
        if os.path.exists("initial_pars/%s/%s.pkl" % (modelname, SNname)):
            par0 = pickle.load(open("initial_pars/%s/%s.pkl" % (modelname, SNname), 'rb'))
            print(par0)
            if 'MJDmin' in par0.keys() and 'MJDmax' in par0.keys():
                mask = (sn_mjd > par0['MJDmin']) & (sn_mjd < par0['MJDmax'])
                sn_mjd = sn_mjd[mask]
                sn_flux = sn_flux[mask]
                sn_e_flux = sn_e_flux[mask]
                sn_filters = sn_filters[mask]
    else:
        if not overwrite and os.path.exists("initial_pars/%s/%s.pkl" % (modelname, SNname)):
            print("Exiting, initial parameter estimation already exists. dotest: %s" % dotest)
            sys.exit()

    print(par0)

    # load models
    # --------------------------
    
    modelsdir = "%s/models" % os.environ["SURVEYSIM_PATH"]
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
        files = np.array(list(map(lambda name: "%s.fr" % name, modelfile)))
    except:
        files = "%s.fr" % modelfile
    #print(files)

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
    paramnames = ["mass", "energy", "mdot", "rcsm", "vwindinf", "beta"]
    paramunits = ["Msun", "B", "Msun/yr", "1e15 cm", "km/s", ""]
    LCs = LCz_Av_params(modelsdir = modelsdir, modelname = modelname, files = files, paramnames = paramnames, paramunits = paramunits, params = params, zs = zs, Avs = Avs, Rv = Rv, times = times)

    # do cosmology
    LCs.docosmo()

    # compute models in given bands
    if computemodels:
        if "selfiles" in locals(): 
            LCs.compute_models(obsname = obsname, bands = bands, save = True, selfiles = selfiles)
        else:
            LCs.compute_models(obsname = obsname, bands = bands, save = True)
        sys.exit()
    else:
        LCs.compute_models(obsname = obsname, bands = bands, load = True)
        
    # set metric
    LCs.setmetric(metric = np.array([1., 1., 1e-6, 1., 10., 1.]), logscale = np.array([False, False, True, False, False, False], dtype = bool))
        
    # set observations
    if not diffLC:
        LCs.set_observations(mjd = sn_mjd, flux = sn_flux, e_flux = sn_e_flux, filters = sn_filters, objname = SNname, plot = False, bandcolors = {'ROTSEIII': 'k', 'Kepler': 'k', 'g': 'g', 'R': 'r', 'r': 'r', 'i': 'brown', 'z': 'k'})
    else:
        LCs.set_observations(mjd = sn_mjd, mjdref = sn_mjdref, flux = sn_flux, e_flux = sn_e_flux, filters = sn_filters, objname = SNname, plot = False, bandcolors = {'ROTSEIII': 'k', 'Kepler': 'k', 'g': 'g', 'R': 'r', 'r': 'r', 'i': 'brown', 'z': 'k'})
    
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
    if 'mass' in par0.keys():
        mass = par0['mass']
    else:
        mass = 14.
    if 'energy' in par0.keys():
        energy = par0['energy']
    else:
        energy = 1. # foe
    if 'mdot' in par0.keys():
        mdot = 10**par0['mdot']
    else:
        mdot = 1e-6
    log10mdot = np.log10(mdot)
    if 'log10mdot' in par0.keys():
        log10mdot = par0['log10mdot']
    if 'beta' in par0.keys():
        beta = par0['beta']
    else:
        beta = 3.
        
    print(par0.keys(), par0, texp)

    rcsm = 1. # 1e15
    vwindinf = 10.
    parvals = np.array([scale, texp, logz, logAv, mass, energy, log10mdot, rcsm, vwindinf, beta])
    #parbounds = np.array([[0.1, 10.], [texp - 5, texp + 5], [np.log(1e-4), np.log(10.)], [np.log(1e-4), np.log(10.)], [12, 16], [0.5, 2.], [3e-5, 1e-2], [1., 1.], [10, 10], [1., 5.]])
    parbounds = np.array([[0.95, 1.05], [texp - 5, texp + 5], [np.log(1e-4), np.log(10.)], [np.log(1e-4), np.log(10.)], [12, 16], [0.5, 2.], [-8, -2], [1., 1.], [10, 10], [1., 5.]])
    parlabels = np.array(["scale", "texp", "logz", "logAv", "mass", "energy", "log10mdot", "rcsm", "vwindinf", "beta"])
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
    scale, texp, logz, logAv, mass, energy, log10mdot, rcsm, vwindinf, beta = LCs.parvals

    # check best solution
    print("Best fit parameters:", list(zip(parlabels, LCs.parvals)))
    print("...")
    print(LCs.parvals[4:])
    print(LCs.uniquefilters)
    LCmag, LCmagref = LCs.evalmodel(scale, texp, logz, logAv, LCs.parvals[4:], True, False)

    fig, ax = plt.subplots(figsize = (17, 11))
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
            modelplot[band] = ax.plot(LCs.times + texp, scale * (mag2flux(LCmag[band]) - mag2flux(LCmagref[band])), label = "%s" % band, c = LCs.bandcolors[band])
    #title = ax.set_title("scale: %5.3f, texp: %f, Av: %f, mass: %f, energy: %f, mdot: %3.1e, rcsm: %3.1f, beta: %f" % (scale, texp, np.exp(logAv), mass, energy, mdot, rcsm, beta), fontsize = 8)
    ax.legend(loc = 1, fontsize = 8, framealpha = 0.5)
    ax.set_xlim(min(texp, min(LCs.mjd)) - 5, max(LCs.mjd) + 10)
    plt.savefig("plots/Bestfit_%s_%s.png" % (LCs.modelname, LCs.objname))

    if dointeractive:

        # TextBox with limits
        MJDminall = 1e99
        MJDmaxall = -1e99
        for band in LCs.uniquefilters:
            MJDminall = min(MJDminall, min(LCs.mjd[LCs.maskband[band]]) - 5) - 5
            MJDmaxall = max(MJDmaxall, max(LCs.mjd[LCs.maskband[band]]) + 5)
        
        # slider axes
        texp_slider_ax =   fig.add_axes([0.15, 0.985, 0.75, 0.015], axisbg='w')
        #scale_slider_ax  = fig.add_axes([0.15, 0.97, 0.75, 0.015], axisbg='w')
        if not fixz: 
            logz_slider_ax =      fig.add_axes([0.15, 0.955, 0.75, 0.015], axisbg='w')
        av_slider_ax =     fig.add_axes([0.15, 0.94, 0.75, 0.015], axisbg='w')
        MJDmin_slider_ax = fig.add_axes([0.15, 0.925, 0.75, 0.015], axisbg='w')
        MJDmax_slider_ax = fig.add_axes([0.15, 0.91, 0.75, 0.015], axisbg='w')
        
        mass_slider_ax = fig.add_axes([0.15, 0.045, 0.75, 0.015], axisbg='w')
        energy_slider_ax = fig.add_axes([0.15, 0.03, 0.75, 0.015], axisbg='w')
        log10mdot_slider_ax = fig.add_axes([0.15, 0.015, 0.75, 0.015], axisbg='w')
        beta_slider_ax = fig.add_axes([0.15, 0.0, 0.75, 0.015], axisbg='w')
        
        # slider objects
        MJDmin_slider = Slider(MJDmin_slider_ax, 'MJD min', MJDminall, MJDmaxall, valinit=MJDminall, dragging = True)
        MJDmax_slider = Slider(MJDmax_slider_ax, 'MJD max', MJDminall, MJDmaxall, valinit=MJDmaxall, dragging = True)
        texp_slider = Slider(texp_slider_ax, 'texp', MJDminall, MJDmaxall, valinit=texp0, dragging = False)
        #scale_slider = Slider(scale_slider_ax, 'log10 scale', -3., 1., valinit=1., dragging = False)
        if not fixz:
            logz_slider = Slider(logz_slider_ax, 'logz', LCs.parbounds[LCs.parlabels == 'logz'][0][0], LCs.parbounds[LCs.parlabels == 'logz'][0][1], valinit=LCs.parvals[LCs.parlabels == 'logz'][0], dragging = False)
        else:
            logz = LCs.parvals[LCs.parlabels == 'logz'][0]
        av_slider = Slider(av_slider_ax, 'logAv', LCs.parbounds[LCs.parlabels == 'logAv'][0][0], LCs.parbounds[LCs.parlabels == 'logAv'][0][1], valinit=LCs.parvals[LCs.parlabels == 'logAv'][0], dragging = False)
        
        mass_slider = Slider(mass_slider_ax, 'mass', LCs.parbounds[LCs.parlabels == 'mass'][0][0], LCs.parbounds[LCs.parlabels == 'mass'][0][1], valinit=LCs.parvals[LCs.parlabels == 'mass'][0], dragging = False)
        energy_slider = Slider(energy_slider_ax, 'energy', LCs.parbounds[LCs.parlabels == 'energy'][0][0], LCs.parbounds[LCs.parlabels == 'energy'][0][1], valinit=LCs.parvals[LCs.parlabels == 'energy'][0], dragging = False)
        log10mdot_slider = Slider(log10mdot_slider_ax, 'log10mdot', LCs.parbounds[LCs.parlabels == 'log10mdot'][0][0], LCs.parbounds[LCs.parlabels == 'log10mdot'][0][1], valinit=LCs.parvals[LCs.parlabels == 'log10mdot'][0], dragging = False)
        beta_slider = Slider(beta_slider_ax, 'beta', LCs.parbounds[LCs.parlabels == 'beta'][0][0], LCs.parbounds[LCs.parlabels == 'beta'][0][1], valinit=LCs.parvals[LCs.parlabels == 'beta'][0], dragging = False)
        
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
            LCs.parvals[LCs.parlabels == 'mass'] = mass_slider.val
            LCs.parvals[LCs.parlabels == 'mass'] = mass_slider.val
            LCs.parvals[LCs.parlabels == 'energy'] = energy_slider.val
            LCs.parvals[LCs.parlabels == 'log10mdot'] = log10mdot_slider.val
            LCs.parvals[LCs.parlabels == 'beta'] = beta_slider.val

            # compute new model
            LCmag, LCmagref = LCs.evalmodel(scale, texp, logz, logAv, LCs.parvals[LCs.nvext:], True, False)
            texpplot.set_xdata(texp)
            fig.canvas.draw()
            # update plots
            for band in LCs.uniquefilters:
                modelplot[band][0].set_ydata(scale * (mag2flux(LCmag[band]) - mag2flux(LCmagref[band])))
                modelplot[band][0].set_xdata(LCs.times + texp)
                ax.relim()
                ax.autoscale_view()
                ax.set_title("SN: %s, scale: %5.3f, texp: %f, Av: %f, mass: %f, energy: %f, log10mdot: %f, rcsm: %3.1f, beta: %f" % (SNname, scale, texp, np.exp(logAv), mass, energy, log10mdot, rcsm, beta), fontsize = 8)

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
        mass_slider.on_changed(slider_update)
        energy_slider.on_changed(slider_update)
        log10mdot_slider.on_changed(slider_update)
        beta_slider.on_changed(slider_update)
        
        plt.show()
        plt.savefig("plots/Bestfit_%s_%s.png" % (LCs.modelname, LCs.objname))

    if dotest:
        LCs.parvals[-4] = np.log10(5e-4) # log10mdot
        LCs.parvals[-1] = 3.5 # beta
        LCs.parvals[-5] = 1. # energy
        print("Testing interpolation\n\n")
        LCs.test_interpolation("mass")
        LCs.test_interpolation("energy")
        LCs.test_interpolation("beta")
        LCs.test_interpolation("log10mdot")
        sys.exit()

    # recover values
    if dointeractive:

        par0['MJDmin'] = MJDmin_slider.val
        par0['MJDmax'] = MJDmax_slider.val
        par0['texp'] = texp_slider.val
        par0['mass'] = mass_slider.val
        par0['energy'] = energy_slider.val
        par0['log10mdot'] = log10mdot_slider.val
        par0['beta'] = beta_slider.val
        if not fixz:
            par0['logz'] = logz_slider.val
        pickle.dump(par0, open("initial_pars/%s/%s.pkl" % (modelname, SNname), 'wb'), protocol = pickle.HIGHEST_PROTOCOL)
        sys.exit()
    
    # -------------------------
    # ---- MCMC ---------------
    # -------------------------

    
    # set prior distributions
    from scipy.stats import lognorm, norm, uniform  # leave this here, otherwise it fails!

    
    priors = np.array([lambda scale: norm.pdf(scale, loc = 1., scale = 0.01), \
                       lambda texp: norm.pdf(texp, loc = theta0[1], scale = 4.), \
                       lambda logz: norm.pdf(logz, loc = np.log(0.18), scale = 2), \
                       lambda logAv: norm.pdf(logAv, loc = np.log(0.05), scale = 2.), \
                       lambda mass: norm.pdf(mass, loc = 14, scale = 3), \
                       lambda energy: norm.pdf(energy, loc = 1., scale = 1.), \
                       lambda log10mdot: uniform.pdf(log10mdot, loc = -8., scale = 6.), \
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
    LCs.doMCMC(bestfit = np.array(sol.x), nwalkers = nwalkers, deltabestfit = 1e-5, nsteps = nsteps, nburn = burnin, parlabels = parlabels, load = loadMCMC) 

    # plot results
    LCs.plotMCMC(nburn = burnin)#, correctlogs = True) #, correctmdot = True)
