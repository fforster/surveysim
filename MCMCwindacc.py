import numpy as np
import re, sys, os, getopt
#import matplotlib # uncomment for using in leftraru
#matplotlib.use('Agg') # uncomment for using in leftraru
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

                
if __name__ == "__main__":

    # default values
    dointeractive = False
    dohits = False
    dodes = False
    verbose = False
    overwrite = False
    dotest = False
    loadMCMC = False
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "p:s:n:w:b:ioltvh", ["project=", "supernova=", "nsteps=", "walkers=", "burnin=", "interactive", "overwrite", "loadMCMC", "test", "verbose", "help"])
    except getopt.GetoptError:
        print 'python MCMCwindacc.py --help'
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print("Markov chain Monte Carlo fitting using Moriya wind acceleration models.")
            print("e.g. python ./MCMCwindacc.py --project DES --supernova DES15E2avs --interactive --verbose (interactively choose starting points)")
            print("e.g. python ./MCMCwindacc.py --project DES --supernova DES15E2avs --interactive --overwrite --verbose (interactively choose starting points, overwrite previously defined values)")
            print("e.g. python ./MCMCwindacc.py --project DES --supernova DES15E2avs --nsteps 1000 --walkers 400 --burnin 500 (run MCMC chain with 1000 steps, 400 walkers, burnin of 500)")
            print("e.g. python ./MCMCwindacc.py --project DES --supernova DES15E2avs --nsteps 1000 --walkers 400 --burnin 800 --loadMCMC --verbose (load MCMC chain for plotting, can choose new burnin value)")
        elif opt in ('-p', '--project'):
            if arg == 'HiTS':
                dohits = True
            elif arg == 'DES':
                dodes = True
        elif opt in ('-s', '--supernova'):
            SNname = arg
            print SNname
        elif opt in ('-n', '--nsteps'):
            nsteps = int(arg)
        elif opt in ('-w', '--walkers'):
            nwalkers = int(arg)
        elif opt in ('-b', '--burnin'):
            burnin = int(arg)
        elif opt in ('-o', '--overwrite'):
            overwrite = True
        elif opt in ('-', '--loadMCMC'):
            loadMCMC = True
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
    
    #########################
    # Observational data
    #########################
    
    # DES SNe
    # -----------------------------

    if dodes:
        
        DESdir = "../DES"
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
    
        #SNname = "DES15X2mku"
        #SNname = "DES13C2jtx"
        #SNname = "DES15S2eaq"
        #SNname = "DES15X1lzp"
        #SNname = "DES15E2avs"
        
        for SN in SNe.keys():
    
            if SN != SNname:
                continue
            
            zcmb = zSNe[SN]
            fixz = True
        
            for band in SNe[SN].keys():
    
                mjd = SNe[SN][band]["MJD"]
                mag = SNe[SN][band]["mag"]
                e_mag = SNe[SN][band]["e_mag"]
                
                #if SNname == "DES15X2mku":
                #    mask = (mjd > 57300) & (mjd < 57500)
                #elif SNname == "DES13C2jtx":
                #    mask = (mjd > 56450) & (mjd < 56700)
                #elif SNname == "DES15S2eaq":
                #    mask = (mjd > 57200) & (mjd < 57500)
                #elif SNname == "DES15X1lzp":
                #    mask = (mjd > 57300) & (mjd < 57500)

                if "mask" in locals():
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

        maskg = sn_filters == 'g'
        texp0 = sn_mjd[maskg][1 + np.argmax(np.diff(np.abs(sn_flux[maskg])))]

        #if np.argmax(sn_flux[maskg]) == 0:
        #    texp0 = sn_mjd[maskg][np.argmax(sn_flux[maskg])]
        #if SNname == "DES15X2mku":
        #    texp0 = 57325
        #elif SNname == "DES13C2jtx":
        #    texp0 = 56550
        #elif SNname == "DES15S2eaq":
        #    texp0 = 57275
        #elif SNname == "DES15X1lzp":
        #    texp0 = 57320

    # HiTS SNe
    # -----------------------------------------------------------------

    elif dohits:
        #SNname = "SNHiTS15A"
        #SNname = "SNHiTS15P"
        #SNname = "SNHiTS15D"
        #SNname = "SNHiTS15aw"
        #SNname = "SNHiTS15K"
        #SNname = "SNHiTS14B"
        #SNname = "SNHiTS15B"

        (MJDs, MJDrefs, ADUs, e_ADUs, mags, e1_mags, e2_mags, sn_filters) \
            = np.loadtxt("../HiTS/LCs/%s.txt" % SNname, usecols = (0, 1, 5, 6, 7, 8, 9, 10), dtype = str).transpose()

        sn_mjd = np.array(MJDs, dtype = float)
        sn_mjdref = np.array(MJDrefs, dtype = float)
        sn_adu = np.array(ADUs, dtype = float)
        sn_e_adu = np.array(e_ADUs, dtype = float)
        sn_mag = np.array(mags, dtype = float)
        sn_flux = np.array(sn_adu)
        sn_e_flux = np.array(sn_e_adu)
        maskg = sn_filters == 'g'
        if np.sum(maskg) > 0:
            idxmax = np.argmax(sn_adu[maskg])
            factorg = mag2flux(sn_mag[maskg][idxmax]) / sn_adu[maskg][idxmax]
            sn_flux[maskg] = sn_flux[maskg] * factorg
            sn_e_flux[maskg] = sn_e_flux[maskg] * factorg
        maskr = sn_filters == 'r'
        if np.sum(maskr) > 0:
            factorr = mag2flux(sn_mag[maskr][-1]) / sn_adu[maskr][-1]
            sn_flux[maskr] = sn_flux[maskr] * factorr
            sn_e_flux[maskr] = sn_e_flux[maskr] * factorr
        #texp0 = 57077.

        if SNname == "SNHiTS14B":
            sn_mjdref = np.hstack([sn_mjdref, sn_mjdref[-1], sn_mjdref[-1]])
            sn_mjd = np.hstack([sn_mjd, sn_mjd[0] + 9.01680793, sn_mjd[0] + 23.85])
            sn_flux = np.hstack([sn_flux, mag2flux(22.34), mag2flux(22.9)])
            sn_e_flux = np.hstack([sn_e_flux, sn_e_flux[-1], sn_e_flux[-1]])
            sn_filters = np.hstack([sn_filters, 'g', 'g'])
            
        mask = sn_filters == 'g'
        texp0 = sn_mjd[mask][np.argmax(np.diff(sn_flux[mask]))]

        if SNname == "SNHiTS14A":
            zcmb = 0.2175
            fixz = True
        elif SNname == "SNHiTS14Y":
            zcmb = 0.108
            fixz = True
        elif SNname == "SNHiTS14C":
            zcmb = 0.084
            fixz = True
        elif SNname == "SNHiTS14C":
            zcmb = 0.084
            fixz = True
        elif SNname == "SNHiTS15B":
            zcmb = 0.23
            fixz = True
        elif SNname == "SNHiTS15J":
            zcmb = 0.108
            fixz = True
        elif SNname == "SNHiTS15L":
            zcmb = 0.15
            fixz = True
        elif SNname == "SNHiTS15O":
            zcmb = 0.142
            fixz = True
        elif SNname == "SNHiTS15U":
            zcmb = 0.308
            fixz = True
        elif SNname == "SNHiTS15X":
            zcmb = 0.055807
            fixz = True
        elif SNname == "SNHiTS15ad":
            zcmb = 0.055392
            fixz = True
        elif SNname == "SNHiTS15al":
            zcmb = 0.2
            fixz = True
        elif SNname == "SNHiTS15aw":
            zcmb = 0.0663
            fixz = True
        elif SNname == "SNHiTS15be":
            zcmb = 0.151
            fixz = True
        elif SNname == "SNHiTS15bs":
            zcmb = 0.07
            fixz = True
        elif SNname == "SNHiTS15by":
            zcmb = 0.0524
            fixz = True
        elif SNname == 'SNHiTS15ck':
            zcmb = 0.042
            fixz = True
        else:
            zcmb = 0.2
            fixz = False


    else:
        print("Define observations...")
        sys.exit()

    # Theoretical  models
    # -------------------------------------------------------------

    modelsdir = "models"
    modelname = "MoriyaWindAcc"

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

    print par0

    # load models
    # --------------------------
    
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
    LCs = LCz_Av_params(modelsdir = modelsdir, modelname = modelname, files = files, paramnames = ["mass", "energy", "mdot", "rcsm", "vwindinf", "beta"], paramunits = ["Msun", "B", "Msun/yr", "1e15 cm", "km/s", ""], params = params, zs = zs, Avs = Avs, Rv = Rv, times = times)

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
    if 'texp' in par0.keys():
        texp = par0['texp']
    else:
        texp = texp0
    if 'logz' in par0.keys():
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
    if 'beta' in par0.keys():
        beta = par0['beta']
    else:
        beta = 3.
        
    rcsm = 1. # 1e15
    vwindinf = 10.
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
    print("...")
    print LCs.parvals[4:]
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
            modelplot[band] = ax.plot(LCs.times + texp, scale * (mag2flux(LCmag[band]) - mag2flux(LCmagref[band])), label = "%s" % band, c = LCs.bandcolors[band])
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
        
        mass_slider_ax = fig.add_axes([0.15, 0.045, 0.75, 0.015], axisbg='w')
        energy_slider_ax = fig.add_axes([0.15, 0.03, 0.75, 0.015], axisbg='w')
        mdot_slider_ax = fig.add_axes([0.15, 0.015, 0.75, 0.015], axisbg='w')
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
        mdot_slider = Slider(mdot_slider_ax, 'log mdot', np.log10(LCs.parbounds[LCs.parlabels == 'mdot'][0][0]), np.log10(LCs.parbounds[LCs.parlabels == 'mdot'][0][1]), valinit=np.log10(LCs.parvals[LCs.parlabels == 'mdot'][0]), dragging = False)
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
            LCs.parvals[LCs.parlabels == 'mdot'] = 10**mdot_slider.val
            LCs.parvals[LCs.parlabels == 'beta'] = beta_slider.val

            # compute new model
            LCmag, LCmagref = LCs.evalmodel(scale, texp, logz, logAv, LCs.parvals[LCs.nvext:], True, False)
            texpplot.set_xdata(texp)
            fig.canvas.draw()
            # update plots
            for band in LCs.uniquefilters:
                modelplot[band][0].set_ydata(scale * (mag2flux(LCmag[band]) - mag2flux(LCmagref[band])))
                modelplot[band][0].set_xdata(LCs.times + texp)
                ax.set_title("scale: %5.3f, texp: %f, Av: %f, mass: %f, energy: %f, mdot: %3.1e, rcsm: %3.1f, beta: %f" % (scale, texp, np.exp(logAv), mass, energy, mdot, rcsm, beta), fontsize = 8)

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
        mdot_slider.on_changed(slider_update)
        beta_slider.on_changed(slider_update)
        
        plt.show()
        plt.savefig("plots/Bestfit_%s_%s.png" % (LCs.modelname, LCs.objname))

    if dotest:
        LCs.parvals[-4] = 5e-4 # mdot
        LCs.parvals[-1] = 3.5 # beta
        LCs.parvals[-5] = 1. # energy
        print("Testing interpolation\n\n")
        LCs.test_interpolation("mass")
        LCs.test_interpolation("energy")
        LCs.test_interpolation("beta")
        LCs.test_interpolation("mdot")
        sys.exit()

    # recover values
    if dointeractive:

        par0['MJDmin'] = MJDmin_slider.val
        par0['MJDmax'] = MJDmax_slider.val
        par0['texp'] = texp_slider.val
        par0['mass'] = mass_slider.val
        par0['energy'] = energy_slider.val
        par0['mdot'] = mdot_slider.val
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

    priors = np.array([lambda scale: norm.pdf(scale, loc = 1., scale = 0.1), \
                       lambda texp: norm.pdf(texp, loc = theta0[1], scale = 3.), \
                       lambda logz: norm.pdf(logz, loc = np.log(0.18), scale = 2), \
                       lambda logAv: norm.pdf(logAv, loc = np.log(0.05), scale = 1.), \
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
    LCs.doMCMC(bestfit = np.array(sol.x), nwalkers = nwalkers, deltabestfit = 1e-5, nsteps = nsteps, nburn = burnin, parlabels = parlabels, load = loadMCMC) 

    # plot results
    np.array(LCs.plotMCMC(nburn = burnin, correctlogs = True, correctmdot = True))
