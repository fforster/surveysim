import numpy as np
import sys, os
import matplotlib.pyplot as plt
plt.switch_backend('agg')

os.environ["SURVEYSIM_PATH"] = "/home/rodrigo/supernovae_detection/surveysim"
sys.path.append("%s/lib" % os.environ["SURVEYSIM_PATH"])

from constants import *
from obsplan import *
from LCz import *
from LCz_Av import *
from LCz_Av_params import *
from SFHs import *
from survey_multimodel import *
import cos_calc
import glob
import pickle
import time

modelname = "MoriyaWindAcc"
obsname = "Blanco-DECam"

obs_list = sorted(glob.glob("../obsplans/HiTS_fields/*.dat"), key=str.lower)
aux_list = []
for obs in obs_list:
    aux_list.append(obs[-23:])
obs_list = aux_list
for obs in obs_list:
    print(obs)

real_obs = np.load("/home/rodrigo/supernovae_detection/HiTS_simulations/real_obs/pickles/camera_and_obs_cond.pkl")["obs_conditions"]
print(list((real_obs.keys()))[:5])
print("keys_per_field")
print(real_obs["Field01"][0].keys())

unique_bands = ["g", "r", "i"]
plot_comparison = False
nsim_per_field = 100

field_lightcurves = {}
field_params = {}

for i, obs_dir in enumerate(obs_list):
#for i in range(1):
    #if i == 2:
    #    continue
    plan = obsplan(obsname = obsname, mode = 'file-cols', inputfile = obs_dir, nfields = 1, nepochspernight = 1, nightfraction = 0.045, nread = 5, 
            doplot = True, doload = False, bandcolors = {'g': 'g', 'r':'r', 'i': 'brown'})
    plt.close("all")
    mask = np.array(plan.bands == "g")
    limmag = plan.limmag[mask]
    mjd = plan.MJDs[mask]
    real_limmag = []
    real_mjd = []
    field_name = obs_dir[-11:-4]
    for cond in real_obs[obs_dir[-11:-4]]:
        if cond["filter"] == "g":
            real_limmag.append(cond["limmag5"])
            real_mjd.append(cond["obs_day"])
    real_limmag = np.array(real_limmag)
    real_mjd = np.array(real_mjd)

    if plot_comparison:
        #plt.figure(figsize=(12,7))
        print("REAL, EST, MASK_LEN")
        print(len(real_mjd), len(mjd), len(mask))
        plt.plot(mjd, limmag, "-o", label="estimated")
        length = np.min([len(real_limmag), len(mjd)])
        index = np.argsort(real_mjd)
        plt.plot(real_mjd[index], real_limmag[index], "-o", label="Jorge")
        plt.legend()
        plt.xlabel("MJD")
        plt.ylabel("limmag")
        plt.title("Field"+str(i+1).zfill(2)+" limmag comparison")
        plt.ylim([26, 21])
        plt.savefig("./plots/HiTS_fields/"+obs_dir[-11:-4]+"_limmag.png")
        plt.show()

    # load models                                                                                                                                                                                                   
    modelsdir = "%s/models" % os.environ["SURVEYSIM_PATH"]
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
        files = np.array(list(map(lambda name: "%s.fr" % name, modelfile)))
    except:
        files = "%s.fr" % modelf

    # Redshift, Avs and time grids                                                                                                                                                                                        
    nz = 30
    ntimes = 100
    nAvs = 10
    zs = np.logspace(-3, 0, nz)
    times = np.logspace(-3, 3, ntimes)
    Avs = np.logspace(-4, 1, nAvs)
    Rv = 3

    # initialize LCz_Av_params models                                                                                                                                                                               
    paramnames = ["mass", "energy", "mdot", "rcsm", "vwindinf", "beta"]
    paramunits = ["Msun", "B", "Msun/yr", "1e15 cm", "km/s", ""]
    LCs = LCz_Av_params(modelsdir = modelsdir, modelname = modelname, \
                        files = files, paramnames = paramnames, paramunits = paramunits, params = params, \
                        zs = zs, Avs = Avs, Rv = Rv, times = times)
    
    LCs.docosmo()
    LCs.compute_models(bands = ['u', 'g', 'r', 'i', 'z'], obsname = obsname, load = True)
    parammetric = np.array([1., 1., 1e-6, 1., 10., 1.])
    paramlogscale = np.array([False, False, True, False, False, True], dtype = bool)
    LCs.setmetric(metric = parammetric, logscale = paramlogscale)

    LCs.set_observations(mjd = plan.MJDs, flux = None, e_flux = None, filters = plan.bands, objname = None, plot = False, bandcolors = {'g': 'g', 'r': 'r', 'i': 'brown', 'z': 'k'})
    # star formation
    SFH = SFHs(SFH = "MD14")
    knorm = 0.0091
    IIPfrac = 0.54
    efficiency = knorm * IIPfrac
    maxrestframeage=1.0
    newsurvey = survey_multimodel(obsplan = plan, SFH = SFH, efficiency = efficiency, LCs = LCs, maxrestframeage = maxrestframeage)
    newsurvey.set_maxz(0.55)
    newsurvey.do_cosmology()

    # set distribution of physical parameters                                                                                                                                                                       
    minMJD, maxMJD = min(newsurvey.obsplan.MJDs) - (20. + newsurvey.maxrestframeage) * (1. + max(newsurvey.zs)), max(newsurvey.obsplan.MJDs)
    rvs = {'texp': lambda nsim: uniform.rvs(loc = minMJD, scale = maxMJD - minMJD, size = nsim), \
           'logAv': lambda nsim: norm.rvs(loc = np.log(0.1), scale = 1., size = nsim), \
           #'mass': lambda nsim: norm.rvs(loc = 14, scale = 3, size = nsim), \                                                                                                                                      
           'mass': lambda nsim: uniform.rvs(loc = 12., scale = 4., size = nsim), \
           'energy': lambda nsim: norm.rvs(loc = 1., scale = 1., size = nsim), \
           #'log10mdot': lambda nsim: uniform.rvs(loc = -8, scale = 6, size = nsim), \
           'log10mdot' :lambda nsim: uniform.rvs(loc = -4, scale = 2, size = nsim), \
           'beta': lambda nsim: uniform.rvs(loc = 1., scale = 4., size = nsim)}
    bounds = {'texp': [minMJD, maxMJD], \
              'logAv': [np.log(1e-4), np.log(10.)], \
              'mass': [12, 16], \
              'energy': [0.5, 2.], \
              'log10mdot': [-4, -2], \
              'beta': [1., 5.]}

    mass = None
    energy = None
    mdot = None
    rcsm = 1. # 1e15
    vwindinf = 10.
    beta = None
    pars = np.array([mass, energy, mdot, rcsm, vwindinf, beta]) # must be in same order as paramnames

    newsurvey.sample_events(nsim = nsim_per_field, doload = False, doplot = False, doemergence = False, \
            rvs = rvs, bounds = bounds, pars = pars, dosave = True)
    field_lightcurves[field_name] = newsurvey.LCsamples
    field_params[field_name] = newsurvey.parsamples
    #if i == 0:
    #    break

file_name = "hits_sn_bounded"
pickle.dump(field_lightcurves, open("../pickles/"+file_name+"_"+str(nsim_per_field)+"_lc.pkl", "wb"), protocol = pickle.HIGHEST_PROTOCOL)
pickle.dump(field_params, open("../pickles/"+file_name+"_"+str(nsim_per_field)+"_params.pkl", "wb"), protocol = pickle.HIGHEST_PROTOCOL)





