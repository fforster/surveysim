import numpy as np
import sys, os
import matplotlib.pyplot as plt

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

modelname = "MoriyaWindAcc"
obsname = "Blanco-DECam"

obs_list = sorted(glob.glob("../obsplans/HiTS_fields/*.dat"), key=str.lower)
aux_list = []
for obs in obs_list:
    aux_list.append(obs[-23:])
obs_list = aux_list
for obs in obs_list:
    print(obs)

real_obs = np.load("/home/rodrigo/supernovae_detection/HiTS_simulations/real_data/camera_and_obs_cond.pkl")["obs_conditions"]
print(list((real_obs.keys()))[:5])
print("keys_per_field")
print(real_obs["Field01"][0].keys())

unique_bands = ["g", "r", "i"]

for i, obs_dir in enumerate(obs_list):
    if i == 2:
        continue
    plan = obsplan(obsname = obsname, mode = 'file-cols', inputfile = obs_dir, nfields = 1, nepochspernight = 1, nightfraction = 0.045, nread = 5, 
            doplot = True, doload = False, bandcolors = {'g': 'g', 'r':'r', 'i': 'brown'})
    plt.close("all")
    mask = np.array(plan.bands == "g")
    limmag = plan.limmag[mask]
    mjd = plan.MJDs[mask]
    real_limmag = []
    real_mjd = []

    for cond in real_obs[obs_dir[-11:-4]]:
        if cond["filter"] == "g":
            real_limmag.append(cond["limmag5"])
            real_mjd.append(cond["obs_days"])
    real_limmag = np.array(real_limmag)
    real_mjd = np.array(real_mjd)
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


