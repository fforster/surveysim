import re, os, sys
import numpy as np
from scipy.interpolate import interp1d

os.environ["SURVEYSIM_PATH"] = "/home/fforster/Work/surveysim"
sys.path.append("%s/lib" % os.environ["SURVEYSIM_PATH"])

leftraru = False
if os.getcwd() == "/home/fforster/surveysim":
    leftraru = True
    import matplotlib # uncomment for using in leftraru
    matplotlib.use('Agg') # uncomment for using in leftraru
import matplotlib.pyplot as plt

from sklearn.neighbors.kde import KernelDensity

from constants import *
from LCz import *
from LCz_Av import *
from LCz_Av_params import *

from readSNdata import *

import pandas as pd

survey = sys.argv[1]
mode = sys.argv[2]

if mode == "LCs":
    doLCs = True
    doMCMC = False
elif mode == "MCMC":
    doLCs = False
    doMCMC = True

if survey == 'HiTS':

    df = pd.read_csv("SNIIpaper/HiTS_classification.out", sep = "\s+", comment = "#")
    print(df.keys(), df.dtypes)
    SNe = df.SNe
    BICII = df.BICII
    BICIa = df.BICIa
    spec_class = df.spec_class
    banned = df.banned
    poor_rise =  df.poor_rise
    HiTS = sorted(SNe[((spec_class == 'II') | (BICII < BICIa)) & (banned == False) & (poor_rise == False)])
    DES = []

elif survey == 'DES':
    
    DES = ["DES13C2jtx",
       #"DES13C3ui",
       "DES13X3fca",
       #"DES14C1coz",
       #"DES14C2apq",
       #"DES14C3aol",
       #"DES14C3nm",
       #"DES14C3oz",
       #"DES14C3rhw",
       #"DES14C3tsg",
       ##"DES14E2ar",
       ##"DES14X1qt",
       #"DES14X2cy",
       #"DES14X3ili",
       #"DES15C1okz",
       #"DES15C1pkx",
       #"DES15C2eaz",
       #"DES15C2lna",
       "DES15C2lpp",
       "DES15C2npz",
       #"DES15C3bj",
       #"DES15E1iuh",
       #"DES15E2avs",
       #"DES15E2ni",
       #"DES15S1by",
       ##"DES15S1cj",
       ##"DES15S1lrp",
       "DES15S2eaq",
       "DES15X1lzp",
       "DES15X2mku",
       ##"DES15X3kxq",
       "DES15X3mpq",
       "DES15X3nad",
       #"DES16C1cbg",
       ###"DES16C2cbv",
       #"DES16C3at",
       #"DES16E1ah",
       ###"DES16E1bkh",
       "DES16S1gn",
       "DES16X1ey",
       #"DES16X2bkr",
       #"DES16X3cpl",
       #"DES16X3dvb",
       #"DES16X3jj",
       #"DES16X3km"
       ]

spectra = {}
spectra["SNHiTS15al"] = "Ia"
spectra["SNHiTS15be"] = "Ia"
spectra["SNHiTS15bs"] = "Ia"
spectra["SNHiTS15by"] = "II"
spectra["SNHiTS15bu"] = "Ia"
spectra["SNHiTS15cf"] = "Ia"
spectra["SNHiTS15by"] = "II"
spectra["SNHiTS14C"] = "II" # Greta, http://www.astronomerstelegram.org/?read=5957
spectra["SNHiTS14D"] = "II" # Emilia, blue continuum, http://www.astronomerstelegram.org/?read=5957
spectra["SNHiTS14H"] = "Ia" # Pamela, http://www.astronomerstelegram.org/?read=6014, http://www.astronomerstelegram.org/?read=5970
spectra["SNHiTS14F"] = "Ia" # Mara, http://www.astronomerstelegram.org/?read=6014
spectra["SNHiTS14B"] = "II" # Bel, http://www.astronomerstelegram.org/?read=6014
spectra["SNHiTS15L"] = "Ia" # Natalia, http://www.astronomerstelegram.org/?read=7144
spectra["SNHiTS15I"] = "Ia" # Olga-Lucia, http://www.astronomerstelegram.org/?read=7154
spectra["SNHiTS15J"] = "Ia" # Teahine, http://www.astronomerstelegram.org/?read=7154
spectra["SNHiTS15D"] = "II" # Daniela, http://www.astronomerstelegram.org/?read=7162
spectra["SNHiTS15P"] = "II" # Rosemary, http://www.astronomerstelegram.org/?read=7162
spectra["SNHiTS15ad"] = "Ia" # Gabriela, http://www.astronomerstelegram.org/?read=7164
spectra["SNHiTS15aw"] = "II" # Maria Soledad, http://www.astronomerstelegram.org/?read=7246
spectra["SNHiTS15al"] = "Ia" # Goretti, http://www.astronomerstelegram.org/?read=7291
spectra["SNHiTS15be"] = "Ia" # Agustina, http://www.astronomerstelegram.org/?read=7291
spectra["SNHiTS15bs"] = "Ia" # Rita, http://www.astronomerstelegram.org/?read=7291
spectra["SNHiTS15by"] = "II" # Tanit, PS15ou, http://www.astronomerstelegram.org/?read=7291
spectra["SNHiTS15bu"] = "Ia" # Ane, http://www.astronomerstelegram.org/?read=7335
spectra["SNHiTS15cf"] = "Ia" # Nines, http://www.astronomerstelegram.org/?read=7335


# Theoretical  models
# -------------------------------------------------------------

modelsdir = "models"
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
    files = np.array(list(map(lambda name: "%s.fr" % name, modelfile)))
except:
    files = "%s.fr" % modelfile
    print(modelfile)
    #print(files)
    
# Redshift, Avs and time
nz = 30
ntimes = 100
nAvs = 10
zs = np.logspace(-3, 0, nz)
times = np.logspace(-3, 3, ntimes)
Avs = np.logspace(-4, 1, nAvs)
Rv = 3.25


if mode == 'LCs':
    # initialize MCMCfit model
    LCs = LCz_Av_params(modelsdir = modelsdir, modelname = modelname, files = files, paramnames = ["mass", "energy", "mdot", "rcsm", "vwindinf", "beta"], paramunits = ["Msun", "B", "Msun/yr", "1e15 cm", "km/s", ""], params = params, zs = zs, Avs = Avs, Rv = Rv, times = times)
    
    # do cosmology
    LCs.docosmo()
    
    # compute models in given bands
    LCs.compute_models(bands = ['u', 'g', 'r', 'i', 'z'], load = True)#, save = True)#, 'r'])#, 'i', 'z'])
    
    # set metric
    LCs.setmetric(metric = np.array([1., 1., 1e-6, 1., 10., 1.]), logscale = np.array([False, False, True, False, False, True], dtype = bool))
        
master = {}
limlog10mdot = []
limbeta = []
limAv = []
limz = []
limmass = []
limenergy = []
limtexp14 = []
limtexp15 = []

# control of what to do
ncols = 4
nrows = int(np.ceil(1. * len(HiTS) / ncols))
idxfilled = {}
if doLCs:
    fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize = (11, 12))
elif doMCMC:
    fig, ax = plt.subplots(figsize = (12, 8))

# string to convert all LCs into a pdf    
pdfout = ""

# markers
ms = matplotlib.markers.MarkerStyle().markers.keys()
ms_sel = []
for m in ms:
    try:
        int(m)
    except:
        if m != None and m != 'None' and m != "" and m != " " and m != "|" and m != "+":
            ms_sel.append(m)
    counter = 1
print(ms_sel)

# iterate among samples files
files = sorted(os.listdir("samples"))
model = "MoriyaWindAcc"
if mode == 'MCMC':
    fileout = open("summary_HiTS.out", 'w')
    fileout.write("SN log10mdotp5 log10mdotp50 log10mdotp95 betap5 betap50 betap95 massp5 massp50 massp95 energyp5 energyp50 energyp95 zp5 zp50 zp95 Avp5 Avp50 Avp95 texpp5 texpp50 texpp95\n")

for f in files:
    
    SNfound = re.findall("chain_%s_(.*)_.*?.dat" % (model), f)
    if SNfound != []:
        SN = SNfound[0]
    else:
        continue

    if SN in HiTS or SN in DES:

        print(SN)

        #if SN != "SNHiTS14B":
        #    continue
        if SN in HiTS:
            m = ms_sel[np.mod(counter, len(ms_sel))]
        elif SN in DES:
            m = '*'

        #if SN != "SNHiTS15X":
        #    continue
            
        if SN in HiTS:

            # read observational data
            sn_mjd, sn_mjdref, sn_flux, sn_e_flux, sn_filters, fixz, zcmb, texp0 = readSNdata(survey, SN)

        pdf = f.replace("chain", "plots/MCMC").replace(".dat", "_models.pdf")
        pdfout = "%s %s" % (pdfout, pdf)

        if re.search(".*logz.*", f) and not fixz:
            nchain, nwalker, scale, texp, logz, logAv, mass, energy, log10mdot, beta = np.loadtxt("samples/%s" % f).transpose()
        elif re.search(".*logz.*", f) == None and fixz:
            nchain, nwalker, scale, texp, logAv, mass, energy, log10mdot, beta = np.loadtxt("samples/%s" % f).transpose()
        else:
            print("Skipping file %s" % f)
            continue

        # get array positions
        (iy, ix) = int(np.mod(counter - 1, ncols)), int((counter - 1) / ncols)
        idxfilled[(ix, iy)] = True
        counter = counter + 1

        mask = (nchain > 500)
            
        # prepare light curves
        if doLCs:
            
            # set observations
            LCs.set_observations(mjd = sn_mjd, mjdref = sn_mjdref, flux = sn_flux, e_flux = sn_e_flux, filters = sn_filters, objname = SN, plot = False, bandcolors = {'g': 'g', 'r': 'r', 'i': 'brown', 'z': 'k'})

            # plot light curves
            for band in LCs.uniquefilters:
                maskb = LCs.maskband[band]
                if np.sum(maskb) > 0:
                    if band == 'g':
                        marker = 'o'
                    elif band == 'r':
                        marker = 's'
                    ax[ix, iy].errorbar(LCs.mjd[maskb], LCs.flux[maskb], yerr = LCs.e_flux[maskb], marker = marker, c = LCs.bandcolors[band], lw = 0, elinewidth = 1, markersize = 5, alpha = 0.7, label = "%s" % band)
                    ax[ix, iy].set_xlim(min(min(texp[mask]), min(LCs.mjd)) - 1, max(LCs.mjd) + 3)
            

            # plot light curves sampled from the posterior
            nplots = 100
            idxselected = np.array(np.random.random(size = nplots) * np.sum(mask), dtype = int)
            zerolevels = []
            for count, idxsel in enumerate(idxselected):
            
                scale_val = scale[mask][idxsel]
                texp_val = texp[mask][idxsel]
                if fixz:
                    logz_val = np.log(zcmb)
                else:
                    logz_val = logz[mask][idxsel]
                logAv_val = logAv[mask][idxsel]
                mass_val = mass[mask][idxsel]
                energy_val = energy[mask][idxsel]
                log10mdot_val = log10mdot[mask][idxsel]
                beta_val = beta[mask][idxsel]
                rcsm_val = 1. # 1e15
                vwindinf_val = 10.
            
                # values
                parvals = np.array([scale_val, texp_val, logz_val, logAv_val, mass_val, energy_val, log10mdot_val, rcsm_val, vwindinf_val, beta_val])
                parbounds = np.array([[0.1, 10.], [texp_val - 5, texp_val + 5], [np.log(1e-4), np.log(10.)], [np.log(1e-4), np.log(10.)], [12, 16], [0.5, 2.], [1e-6, 1e-2], [1., 1.], [10, 10], [1., 5.]])
                parlabels = np.array(["scale", "texp", "logz", "logAv", "mass", "energy", "log10mdot", "rcsm", "vwindinf", "beta"])
                fixedvars = np.array([False,     False,  fixz,   False,   False,   False,    False,   True,   True,      False], dtype = bool)  # rcsm and vwinf should be True with current model grid
            
                # initialize with previous parameters
                theta0 = parvals[np.invert(fixedvars)]
                sol = LCs.findbest(theta0 = theta0, parbounds = parbounds, fixedvars = fixedvars, parvals = parvals, parlabels = parlabels, skip = True)
            
                LCmag, LCmagref = LCs.evalmodel(scale_val, texp_val, logz_val, logAv_val, LCs.parvals[4:], True, False)
            
                # plot light curves
                for band in LCs.uniquefilters:
                    maskb = LCs.maskband[band]
                    if np.sum(maskb) > 0:
                        if count == 0:
                            l1 = "%s model" % band
                            l2 = r"$t_{\rm exp}$"
                        else:
                            l1 = ""
                            l2 = ""
                        ax[ix, iy].plot(LCs.times + texp_val, mag2flux(LCmag[band]) - mag2flux(LCmagref[band]), c = LCs.bandcolors[band], lw = 1, alpha = 0.05)
                        if band == 'g':
                            zerolevels.append((mag2flux(LCmag[band]) - mag2flux(LCmagref[band]))[0])
                        ax[ix, iy].axvline(texp_val, alpha = 0.05, c = 'gray')

                        
            ## add zero mass loss model
            #scale_val = np.median(scale[mask])
            #texp_val = np.median(texp[mask])
            #logz_val = np.median(logz[mask])
            #logAv_val  = np.median(logAv[mask])
            #rcsm_val = 1. # 1e15
            #vwindinf_val = 10.
            
            parvals = np.array([scale_val, texp_val, logz_val, logAv_val, mass_val, energy_val, -8., rcsm_val, vwindinf_val, 3.])
            #parvals = np.array([scale_val, texp_val, logz_val, logAv_val, np.median(mass[mask]), np.median(energy[mask]), -8., rcsm_val, vwindinf_val, np.median(beta[mask])])
            parbounds = np.array([[0.1, 10.], [texp_val - 5, texp_val + 5], [np.log(1e-4), np.log(10.)], [np.log(1e-4), np.log(10.)], [12, 16], [0.5, 2.], [1e-6, 1e-2], [1., 1.], [10, 10], [1., 5.]])
            parlabels = np.array(["scale", "texp", "logz", "logAv", "mass", "energy", "log10mdot", "rcsm", "vwindinf", "beta"])
            fixedvars = np.array([False,     False,  fixz,   False,   False,   False,    False,   True,   True,      False], dtype = bool)  # rcsm and vwinf should be True with current model grid

            # initialize with previous parameters
            theta0 = parvals[np.invert(fixedvars)]
            sol = LCs.findbest(theta0 = theta0, parbounds = parbounds, fixedvars = fixedvars, parvals = parvals, parlabels = parlabels, skip = True)
            
            LCmag, LCmagref = LCs.evalmodel(scale_val, texp_val, logz_val, logAv_val, LCs.parvals[4:], True, False)
            
            # plot light curves with last values except mass loss
            for iband, band in enumerate(LCs.uniquefilters):
                maskb = LCs.maskband[band]
                if np.sum(maskb) > 0 and band == 'g':
                    ax[ix, iy].plot(LCs.times + texp_val, mag2flux(LCmag[band]) + np.median(zerolevels), c = LCs.bandcolors[band], lw = 1, ls = '--', label = "%s" % band)
                ax[ix, iy].text(0, iband / 10., band, color = LCs.bandcolors[band])
            ax[ix, iy].text(0, 0.3, r"$t_{\rm exp}$", color = 'gray')
            
            # labels                    
            ax[ix, iy].set_yticklabels([])
            (x1, x2) = ax[ix, iy].get_xlim()
            (y1, y2) = ax[ix, iy].get_ylim()
            label = SN
            if SN in spectra.keys():
                label = "%s*" % SN
            ax[ix, iy].text(x2, y1 + (y2 - y1) * 0.05, label, fontsize = 10, ha = 'right')
            if fixz:
                ax[ix, iy].text(x2, y1 + (y2 - y1) * 0.15, r"$z=%4.2f$" % zcmb, fontsize = 10, ha = 'right')

            for iband, band in enumerate(LCs.uniquefilters):
                maskb = LCs.maskband[band]
                if np.sum(maskb) > 0:
                    ax[ix, iy].text(x1 + (x2 - x1) * 0.02, y2 - (y2 - y1) * 0.1 * (iband + 1), band, color = LCs.bandcolors[band], fontsize = 10)
            ax[ix, iy].text(x1 + (x2 - x1) * 0.02, y2 - (y2 - y1) * 0.1 * (iband + 2), r"$t_{\rm exp}$", color = 'dimgray', fontsize = 10)

            #ax[ix, iy].legend(loc = 2, fontsize = 8)
            #break
            
            
        if not doMCMC:
            continue

        if not fixz:
            logzs  = logz[mask]
        else:
            logzs = np.ones(np.sum(mask)) * np.log(zcmb)
            
        if "log10mdot" not in master.keys():
            master['log10mdot'] = log10mdot[mask]
            master['beta'] = beta[mask]
            master['mass'] = mass[mask]
            master['energy'] = energy[mask]
            master['logAv'] = logAv[mask]
            master['logz'] = logzs
        else:
            master['log10mdot'] = np.hstack([master['log10mdot'], log10mdot[mask]])
            master['beta'] = np.hstack([master['beta'], beta[mask]])
            master['mass'] = np.hstack([master['mass'], mass[mask]])
            master['energy'] = np.hstack([master['energy'], energy[mask]])
            master['logAv'] = np.hstack([master['logAv'], logAv[mask]])
            master['logz'] = np.hstack([master['logz'], logzs])

        if SN[:8] == 'SNHiTS14':
            if 'texp14' not in master.keys():
                master['texp14'] = texp[mask]
            else:
                master['texp14'] = np.hstack([master['texp14'], texp[mask]])
        elif SN[:8] == 'SNHiTS15':
            if 'texp15' not in master.keys():
                master['texp15'] = texp[mask]
            else:
                master['texp15'] = np.hstack([master['texp15'], texp[mask]])

        limlog10mdot.append(np.array(list(map(lambda x: np.percentile(log10mdot[mask], x), [5, 50, 95]))))
        limbeta.append(np.array(list(map(lambda x: np.percentile(beta[mask], x), [5, 50, 95]))))
        limmass.append(np.array(list(map(lambda x: np.percentile(mass[mask], x), [5, 50, 95]))))
        limenergy.append(np.array(list(map(lambda x: np.percentile(energy[mask], x), [5, 50, 95]))))
        limAv.append(np.exp(np.array(list(map(lambda x: np.percentile(logAv[mask], x), [5, 50, 95])))))
        if fixz:
            limz.append(zcmb * np.ones(3))
        else:
            limz.append(np.exp(np.array(list(map(lambda x: np.percentile(logz[mask], x), [5, 50, 95])))))
        if SN[:8] == 'SNHiTS14':
            limtexp14.append(np.array(list(map(lambda x: np.percentile(texp[mask], x), [5, 50, 95]))))
        elif SN[:8] == 'SNHiTS15':
            limtexp15.append(np.array(list(map(lambda x: np.percentile(texp[mask], x), [5, 50, 95]))))

        xerr = [[limlog10mdot[-1][1] - limlog10mdot[-1][0]], [limlog10mdot[-1][2] - limlog10mdot[-1][1]]]
        yerr = [[limbeta[-1][1] - limbeta[-1][0]], [limbeta[-1][2] - limbeta[-1][1]]]

        fileout.write("%s %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n" % (SN, limlog10mdot[-1][0], limlog10mdot[-1][1], limlog10mdot[-1][2], limbeta[-1][0], limbeta[-1][1], limbeta[-1][2], limmass[-1][0], limmass[-1][1], limmass[-1][2], limenergy[-1][0], limenergy[-1][1], limenergy[-1][2], limz[-1][0], limz[-1][1], limz[-1][2], limAv[-1][0], limAv[-1][1], limAv[-1][2], np.percentile(texp[mask], 5), np.percentile(texp[mask], 50), np.percentile(texp[mask], 95)))
        ax.errorbar(limlog10mdot[-1][1], limbeta[-1][1], marker = m, markersize = 10, xerr = xerr, yerr = yerr, label = SN, alpha = 0.8)

if mode == "MCMC":
    fileout.close()

if doLCs:
    #for ix in range(nrows):
    #    for iy in range(ncols):
    #        if (ix, iy) not in idxfilled.keys():
    #            ax[ix, iy].axis('off')#set_yticklabels([])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.15)
    plt.savefig("plots/samples/LCs.pdf")
        
if not doMCMC:
    sys.exit()

    
# limits
limlog10mdot = np.array(limlog10mdot)
limbeta = np.array(limbeta)
limmass = np.array(limmass)
limenergy = np.array(limenergy)
limAv = np.array(limAv)
limz = np.array(limz)
limtexp14 = np.array(limtexp14)
limtexp15 = np.array(limtexp15)
    
#os.system("convert %s plots/samples/%s.pdf" % (pdfout, survey))

# literature SNe
doliterature = False
if doliterature:
    lit = {}
    lit["SN2006bp"] = np.array([0.0007462115922905706, 0.0020939756582619195, 0.002645035113196285, 3.7430172930300003, 4.68504380389, 4.995671344830001])
    lit["PS1-13arp"] = np.array([0.00044989571364489905, 0.0011752909504824866, 0.0025315320687651294, 1.29554086565, 3.42109825438, 4.81065224547])
    lit["SN2013fs"] = np.array([0.001957510342591815, 0.0024105015199778245, 0.002450768518921163, 3.748174717917, 3.7504185231699996, 4.475784397759999])
    lit["KSN2011a"] = np.array([0.0018668336766834554, 0.0025203335414241937, 0.0030167088534899474, 1.0782722334475, 2.493082275845, 2.9992118402599997])
    lit["KSN2011d"] = np.array([0.0017007478378221167, 0.0019022813041540691, 0.0026243068983511078, 4.08315159015, 4.39987035739, 4.66486208759])

    for idx, SN in enumerate(lit.keys()):
        print(SN)
        lit[SN][:3] = np.log10(lit[SN][:3])
        m = ms_sel[idx + 2]
        xerr = [[lit[SN][1] - lit[SN][0]], [lit[SN][2] - lit[SN][1]]]
        yerr = [[lit[SN][4] - lit[SN][3]], [lit[SN][5] - lit[SN][4]]]
        ax.errorbar(lit[SN][1], lit[SN][4], marker = m, markersize = 10, xerr = xerr, yerr = yerr, label = SN, alpha = 1., c = 'k')

    plt.legend(loc = [0.17, 0.0], fontsize = 11)  # for all SNe
else:
    plt.legend(loc = [0.17, 0.07], fontsize = 11)  # only HiTS

ax.set_xlim(-8, -2)
ax.set_ylim(1, 5)
ax.set_xlabel(r"$\log_{10}\ \dot M\ [M_\odot/yr]$", fontsize = 14)
ax.set_ylabel(r"$\beta$", fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.savefig("plots/samples/betavslog10mdot_%s.pdf" % survey)

fig, ax = plt.subplots()
ax.hist2d(master['mass'], master['energy'])
ax.set_xlabel("mass [Msun]")
ax.set_ylabel("energy [B]")
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.savefig("plots/samples/energyvsmass_hist_%s.pdf" % survey)

fig, ax = plt.subplots()
ax.errorbar(limmass[:, 1], limenergy[:, 1], xerr = [limmass[:, 1] - limmass[:, 0], limmass[:, 2] - limmass[:, 1]],
            yerr = [limenergy[:, 1] - limenergy[:, 0], limenergy[:, 2] - limenergy[:, 1]], lw = 0, elinewidth = 1, marker = 'o')
ax.set_xlabel("mass [Msun]", fontsize = 14)
ax.set_ylabel("energy [B]", fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.savefig("plots/samples/energyvsmass_%s.pdf" % survey)

fig, ax = plt.subplots()
ax.errorbar(limz[:, 1], limlog10mdot[:, 1], xerr = [limz[:, 1] - limz[:, 0], limz[:, 2] - limz[:, 1]],
            yerr = [limlog10mdot[:, 1] - limlog10mdot[:, 0], limlog10mdot[:, 2] - limlog10mdot[:, 1]], lw = 0, elinewidth = 1, marker = 'o')

(z, log10mdot) = np.load("pickles/MoriyaWindAcc_HiTS15A-nf50-ne1-nr1-nn37_Blanco-DECam_gir_zlog10mdot.npy")
H, xedges, yedges = np.histogram2d(z, log10mdot, range = [[0, 0.6], [-8, -2]], bins = (12, 12))
x, y = np.meshgrid((xedges[1:] + xedges[:-1]) / 2., (yedges[1:] + yedges[:-1]) / 2.)
extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
cset = ax.contour(x, y, H.transpose(), origin = 'lower', colors = 'gray')#, levels, origin = 'lower')

ax.set_ylabel(r"$\log_{10}\ \dot M\ [M_\odot/yr]$", fontsize = 14)
ax.set_xlabel("z", fontsize = 14)
ax.set_xlim(0, 0.55)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.savefig("plots/samples/log10mdotvsz_%s.pdf" % survey)

fig, ax = plt.subplots()
ax.hist2d(master['mass'], master['log10mdot'])
ax.set_xlabel("mass [Msun]")
ax.set_ylabel(r"$\log_{10}\ \dot M\ [M_\odot/yr]$")
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.savefig("plots/samples/log10mdotvsmass_hist_%s.pdf" % survey)


fig, ax = plt.subplots()
ax.errorbar(limmass[:, 1], limlog10mdot[:, 1], xerr = [limmass[:, 1] - limmass[:, 0], limmass[:, 2] - limmass[:, 1]],
            yerr = [limlog10mdot[:, 1] - limlog10mdot[:, 0], limlog10mdot[:, 2] - limlog10mdot[:, 1]], lw = 0, elinewidth = 1, marker = 'o')
ax.set_ylabel(r"$\log_{10}\ \dot M\ [M_\odot/yr]$", fontsize = 14)
ax.set_xlabel("mass [Msun]", fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.savefig("plots/samples/log10mdotvsmass_%s.pdf" % survey)

fig, ax = plt.subplots()
ax.errorbar(limAv[:, 1], limlog10mdot[:, 1], xerr = [limAv[:, 1] - limAv[:, 0], limAv[:, 2] - limAv[:, 1]],
            yerr = [limlog10mdot[:, 1] - limlog10mdot[:, 0], limlog10mdot[:, 2] - limlog10mdot[:, 1]], lw = 0, elinewidth = 1, marker = 'o')
ax.set_ylabel(r"$\log_{10}\ \dot M\ [M_\odot/yr]$", fontsize = 14)
ax.set_xlabel("Av", fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.savefig("plots/samples/log10mdotvsAv_%s.pdf" % survey)

fig, ax = plt.subplots()
ax.errorbar(limmass[:, 1], limbeta[:, 1], xerr = [limmass[:, 1] - limmass[:, 0], limmass[:, 2] - limmass[:, 1]],
            yerr = [limbeta[:, 1] - limbeta[:, 0], limbeta[:, 2] - limbeta[:, 1]], lw = 0, elinewidth = 1, marker = 'o')
ax.set_ylabel(r"$\log_{10}\ \dot M\ [M_\odot/yr]$", fontsize = 14)
ax.set_xlabel("mass [Msun]", fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.savefig("plots/samples/betavsmass_%s.pdf" % survey)


fig, ax = plt.subplots()
ax.errorbar(limenergy[:, 1], limlog10mdot[:, 1], xerr = [limenergy[:, 1] - limenergy[:, 0], limenergy[:, 2] - limenergy[:, 1]],
            yerr = [limlog10mdot[:, 1] - limlog10mdot[:, 0], limlog10mdot[:, 2] - limlog10mdot[:, 1]], lw = 0, elinewidth = 1, marker = 'o')

ax.set_ylabel(r"$\log_{10}\ \dot M\ [M_\odot/yr]$", fontsize = 14)
ax.set_xlabel("energy [B]", fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.savefig("plots/samples/log10mdotvsenergy_%s.pdf" % survey)

fig, ax = plt.subplots()
ax.errorbar(limz[:, 1], limbeta[:, 1], xerr = [limz[:, 1] - limz[:, 0], limz[:, 2] - limz[:, 1]],
            yerr = [limbeta[:, 1] - limbeta[:, 0], limbeta[:, 2] - limbeta[:, 1]], lw = 0, elinewidth = 1, marker = 'o')
ax.set_ylabel(r"$\beta$", fontsize = 14)
ax.set_xlabel("z", fontsize = 14)
ax.set_xlim(0, 0.55)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.savefig("plots/samples/betavsz_%s.pdf" % survey)

fig, ax = plt.subplots()
ax.errorbar(limz[:, 1], limmass[:, 1], xerr = [limz[:, 1] - limz[:, 0], limz[:, 2] - limz[:, 1]],
            yerr = [limmass[:, 1] - limmass[:, 0], limmass[:, 2] - limmass[:, 1]], lw = 0, elinewidth = 1, marker = 'o')
ax.set_ylabel(r"mass [$M_\odot$]", fontsize = 14)
ax.set_xlabel("z", fontsize = 14)
ax.set_xlim(0, 0.55)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.savefig("plots/samples/massvsz_%s.pdf" % survey)

fig, ax = plt.subplots()
ax.errorbar(limz[:, 1], limenergy[:, 1], xerr = [limz[:, 1] - limz[:, 0], limz[:, 2] - limz[:, 1]],
            yerr = [limenergy[:, 1] - limenergy[:, 0], limenergy[:, 2] - limenergy[:, 1]], lw = 0, elinewidth = 1, marker = 'o')
ax.set_ylabel(r"energy [B]", fontsize = 14)
ax.set_xlabel("z", fontsize = 14)
ax.set_xlim(0, 0.55)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.savefig("plots/samples/energyvsz_%s.pdf" % survey)


effsfile = "pickles/MoriyaWindAcc_HiTS15A-nf50-ne1-nr1-nn37_Blanco-DECam_gir_LCs_100000_effs.pkl"
xeffs, yeffs = pickle.load(open(effsfile, 'rb'))
effs = {}
for label in xeffs.keys():
    effs[label] = interp1d(xeffs[label], yeffs[label], bounds_error = False, fill_value = 'extrapolate')

def hist1D(allvals, medianvals, varname, xlabel, bw = None):
    fig, ax = plt.subplots()

    ## Silverman's rule
    #if bw == None:
    #    bw = 0.9 * min(np.std(medianvals), np.abs(np.percentile(medianvals, 75) - np.percentile(medianvals, 25)) / 1.349) * len(medianvals)**(-1./5.)
    #    print(varname, bw, np.median(medianvals), np.std(medianvals), np.median(np.abs(np.median(medianvals) - medianvals)))
    #kde = KernelDensity(kernel='gaussian', bandwidth = bw).fit(medianvals[:, np.newaxis])
    #if varname == "z":
    #    X_plot = np.linspace(0, 0.55, 1000)[:, np.newaxis]
    #else:
    #    X_plot = np.linspace(min(allvals) - 1.5 * bw, max(allvals) + 1.5 * bw, 1000)[:, np.newaxis]
    #log_dens = kde.score_samples(X_plot)
    #
    #ax.fill_between(X_plot[:, 0], 0, np.exp(log_dens), facecolor = 'b', alpha = 0.5)

    rdn = np.random.choice(allvals, size = 10000, replace = True)
    # Silverman's rule
    if bw == None:
        bw = 0.9 * min(np.std(rdn), np.abs(np.percentile(rdn, 75) - np.percentile(rdn, 25)) / 1.349) * len(rdn)**(-1./5.)
        print(varname, bw, np.median(rdn), np.std(rdn), np.median(np.abs(np.median(rdn) - rdn)))
    kde = KernelDensity(kernel='gaussian', bandwidth = bw).fit(rdn[:, np.newaxis])
    if varname == "z":
        X_plot = np.linspace(0, 0.55, 1000)[:, np.newaxis]
    else:
        X_plot = np.linspace(min(allvals) - 1.5 * bw, max(allvals) + 1.5 * bw, 1000)[:, np.newaxis]
    log_dens = kde.score_samples(X_plot)

    ax.fill_between(X_plot[:, 0], 0, np.exp(log_dens), facecolor = 'b', alpha = 0.5, label = "Observed")
    if varname == 'log10mdot' or varname == 'beta':
        factor = effs[varname](X_plot[:, 0])
        corrected = np.exp(log_dens) / factor
        corrected = corrected / (np.sum(corrected) * (X_plot[1, 0] - X_plot[0, 0]))
        print(varname, np.sum(corrected[X_plot[:, 0] < (X_plot[:, 0] - X_plot[:, 0]) / 2.]) / np.sum(corrected))
        ax.plot(X_plot[:, 0], corrected, c = 'k', alpha = 0.5, label = "Corrected")
        ax.legend(loc = 2)

    ax.set_xlabel(xlabel, fontsize = 14)
    ax.set_ylabel("p.d.f.", fontsize = 14)
    ax.set_ylim(0, ax.get_ylim()[1])
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    if varname[:4] == "texp":
        plt.xticks(fontsize = 14, rotation = 30)
        plt.ticklabel_format(useOffset=False)
    plt.tight_layout()
    plt.savefig("plots/samples/%s_hist_HiTS.pdf" % (varname))
    
hist1D(master['log10mdot'], limlog10mdot[:, 1], "log10mdot", r"$\log_{10}\ \dot M\ [M_\odot/yr]$")

hist1D(master['beta'], limbeta[:, 1], "beta", r"$\beta$")

hist1D(master['texp14'], limtexp14[:, 1], "texp14", r"$t_{\rm exp}$ HiTS14A [MJD]")

hist1D(master['texp15'], limtexp15[:, 1], "texp15", r"$t_{\rm exp}$ HiTS15A [MJD]")
  
hist1D(np.exp(master['logz']), limz[:, 1], "z", "z")

hist1D(np.exp(master['logAv']), limAv[:, 1], "Av", r"$A_{\rm V}$")

hist1D(master['mass'], limmass[:, 1], "mass", r"mass [$M_\odot$]")
       
hist1D(master['energy'], limenergy[:, 1], "energy", r"energy [B]")



    
