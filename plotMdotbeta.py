import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import os, sys

from constants import *
from LCz import *
from LCz_Av import *
from LCz_Av_params import *

from sklearn.neighbors.kde import KernelDensity


survey = sys.argv[1]
mode = sys.argv[2]

if mode == "LCs":
    doLCs = True
    doMCMC = False
elif mode == "MCMC":
    doLCs = False
    doMCMC = True

if survey == 'HiTS':
    HiTS = sorted(['SNHiTS14B', 'SNHiTS14N', 'SNHiTS14Q', 'SNHiTS14ac', 'SNHiTS15A', 'SNHiTS15D', 'SNHiTS15F', 'SNHiTS15K', 'SNHiTS15M', 'SNHiTS15P', 'SNHiTS15X', 'SNHiTS15ag', 'SNHiTS15ah', 'SNHiTS15ai', 'SNHiTS15ak', 'SNHiTS15aq', 'SNHiTS15as', 'SNHiTS15at', 'SNHiTS15aw', 'SNHiTS15ay', 'SNHiTS15az', 'SNHiTS15bc', 'SNHiTS15bl', 'SNHiTS15bm', 'SNHiTS15bn', 'SNHiTS15ch', 'SNHiTS14C', 'SNHiTS14D'])
    #HiTS = ["SNHiTS14B",
    #    "SNHiTS14N",
    #    "SNHiTS14Q",
    #    "SNHiTS14P",
    #    #"SNHiTS15aa", Ia?
    #    #"SNHiTS15ab", Ia?
    #    "SNHiTS15A",
    #    "SNHiTS15D",
    #    "SNHiTS15F",
    #    "SNHiTS15K",
    #    "SNHiTS15M",
    #    "SNHiTS15P",
    #    "SNHiTS15X",
    #    "SNHiTS15ag",
    #    "SNHiTS15ah",
    #    "SNHiTS15ai",
    #    #"SNHiTS15aj", Ia?
    #    "SNHiTS15ak",
    #    "SNHiTS15aq",
    #    #"SNHiTS15at", Ia?
    #    #"SNHiTS15av", Ia?
    #    "SNHiTS15aw",
    #    "SNHiTS15ay",
    #    "SNHiTS15az",
    #    #"SNHiTS15ba",
    #    #"SNHiTS15bb", Ia?
    #    "SNHiTS15bc",
    #    "SNHiTS15bl",
    #    "SNHiTS15bm",
    #    #"SNHiTS15bz", no usar
    #    "SNHiTS15ch"]
    #    #"SNHiTS15ci"]
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
    #print(files)
    
# Redshift, Avs and time
nz = 30
ntimes = 100
nAvs = 10
zs = np.logspace(-3, 0, nz)
times = np.logspace(-3, 3, ntimes)
Avs = np.logspace(-4, 1, nAvs)
Rv = 3.25


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
if doLCs:
    fig, ax = plt.subplots(nrows = 7, ncols = 4, figsize = (10, 12))
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
    counter = 0
print ms_sel

# iterate among samples files
files = sorted(os.listdir("samples/nodiff"))
model = "MoriyaWindAcc"
for f in files:

    SNfound = re.findall("chain_%s_(.*)_.*?.dat" % (model), f)
    if SNfound != []:
        SN = SNfound[0]
    else:
        continue
    
    print("Supernova:",  SN)

    if SN in HiTS or SN in DES:

        if SN in HiTS:
            m = ms_sel[np.mod(counter, len(ms_sel))]
            counter = counter + 1
        elif SN in DES:
            m = '*'
            
        if SN != "SNHiTS15X":
            continue
        
        print(f)

        png = f.replace("chain", "plots/nodiff/MCMC").replace(".dat", "_models.png")
        pdfout = "%s %s" % (pdfout, png)

        if re.search(".*logz.*", f):
            nchain, nwalker, scale, texp, logz, logAv, mass, energy, mdot, beta = np.loadtxt("samples/nodiff/%s" % f).transpose()
        else:
            nchain, nwalker, scale, texp, logAv, mass, energy, mdot, beta = np.loadtxt("samples/nodiff/%s" % f).transpose()

        mask = (nchain > 500)

        if SN in HiTS:

            
            (MJDs, MJDrefs, ADUs, e_ADUs, mags, e1_mags, e2_mags, sn_filters) \
                = np.loadtxt("/home/fforster/Work/HiTS/LCs/%s.txt" % SN, usecols = (0, 1, 5, 6, 7, 8, 9, 10), dtype = str).transpose()
       
            sn_mjd = np.array(MJDs, dtype = float)
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

            print SN, factorg, factorg

                
            if SN == "SNHiTS14B":
                #sn_mjdref = np.hstack([sn_mjdref, sn_mjdref[-1], sn_mjdref[-1]])
                sn_mjd = np.hstack([sn_mjd, sn_mjd[0] + 9.01680793, sn_mjd[0] + 23.85])
                sn_flux = np.hstack([sn_flux, mag2flux(22.34), mag2flux(22.9)])
                sn_e_flux = np.hstack([sn_e_flux, sn_e_flux[-1], sn_e_flux[-1]])
                sn_filters = np.hstack([sn_filters, 'g', 'g'])

            if SN == "SNHiTS14A":
                zcmb = 0.2175
                fixz = True
            elif SN == "SNHiTS14Y":
                zcmb = 0.108
                fixz = True
            elif SN == "SNHiTS14C":
                zcmb = 0.084
                fixz = True
            elif SN == "SNHiTS14D":
                zcmb = 0.135
                fixz = True
            elif SN == "SNHiTS15B":
                zcmb = 0.23
                fixz = True
            elif SN == "SNHiTS15J":
                zcmb = 0.108
                fixz = True
            elif SN == "SNHiTS15L":
                zcmb = 0.15
                fixz = True
            elif SN == "SNHiTS15O":
                zcmb = 0.142
                fixz = True
            elif SN == "SNHiTS15U":
                zcmb = 0.308
                fixz = True
            elif SN == "SNHiTS15X":
                zcmb = 0.055807
                fixz = True
            elif SN == "SNHiTS15ad":
                zcmb = 0.055392
                fixz = True
            elif SN == "SNHiTS15al":
                zcmb = 0.2
                fixz = True
            elif SN == "SNHiTS15aw":
                zcmb = 0.0663
                fixz = True
            elif SN == "SNHiTS15be":
                zcmb = 0.151
                fixz = True
            elif SN == "SNHiTS15bs":
                zcmb = 0.07
                fixz = True
            elif SN == "SNHiTS15by":
                zcmb = 0.0524
                fixz = True
            elif SN == 'SNHiTS15ck':
                zcmb = 0.042
                fixz = True
            else:
                zcmb = 0.2
                fixz = False

        # prepare light curves
        if doLCs:
            
            # set observations
            LCs.set_observations(mjd = sn_mjd, flux = sn_flux, e_flux = sn_e_flux, filters = sn_filters, objname = SN, plot = False, bandcolors = {'g': 'g', 'r': 'r', 'i': 'brown', 'z': 'k'})

            # get array positions
            (iy, ix) = int(np.mod(counter - 1, 4)), int((counter - 1) / 4)

            # plot light curves
            for band in LCs.uniquefilters:
                maskb = LCs.maskband[band]
                if np.sum(maskb) > 0:
                    ax[ix, iy].errorbar(LCs.mjd[maskb], LCs.flux[maskb], yerr = LCs.e_flux[maskb], marker = 'o', c = LCs.bandcolors[band], lw = 0, elinewidth = 1, markersize = 5, alpha = 0.7)
                    ax[ix, iy].set_xlim(min(min(texp[mask]), min(LCs.mjd)) - 1, max(LCs.mjd) + 3)
            
            # plot light curves sampled from the posterior
            nplots = 100
            idxselected = np.array(np.random.random(size = nplots) * np.sum(mask), dtype = int)
            for idxsel in idxselected:

                scale_val = scale[mask][idxsel]
                texp_val = texp[mask][idxsel]
                if fixz:
                    logz_val = np.log(zcmb)
                else:
                    logz_val = logz[mask][idxsel]
                logAv_val = logAv[mask][idxsel]
                mass_val = mass[mask][idxsel]
                energy_val = energy[mask][idxsel]
                mdot_val = mdot[mask][idxsel]
                beta_val = beta[mask][idxsel]
                rcsm_val = 1. # 1e15
                vwindinf_val = 10.

                # values
                parvals = np.array([scale_val, texp_val, logz_val, logAv_val, mass_val, energy_val, mdot_val, rcsm_val, vwindinf_val, beta_val])
                parbounds = np.array([[0.1, 10.], [texp_val - 5, texp_val + 5], [np.log(1e-4), np.log(10.)], [np.log(1e-4), np.log(10.)], [12, 16], [0.5, 2.], [1e-6, 1e-2], [1., 1.], [10, 10], [1., 5.]])
                parlabels = np.array(["scale", "texp", "logz", "logAv", "mass", "energy", "mdot", "rcsm", "vwindinf", "beta"])
                fixedvars = np.array([False,     False,  fixz,   False,   False,   False,    False,   True,   True,      False], dtype = bool)  # rcsm and vwinf should be True with current model grid
            
                # initialize with previous parameters
                theta0 = parvals[np.invert(fixedvars)]
                sol = LCs.findbest(theta0 = theta0, parbounds = parbounds, fixedvars = fixedvars, parvals = parvals, parlabels = parlabels, skip = True)

                LCmag, LCmagref = LCs.evalmodel(scale_val, texp_val, logz_val, logAv_val, LCs.parvals[4:], True, False)

                # plot light curves
                for band in LCs.uniquefilters:
                    maskb = LCs.maskband[band]
                    if np.sum(maskb) > 0:
                        ax[ix, iy].plot(LCs.times + texp_val, mag2flux(LCmag[band]), label = "%s" % band, c = LCs.bandcolors[band], lw = 1, alpha = 0.05)
                        ax[ix, iy].axvline(texp_val, alpha = 0.05, c = 'gray')
            # labels                    
            #ax[ix, iy].set_yticklabels([])
            (x1, x2) = ax[ix, iy].get_xlim()
            (y1, y2) = ax[ix, iy].get_ylim()
            ax[ix, iy].text(x2, y1 + (y2 - y1) * 0.05, SN, fontsize = 10, ha = 'right')
            
            
        if not doMCMC:
            continue
        
        if "mdot" not in master.keys():
            master['mdot'] = mdot[mask]
            master['beta'] = beta[mask]
            master['mass'] = mass[mask]
            master['energy'] = energy[mask]
            master['logAv'] = logAv[mask]
            if not fixz:
                master['logz'] = logz[mask]
        else:
            master['mdot'] = np.hstack([master['mdot'], mdot[mask]])
            master['beta'] = np.hstack([master['beta'], beta[mask]])
            master['mass'] = np.hstack([master['mass'], mass[mask]])
            master['energy'] = np.hstack([master['energy'], energy[mask]])
            master['logAv'] = np.hstack([master['logAv'], logAv[mask]])
            if not fixz:
                master['logz'] = np.hstack([master['logz'], logz[mask]])

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

        limlog10mdot.append(np.log10(np.array(map(lambda x: np.percentile(mdot[mask], x), [5, 50, 95]))))
        limbeta.append(np.array(map(lambda x: np.percentile(beta[mask], x), [5, 50, 95])))
        limmass.append(np.array(map(lambda x: np.percentile(mass[mask], x), [5, 50, 95])))
        limenergy.append(np.array(map(lambda x: np.percentile(energy[mask], x), [5, 50, 95])))
        limAv.append(np.exp(np.array(map(lambda x: np.percentile(logAv[mask], x), [5, 50, 95]))))
        if fixz:
            limz.append(zcmb * np.ones(3))
        else:
            limz.append(np.exp(np.array(map(lambda x: np.percentile(logz[mask], x), [5, 50, 95]))))
        print(SN, np.exp(np.array(map(lambda x: np.percentile(logz[mask], x), [5, 50, 95]))))
        if SN[:8] == 'SNHiTS14':
            limtexp14.append(np.array(map(lambda x: np.percentile(texp[mask], x), [5, 50, 95])))
        elif SN[:8] == 'SNHiTS15':
            limtexp15.append(np.array(map(lambda x: np.percentile(texp[mask], x), [5, 50, 95])))

        xerr = [[limlog10mdot[-1][1] - limlog10mdot[-1][0]], [limlog10mdot[-1][2] - limlog10mdot[-1][1]]]
        yerr = [[limbeta[-1][1] - limbeta[-1][0]], [limbeta[-1][2] - limbeta[-1][1]]]
        
        ax.errorbar(limlog10mdot[-1][1], limbeta[-1][1], marker = m, markersize = 10, xerr = xerr, yerr = yerr, label = SN, alpha = 0.8)

if doLCs:
    ax[5, 3].axis('off')#set_yticklabels([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.15)
    plt.savefig("plots/samples/LCs.png")
        
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
    
os.system("convert %s plots/samples/%s.pdf" % (pdfout, survey))

plt.legend(loc = 3, fontsize = 11)  
ax.set_xlim(-6, -2)
ax.set_ylim(1, 5)
ax.set_xlabel(r"$\log_{10}\ \dot M\ [M_\odot/yr]$", fontsize = 14)
ax.set_ylabel(r"$\beta$", fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.savefig("plots/samples/betavsmdot_%s.png" % survey)

fig, ax = plt.subplots()
ax.hist2d(master['mass'], master['energy'])
ax.set_xlabel("mass [Msun]")
ax.set_ylabel("energy [B]")
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.savefig("plots/samples/energyvsmass_hist_%s.png" % survey)

fig, ax = plt.subplots()
ax.errorbar(limmass[:, 1], limenergy[:, 1], xerr = [limmass[:, 1] - limmass[:, 0], limmass[:, 2] - limmass[:, 1]],
            yerr = [limenergy[:, 1] - limenergy[:, 0], limenergy[:, 2] - limenergy[:, 1]], lw = 0, elinewidth = 1, marker = 'o')
ax.set_xlabel("mass [Msun]", fontsize = 14)
ax.set_ylabel("energy [B]", fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.savefig("plots/samples/energyvsmass_%s.png" % survey)

fig, ax = plt.subplots()
ax.errorbar(limz[:, 1], limlog10mdot[:, 1], xerr = [limz[:, 1] - limz[:, 0], limz[:, 2] - limz[:, 1]],
            yerr = [limlog10mdot[:, 1] - limlog10mdot[:, 0], limlog10mdot[:, 2] - limlog10mdot[:, 1]], lw = 0, elinewidth = 1, marker = 'o')
ax.set_ylabel(r"$\log_{10}\ \dot M\ [M_\odot/yr]$", fontsize = 14)
ax.set_xlabel("z", fontsize = 14)
ax.set_xlim(0, 0.55)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.savefig("plots/samples/log10mdotvsz_%s.png" % survey)

fig, ax = plt.subplots()
ax.errorbar(limz[:, 1], limbeta[:, 1], xerr = [limz[:, 1] - limz[:, 0], limz[:, 2] - limz[:, 1]],
            yerr = [limbeta[:, 1] - limbeta[:, 0], limbeta[:, 2] - limbeta[:, 1]], lw = 0, elinewidth = 1, marker = 'o')
ax.set_ylabel(r"$\beta$", fontsize = 14)
ax.set_xlabel("z", fontsize = 14)
ax.set_xlim(0, 0.55)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.savefig("plots/samples/betavsz_%s.png" % survey)

fig, ax = plt.subplots()
ax.errorbar(limz[:, 1], limmass[:, 1], xerr = [limz[:, 1] - limz[:, 0], limz[:, 2] - limz[:, 1]],
            yerr = [limmass[:, 1] - limmass[:, 0], limmass[:, 2] - limmass[:, 1]], lw = 0, elinewidth = 1, marker = 'o')
ax.set_ylabel(r"mass [$M_\odot$]", fontsize = 14)
ax.set_xlabel("z", fontsize = 14)
ax.set_xlim(0, 0.55)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.savefig("plots/samples/massvsz_%s.png" % survey)

fig, ax = plt.subplots()
ax.errorbar(limz[:, 1], limenergy[:, 1], xerr = [limz[:, 1] - limz[:, 0], limz[:, 2] - limz[:, 1]],
            yerr = [limenergy[:, 1] - limenergy[:, 0], limenergy[:, 2] - limenergy[:, 1]], lw = 0, elinewidth = 1, marker = 'o')
ax.set_ylabel(r"energy [B]", fontsize = 14)
ax.set_xlabel("z", fontsize = 14)
ax.set_xlim(0, 0.55)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.savefig("plots/samples/energyvsz_%s.png" % survey)


def hist1D(allvals, medianvals, varname, xlabel, bw = None):
    fig, ax = plt.subplots()
    #ax.hist(allvals, normed = True, alpha = 0.5)
    #ax.hist(medianvals, normed = True, color = 'r', alpha = 0.3)

    # Silverman's rule
    if bw == None:
        bw = 0.9 * min(np.std(medianvals), np.abs(np.percentile(medianvals, 75) - np.percentile(medianvals, 25)) / 1.349) * len(medianvals)**(-1./5.)
        print varname, bw
    kde = KernelDensity(kernel='gaussian', bandwidth = bw).fit(medianvals[:, np.newaxis])
    if varname == "z":
        X_plot = np.linspace(0, 0.55, 1000)[:, np.newaxis]
    else:
        X_plot = np.linspace(min(allvals) - 1.5 * bw, max(allvals) + 1.5 * bw, 1000)[:, np.newaxis]
    log_dens = kde.score_samples(X_plot)
    ax.fill_between(X_plot[:, 0], 0, np.exp(log_dens), facecolor = 'b', alpha = 0.5)
    ax.set_xlabel(xlabel, fontsize = 14)
    ax.set_ylabel("p.d.f.", fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    if varname[:4] == "texp":
        print "Time"
        plt.xticks(fontsize = 14, rotation = 30)
        plt.ticklabel_format(useOffset=False)
    plt.tight_layout()
    plt.savefig("plots/samples/%s_hist_%s.png" % (varname, survey))

hist1D(np.log10(master['mdot']), limlog10mdot[:, 1], "log10mdot", r"$\log_{10}\ \dot M\ [M_\odot/yr]$")

hist1D(master['beta'], limbeta[:, 1], "beta", r"$\beta$")

hist1D(master['texp14'], limtexp14[:, 1], "texp14", r"$t_{\rm exp}$ HiTS14A [MJD]")

hist1D(master['texp15'], limtexp15[:, 1], "texp15", r"$t_{\rm exp}$ HiTS15A [MJD]")
  
hist1D(np.exp(master['logz']), limz[:, 1], "z", "z")

hist1D(np.exp(master['logAv']), limAv[:, 1], "Av", r"$A_{\rm V}$")

hist1D(master['mass'], limmass[:, 1], "mass", r"mass [$M_\odot$]")
       
hist1D(master['energy'], limenergy[:, 1], "energy", r"energy [B]")



    
