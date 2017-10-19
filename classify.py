import os, sys, re
import numpy as np
import pandas as pd
# for detecing when in the leftraru
leftraru = False
if os.getcwd() == "/home/fforster/surveysim":
    leftraru = True
    import matplotlib # uncomment for using in leftraru
    matplotlib.use('Agg') # uncomment for using in leftraru
import matplotlib.pyplot as plt

from readSNdata import *

lnlikedir = "lnlikes"
files = os.listdir(lnlikedir)

Moriya = {}
Hsiao = {}

for f in sorted(files):

    if not re.search("MCMC_(.*?)_(SNHiTS.*?)_.*.npy", f):
        continue
    
    print(f)
    
    model, SN = re.findall("MCMC_(.*?)_(SNHiTS.*?)_.*.npy", f)[0]

    #print("Found SN %s (model %s)" % (SN, model))
    
    if model == "MoriyaWindAcc":
        Moriya[SN] = np.load("%s/%s" % (lnlikedir, f))
    elif model == "Hsiao":
        Hsiao[SN] = np.load("%s/%s" % (lnlikedir, f))

print(Moriya.keys())

fig, ax = plt.subplots(figsize = (14, 10))
SNII = []
SNIa = []



spectra = {}
spectra["SNHiTS14B"] = "II" # Bel, http://www.astronomerstelegram.org/?read=6014
spectra["SNHiTS14C"] = "II" # Greta, http://www.astronomerstelegram.org/?read=5957
spectra["SNHiTS14D"] = "II" # Emilia, blue continuum, http://www.astronomerstelegram.org/?read=5957
spectra["SNHiTS14F"] = "Ia" # Mara, http://www.astronomerstelegram.org/?read=6014
spectra["SNHiTS14H"] = "Ia" # Pamela, http://www.astronomerstelegram.org/?read=6014, http://www.astronomerstelegram.org/?read=5970
spectra["SNHiTS15D"] = "II" # Daniela, http://www.astronomerstelegram.org/?read=7162
spectra["SNHiTS15I"] = "Ia" # Olga-Lucia, http://www.astronomerstelegram.org/?read=7154
spectra["SNHiTS15J"] = "Ia" # Teahine, http://www.astronomerstelegram.org/?read=7154
spectra["SNHiTS15L"] = "Ia" # Natalia, http://www.astronomerstelegram.org/?read=7144
spectra["SNHiTS15P"] = "II" # Rosemary, http://www.astronomerstelegram.org/?read=7162
spectra["SNHiTS15ad"] = "Ia" # Gabriela, http://www.astronomerstelegram.org/?read=7164
spectra["SNHiTS15aw"] = "II" # Maria Soledad, http://www.astronomerstelegram.org/?read=7246
spectra["SNHiTS15al"] = "Ia" # Goretti, http://www.astronomerstelegram.org/?read=7291
spectra["SNHiTS15be"] = "Ia" # Agustina, http://www.astronomerstelegram.org/?read=7291
spectra["SNHiTS15bs"] = "Ia" # Rita, http://www.astronomerstelegram.org/?read=7291
spectra["SNHiTS15bu"] = "Ia" # Ane, http://www.astronomerstelegram.org/?read=7335
spectra["SNHiTS15by"] = "II" # Tanit, PS15ou, http://www.astronomerstelegram.org/?read=7291
spectra["SNHiTS15cf"] = "Ia" # Nines, http://www.astronomerstelegram.org/?read=7335


HiTS = ["SNHiTS14B",
        "SNHiTS14P",
        "SNHiTS14N",
        "SNHiTS14Q",
        "SNHiTS14ac",
        "SNHiTS15A",
        "SNHiTS15D",
        "SNHiTS15F",
        "SNHiTS15K",
        "SNHiTS15M",
        "SNHiTS15P",
        "SNHiTS15X",
        "SNHiTS15ag",
        "SNHiTS15ah",
        "SNHiTS15ai",
        "SNHiTS15ak",
        "SNHiTS15aq",
        "SNHiTS15as",
        "SNHiTS15aw",
        "SNHiTS15ay",
        "SNHiTS15az",
        "SNHiTS15bc",
        "SNHiTS15bl",
        "SNHiTS15bm",
        "SNHiTS15ch"]

        
banned = ["SNHiTS14K", # bad LC
          "SNHiTS14U", # same as 14T
          "SNHiTS14ah", # too noisy LC
          "SNHiTS15B", # other class
          "SNHiTS15ap", # same as 15ao
          "SNHiTS15ae", # too noisy LC
          "SNHiTS15bd", # noisy light curve
          "SNHiTS15bi", # too noisy LC
          "SNHiTS15bn", # same as bm
          "SNHiTS15bo", # same as bm
          "SNHiTS15ca", # same as 15bz
          "SNHiTS15cj"] # same as 15cb

# other SNe:
## 15A -> 3.0, 2.8, 2.6
# 15D -> 2.3, 2.2, 1.4
## 15F -> 3.1, 2.9, 2.7
# 15K -> 2.0, 1.9, 1.5
## 15as -> 3.1, 2.9, 2.7

# prob of missing 1st 3 days of rise < 10%

poor_rise = ["SNHiTS14M", # >5.7 (5.2, 4.5) days from emergence 5% prob.
             #"SNHiTS14P", # >1.7 (1.2, 0.6) days from emergence 5% prob.
             #"SNHiTS14W", # >2.5 (2.0, 0.8) days from emergence 5% prob.
             #"SNHiTS15G", # >0.7 days from emergence 5% prob.
             "SNHiTS15O", # >3.8 (3.5, 2.9) days from emergence 5% (20) prob. 
             "SNHiTS15bj", # > 1 night gap during initial rise
             "SNHiTS15bl", # > 1 night gap during initial rise
             "SNHiTS15bt", # > 1 night gap during initial rise
             "SNHiTS15by", # > 1 night gap during initial rise (with spectrum)
             "SNHiTS15ck"] # > 1 night gap during initial rise

short = []
# remove SNe with short time spans
for SN in Moriya.keys():
    sn_mjd, sn_mjdref, sn_flux, sn_e_flux, sn_filters, fixz, zcmb, texp0 = readSNdata("HiTS", SN)
    dt = max(sn_mjd) - min(sn_mjd)
    if dt < 7:
        print("Removed %s due to small time span (%i days)" % (SN, dt))
        banned.append(SN)
        #short.append(SN)

BICII = {}
BICIa = {}
for SN in sorted(Moriya.keys()):

    sn_mjd, sn_mjdref, sn_flux, sn_e_flux, sn_filters, fixz, zcmb, texp0 = readSNdata("HiTS", SN)
      
    nHsiao = 4 # scale, texp, Av, stretch
    nMoriya = 7 # scale, texp, Av, mass, energy, mdot, beta
    if not fixz:
        nHsiao += 1  # redshift
        nMoriya += 1 # redshift

    if SN in Hsiao.keys():

        x = np.log10(nMoriya * (np.log(len(sn_mjd)) - np.log(2. * np.pi)) - 2. * np.median(Moriya[SN]))    
        BICII[SN] = 10**x
            
        y = np.log10(nHsiao * (np.log(len(sn_mjd)) - np.log(2. * np.pi)) - 2. * np.median(Hsiao[SN]))
        BICIa[SN] = 10**y

        if (SN in banned or SN in poor_rise):# and SN not in spectra:
            continue
      
        if (x < y):
            print("---------> SN Moriya best fit: %s" % SN)
        #yerr = np.abs(np.array([[np.arcsinh(np.median(Hsiao[SN])) - np.arcsinh(np.percentile(Hsiao[SN], 5)), np.arcsinh(np.percentile(Hsiao[SN], 95)) - np.arcsinh(np.median(Hsiao[SN]))]]))

        print "SN: %s, n_M: %i, n_H: %i, AIC_M: %f, AIC_H: %f" % (SN, nMoriya, nHsiao, 10**x, 10**y)

        marker = 'o'
        arcsinhscale = 1e-2
        delta = np.arcsinh((y - x) / arcsinhscale)
        delta = np.arcsinh(10**y - 10**x)
        #ax.text(np.minimum(x, y), delta, SN[6:], fontsize = 6)
        if SN in spectra.keys():
            if spectra[SN] == "II" and SN not in banned:
                SNII.append(SN)
                if SN not in HiTS:
                    print("Adding %s with Type II spectroscopic classification" % SN)
                    HiTS.append(SN)
            elif spectra[SN] == "Ia" and SN not in banned:
                SNIa.append(SN)
        elif y - x > 0:#np.sqrt(xerr[0][0]**2 + yerr[0][0]**2):
            SNII.append(SN)
        else:#np.sqrt(xerr[0][0]**2 + yerr[0][0]**2):
            SNIa.append(SN)

        color = 'gray'
        lw = 0
        if SN in spectra.keys():
            print("      Supernova %s has spectra (%s)" % (SN, spectra[SN]))
            if spectra[SN] == "II":
                color = 'r'
            elif spectra[SN] == "Ia":
                color = 'b'

        if SN in poor_rise or (SN in banned and SN in spectra):
            mew = 2.
        else:
            mew = 0

        #print("Plotting %s" % SN)
        ax.errorbar(np.minimum(x, y), delta, marker = 'o', markersize = 20, alpha = 0.5, mew = mew, mec = 'k', color = color, lw = 0)#, xerr = xerr, yerr = np.sqrt(yerr**2 + xerr**2))

        

ax.axhline(0, ls = ':', c = 'gray')
#plt.grid()
ax.set_xlabel(r"$\log_{10}\ BIC_{\rm best}$", fontsize = 14)
ax.set_ylabel(r"${\rm arcsinh}(BIC_{\rm Ia}\ -\ BIC_{\rm II})$", fontsize = 14)

(x1, x2) = ax.get_xlim()
(y1, y2) = ax.get_ylim()
ax.errorbar(x1 + (x2 - x1) * 0.8, y1 + (y2 - y1) * 0.95, marker = "o", markersize = 20, color = 'gray', mew = 0, alpha = 0.5)
ax.text(x1 + (x2 - x1) * 0.82, y1 + (y2 - y1) * 0.95, "No spectrum available", va = 'center', fontsize = 14)
ax.errorbar(x1 + (x2 - x1) * 0.8, y1 + (y2 - y1) * 0.9, marker = "o", markersize = 20, color = 'r', mew = 0, alpha = 0.5)
ax.text(x1 + (x2 - x1) * 0.82, y1 + (y2 - y1) * 0.9, "Type II spectrum", va = 'center', fontsize = 14)
ax.errorbar(x1 + (x2 - x1) * 0.8, y1 + (y2 - y1) * 0.85, marker = "o", markersize = 20, color = 'b', mew = 0, alpha = 0.5)
ax.text(x1 + (x2 - x1) * 0.82, y1 + (y2 - y1) * 0.85, "Type Ia spectrum", va = 'center', fontsize = 14)
ax.errorbar(x1 + (x2 - x1) * 0.8, y1 + (y2 - y1) * 0.8, marker = "o", markersize = 20, color = 'white', mec = 'k', mew = 2., alpha = 0.5)
ax.text(x1 + (x2 - x1) * 0.82, y1 + (y2 - y1) * 0.8, "Poor sampling", va = 'center', fontsize = 14)

ax.arrow(x1 + (x2 - x1) * 0.7, (y2 - y1) * 0.005, 0, (y2 - y1) * 0.05, head_length = (y2 - y1) * 0.01, head_width = (x2 - x1) * 0.01, color = 'k')
ax.text(x1 + (x2 - x1) * 0.71, (y2 - y1) * 0.04, "SN II photometric classification", fontsize = 14)
#ax.arrow(x1 + (x2 - x1) * 0.7, -(y2 - y1) * 0.005, 0, -(y2 - y1) * 0.05, head_length = (y2 - y1) * 0.01, head_width = (x2 - x1) * 0.01, color = 'k')
#ax.text(x1 + (x2 - x1) * 0.71, -(y2 - y1) * 0.04, "SN Ia photometric classification", fontsize = 14)

plt.tight_layout()
plt.legend(loc = 2)
plt.savefig("plots/classification.png")
plt.savefig("plots/classification.pdf")
#plt.show()

# save SNe II, SNe Ia
fileout = open("HiTS_classification.out", "w")
fileout.write("# SNe BICII BICIa spec_class banned poor_rise\n")
for SN in sorted(Moriya.keys()): 
    b = False
    p = False
    c = "NA"
    if SN in banned:
        b = True
    if SN in poor_rise:
        p = True
    if SN in spectra.keys():
        c = spectra[SN]
    fileout.write("%s %.2f %.2f %s %s %s\n" % (SN, BICII[SN], BICIa[SN], c, b, p))
fileout.close()


print("HiTS", len(HiTS), sorted(HiTS))
print("SNII", len(SNII), sorted(SNII))
print("SNIa", len(SNIa), sorted(SNIa))

s = "feh"
print("\n\nAll SNe classified as SNe II")
for SN in sorted(SNII):
    s = "%s plots/*Moriya*%s*models.png" % (s, SN)
print(s)
if not leftraru:
    os.system(s)
    
s = "feh"
print("\n\nSNe in HiTS not classified as SNe II")
for SN in HiTS:
    if SN not in SNII and SN not in banned:
        print(SN)
        s = "%s plots/*Moriya*%s*models.png" % (s, SN)
        s = "%s plots/*Hsiao*%s*models.png" % (s, SN)
if s != "feh":
    print(s)
    if not leftraru:
        os.system(s)

print()
s = "feh"
print("\n\nSNe classified as SNe II not in HiTS")
for SN in SNII:
    if SN not in HiTS and SN not in banned:
        print(SN)
        s = "%s plots/*Moriya*%s*models.png" % (s, SN)
        s = "%s plots/*Hsiao*%s*models.png" % (s, SN)
if s != "feh":
    print(s)
    if not leftraru:
        os.system(s)

print("SNe classified as SNe II with spectra")
for SN in SNII:
    if SN not in banned and SN in spectra:
        print SN

# check SNII sample
#fig, ax = plt.subplots(ncols = 3, figsize = (20, 6))
#for SN in SNII:
#
#    sn_mjd, sn_mjdref, sn_flux, sn_e_flux, sn_filters, fixz, zcmb, texp0 = readSNdata("HiTS", SN)
#
#    for chain in os.listdir("samples"):
#        if SN in chain and "Moriya" in chain:
#            print(chain)
#
#    
#            if 'logz' in chain:
#                nchain, nwalker, scale, texp, logz, logAv, mass, energy, log10mdot, beta = np.loadtxt("samples/%s" % chain).transpose()
#            else:
#                nchain, nwalker, scale, texp, logAv, mass, energy, log10mdot, beta = np.loadtxt("samples/%s" % chain).transpose()
#
#            t5, t50, t95, tmin, mdot = np.percentile(texp, 5), np.percentile(texp, 50), np.percentile(texp, 95), min(sn_mjd), np.percentile(log10mdot, 50)
#            print(t95 - t5, tmin - t5, tmin - t50, mdot)
#            ax[0].scatter(t95 - t5, mdot)
#            ax[0].text(t95 - t5, mdot, SN)
#            ax[1].scatter(tmin - t5, mdot)
#            ax[1].text(tmin - t5, mdot, SN)
#            ax[2].scatter(tmin - t50, mdot)
#            ax[2].text(tmin - t50, mdot, SN)
#plt.show()
#s = "feh"
#for SN in SNII:
#    s = "%s plots/*Moriya*%s*models.png" % (s, SN)
#os.system(s)


#s = "feh"
#for SN in SNIa:
#    s = "%s plots/*Hsiao*%s*corner.png" % (s, SN)
#print(s)

