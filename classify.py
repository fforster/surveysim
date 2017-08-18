import os, sys, re
import numpy as np
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
Moriyanodiff = {}
Hsiao = {}

for f in sorted(files):

    if not re.search("MCMC_(.*?)_(.*?)_.*.npy", f):
        continue
    
    print(f)
    
    model, SN = re.findall("MCMC_(.*?)_(.*?)_.*.npy", f)[0]

    #print("Found SN %s (model %s)" % (SN, model))
    
    if model == "MoriyaWindAcc":
        Moriya[SN] = np.load("%s/%s" % (lnlikedir, f))
        Moriyanodiff[SN] = np.load(("%s/%s" % (lnlikedir, f)).replace("lnlikes", "lnlikes/nodiff"))
    elif model == "Hsiao":
        Hsiao[SN] = np.load("%s/%s" % (lnlikedir, f))

print(Moriya.keys())

fig, ax = plt.subplots(figsize = (14, 10))
SNII = []
SNIa = []

spectra = {}
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
          "SNHiTS14ah", # bad LC
          "SNHiTS15B", # other class
          "SNHiTS15ap", # same as 15ao
          "SNHiTS15ae", # noisy LC
          "SNHiTS15au", # no information during rise according to SNII best fit
          "SNHiTS15bl", # rise after high cadence
          "SNHiTS15bn", # same as bm
          "SNHiTS15bo", # same as bm
          "SNHiTS15br", # too many days between 1st detection and last non detection
          "SNHiTS15bt", # too many days between 1st detection and last non detection
          "SNHiTS15by", # too many days between 1st detection and last non detection (with spectrum)
          "SNHiTS15ca", # same as 15bz
          "SNHiTS15cg", # no information during rise
          "SNHiTS15cj"] # same as 15cb

# remove SNe with short time spans
for SN in Moriya.keys():
    sn_mjd, sn_mjdref, sn_flux, sn_e_flux, sn_filters, fixz, zcmb, texp0 = readSNdata("HiTS", SN)
    dt = max(sn_mjd) - min(sn_mjd)
    if dt < 7:
        print("Removed %s due to small time span (%i days)" % (SN, dt))
        banned.append(SN)

nodiff = {}
BICIIdiff = {}
BICIInodiff = {}
BICIa = {}
for SN in sorted(Moriya.keys()):

    sn_mjd, sn_mjdref, sn_flux, sn_e_flux, sn_filters, fixz, zcmb, texp0 = readSNdata("HiTS", SN)
      
    nHsiao = 4 # texp, Av, stretch, scale
    nMoriya = 6 # texp, Av, mass, energy, mdot, beta
    if not fixz:
        nHsiao += 1  # redshift
        nMoriya += 1 # redshift

    if SN in Hsiao.keys():

        x = np.log10(nMoriya * (np.log(len(sn_mjd)) - np.log(2. * np.pi)) - 2. * np.median(Moriya[SN]))    
        xnodiff = np.log10(nMoriya * (np.log(len(sn_mjd)) - np.log(2. * np.pi)) - 2. * np.median(Moriyanodiff[SN]))
        BICIIdiff[SN] = 10**x
        BICIInodiff[SN] = 10**xnodiff
        donodiff = False
        if xnodiff < x and donodiff:
            nodiff[SN] = True
            x = xnodiff
            #xerr = np.abs(np.array([[np.arcsinh(np.median(Moriya[SN])) - np.arcsinh(np.percentile(Moriya[SN], 5)), np.arcsinh(np.percentile(Moriya[SN], 95)) - np.arcsinh(np.median(Moriya[SN]))]]))
        else:
            nodiff[SN] = False
            #xerr = np.abs(np.array([[np.arcsinh(np.median(Moriyanodiff[SN])) - np.arcsinh(np.percentile(Moriyanodiff[SN], 5)), np.arcsinh(np.percentile(Moriyanodiff[SN], 95)) - np.arcsinh(np.median(Moriyanodiff[SN]))]]))
            
        y = np.log10(nHsiao * (np.log(len(sn_mjd)) - np.log(2. * np.pi)) - 2. * np.median(Hsiao[SN]))
        BICIa[SN] = 10**y

        if SN in banned:
            if SN in spectra.keys():
                if spectra[SN] != 'II':
                    continue
            else:
                continue
      
        if (x < y) and not nodiff:
            print("---------> SN Moriya best fit with no diffLC: %s" % SN)
        #yerr = np.abs(np.array([[np.arcsinh(np.median(Hsiao[SN])) - np.arcsinh(np.percentile(Hsiao[SN], 5)), np.arcsinh(np.percentile(Hsiao[SN], 95)) - np.arcsinh(np.median(Hsiao[SN]))]]))

        print "SN: %s, n_M: %i, n_H: %i, AIC_M: %f, AIC_Mnodiff: %f, AIC_H: %f" % (SN, nMoriya, nHsiao, 10**x, 10**xnodiff, 10**y)

        marker = 'o'
        arcsinhscale = 1e-2
        delta = np.arcsinh((y - x) / arcsinhscale)
        delta = np.arcsinh(10**y - 10**x)
        ax.text(np.minimum(x, y), delta, SN[6:], fontsize = 6)
        if SN in spectra.keys():
            if spectra[SN] == "II":
                SNII.append(SN)
                if SN not in HiTS:
                    print("Adding %s with Type II spectroscopic classification" % SN)
                    HiTS.append(SN)
                marker = '*'
            elif spectra[SN] == "Ia":
                SNIa.append(SN)
                marker = 's'           
        elif y - x > 0:#np.sqrt(xerr[0][0]**2 + yerr[0][0]**2):
            SNII.append(SN)
            marker = '*'
        else:#np.sqrt(xerr[0][0]**2 + yerr[0][0]**2):
            SNIa.append(SN)
            marker = 's'

        color = 'gray'
        lw = 0
        if SN in spectra.keys():
            print("      Supernova %s has spectra (%s)" % (SN, spectra[SN]))
            if spectra[SN] == "II":
                color = 'r'
            elif spectra[SN] == "Ia":
                color = 'b'

        if marker == '*':
            s = 30
        else:
            s = 15
        #print("Plotting %s" % SN)
        ax.errorbar(np.minimum(x, y), delta, marker = marker, markersize = s, alpha = 0.5, color = color)#, xerr = xerr, yerr = np.sqrt(yerr**2 + xerr**2))

            #if not nodiff and SN in HiTS:
            #    print(SN)


ax.axhline(0)
plt.grid()
ax.set_xlabel(r"$\log_{10}\ BIC_{\rm best}$", fontsize = 14)
ax.set_ylabel(r"${\rm arcsinh}(BIC_{\rm Ia}\ -\ BIC_{\rm II})$", fontsize = 14)
plt.savefig("plots/classification.png")
#plt.show()

# save SNe II, SNe Ia and no diff
fileout = open("HiTS_classification.out", "w")
fileout.write("# SNe BICIIdiff BICIInodiff BICIa spec_class banned\n")
for SN in sorted(Moriya.keys()): 
    b = False
    c = "NA"
    if SN in banned:
        b = True
    if SN in spectra.keys():
        c = spectra[SN]
    fileout.write("%s %.2f %.2f %.2f %s %s\n" % (SN, BICIIdiff[SN], BICIInodiff[SN], BICIa[SN], c, b))
fileout.close()


print("HiTS", len(HiTS), sorted(HiTS))
print("SNII", len(SNII), sorted(SNII))
print("SNIa", len(SNIa), sorted(SNIa))

s = "feh"
print("All SNe classified as SNe II")
for SN in sorted(SNII):
    if nodiff[SN]:
        s = "%s plots/nodiff/*Moriya*%s*models.png" % (s, SN)
    else:
        s = "%s plots/*Moriya*%s*models.png" % (s, SN)
    #s = "%s plots/nodiff/*Moriya*%s*evol.png" % (s, SN)
    #s = "%s plots/nodiff/*Moriya*%s*corner.png" % (s, SN)
print(s)
if not leftraru:
    os.system(s)
    
s = "feh"
print("SNe in HiTS not classified as SNe II")
for SN in HiTS:
    if SN not in SNII and SN not in banned:
        print(SN)
        s = "%s plots/*Moriya*%s*models.png" % (s, SN)
        s = "%s plots/nodiff/*Moriya*%s*models.png" % (s, SN)
        s = "%s plots/*Hsiao*%s*models.png" % (s, SN)
if s != "feh":
    print(s)
    if not leftraru:
        os.system(s)

print()
s = "feh"
print("SNe classified as SNe II not in HiTS")
for SN in SNII:
    if SN not in HiTS and SN not in banned:
        print(SN)
        s = "%s plots/*Moriya*%s*models.png" % (s, SN)
        s = "%s plots/nodiff/*Moriya*%s*models.png" % (s, SN)
        s = "%s plots/*Hsiao*%s*models.png" % (s, SN)
if s != "feh":
    print(s)
    if not leftraru:
        os.system(s)

print("SNe classified as SNe II with spectra")
for SN in SNII:
    if SN not in banned and SN in spectra:
        print SN
        
#s = "feh"
#for SN in SNII:
#    s = "%s plots/*Moriya*%s*models.png" % (s, SN)
#os.system(s)


#s = "feh"
#for SN in SNIa:
#    s = "%s plots/*Hsiao*%s*corner.png" % (s, SN)
#print(s)

