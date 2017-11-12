import os, re, sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm

from sklearn.neighbors.kde import KernelDensity

from readSNdata import *
from scipy.stats import ks_2samp

dropSN = False
droplist = ["SNHiTS15G", "SNHiTS15Q"]

# use simulated light curves
dotest = False
if dotest:
    teststr = "_test3"
    #teststr = "_test2"
else:
    teststr = ""
    dfHiTS = pd.read_table("summary_HiTS.out", sep = "\s+", index_col = 0)
    if dropSN:
        dfHiTS.drop(droplist, inplace = True)
    SNHiTS = list(dfHiTS.index)
    print(list(dfHiTS.index))

# comparison
compstr = "_test2"    


dosavesamplestats = True#True
doreadall = False

if doreadall:

    allvars = {}
    df = pd.DataFrame()

    for f in sorted(os.listdir("samples")):

        if dotest:
            testfile = re.findall("chain_MoriyaWindAcc_(%s_\d+)_scale-texp(.*?)logAv-mass-energy-log10mdot-beta.dat" % teststr[1:], f)
        else:
            testfile = re.findall("chain_MoriyaWindAcc_(.*?)_scale-texp(.*?)logAv-mass-energy-log10mdot-beta.dat", f)
        if testfile != []:

            SN = testfile[0][0]
            fixz = readSNdata("HiTS", SN)[5]
            if fixz and re.search(".*logz.*", f) != None:
                print("Ignoring file %s" % f)
                continue
            elif not fixz and re.search(".*logz.*", f) == None:
                print("Ignoring file %s" % f)
                continue
                
            if not dotest and SN not in SNHiTS:
                continue

            if fixz:
                nchain, nwalker, scale, texp, logAv, mass, energy, log10mdot, beta = np.loadtxt("samples/%s" % f).transpose()
                logz = np.log(readSNdata("HiTS", SN)[6])
                print(SN, logz)
                logz = np.ones(len(mass)) * logz
            else:
                nchain, nwalker, scale, texp, logz, logAv, mass, energy, log10mdot, beta = np.loadtxt("samples/%s" % f).transpose()

        else:
            continue

        print(SN)
        
        mask = (nchain > 500)

        SNvars = {}
        SNvars['SN'] = np.hstack([list(map(lambda i: np.array(SN, dtype = str), range(len(energy))))])
        SNvars['energy'] = energy
        SNvars['mass'] = mass
        SNvars["log10mdot"] = log10mdot
        SNvars["beta"] = beta
        SNvars['z'] = np.exp(logz)
        SNvars['Av'] = np.exp(logAv)
        SNvars['texp'] = texp

        # compute percentiles for several variables and store in data frame
        for var in ['log10mdot', 'beta', 'mass', 'energy', 'z', 'Av', 'texp']:
            for perc in [5, 16, 50, 68, 95]:
                df.at[SN, "%sp%i" % (var, perc)] = np.percentile(SNvars[var][mask], perc)

        # stack all the posteriors for later saving
        for var in ["SN", "energy", "mass", "Av", "z", "log10mdot", "beta", "texp"]:
            if var in allvars.keys():
                allvars[var] = np.hstack([allvars[var], SNvars[var][mask]])
            else:
                allvars[var] = np.array(SNvars[var][mask])


    # save percentiles
    if dosavesamplestats:
        df.to_csv("summary_percentiles%s.out" % teststr, sep = " ", index_label = "SN")

    # select 10000 samples from each stacked posterior 
    nselection = 10000
    idx = np.random.choice(np.array(range(np.shape(allvars["SN"])[0])), size = nselection, replace = True)
    data = {}
    data['SN'] = allvars["SN"][idx]
    data['log10mdot'] = allvars["log10mdot"][idx]
    data['beta'] = allvars["beta"][idx]
    data['mass'] = allvars["mass"][idx]
    data['energy'] = allvars["energy"][idx]
    data['z'] = allvars["z"][idx]
    data['Av'] = allvars["Av"][idx]
    data['texp'] = allvars["texp"][idx]

    # create data frame and save
    df = pd.DataFrame.from_dict(data, orient='columns')
    df.to_csv("summary%s.dat" % teststr, sep = " ")

    sys.exit()


# function to draw cumulative histograms    
def hist1D(allvals, medianvals, varname, xlabel, bw = None):
    fig, ax = plt.subplots()

    # Silverman's rule
    if bw == None:
        bw = 0.9 * min(np.std(medianvals), np.abs(np.percentile(medianvals, 75) - np.percentile(medianvals, 25)) / 1.349) * len(medianvals)**(-1./5.)
        print(varname, bw, np.median(medianvals), np.std(medianvals), np.median(np.abs(np.median(medianvals) - medianvals)))
    kde = KernelDensity(kernel='gaussian', bandwidth = bw).fit(medianvals[:, np.newaxis])
    if varname == "z":
        X_plot = np.linspace(0, 0.55, 1000)[:, np.newaxis]
    else:
        X_plot = np.linspace(min(allvals) - 1.5 * bw, max(allvals) + 1.5 * bw, 1000)[:, np.newaxis]
    log_dens = kde.score_samples(X_plot)

    ax.fill_between(X_plot[:, 0], 0, np.exp(log_dens), facecolor = 'b', alpha = 0.5)


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

    ax.plot(X_plot[:, 0], np.exp(log_dens), c = 'k', alpha = 0.5)

    ax.set_xlabel(xlabel, fontsize = 14)
    ax.set_ylabel("p.d.f.", fontsize = 14)
    ax.set_ylim(0, ax.get_ylim()[1])
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    if varname[:4] == "texp":
        plt.xticks(fontsize = 14, rotation = 30)
        plt.ticklabel_format(useOffset=False)
    plt.tight_layout()
    plt.savefig("plots/samples/%s_hist_HiTS.png" % (varname))
                

# read percentile files and files with all posterior samples
df = pd.read_table("summary_percentiles%s.out" % teststr, sep = '\s+', index_col = 0)
if not dotest and dropSN:
    df.drop(droplist, inplace = True)

#print(df.columns)
#fig, ax = plt.subplots()
#ax.scatter(df['log10mdotp50'], df['texpp95'] - df['texpp5'], lw = 0)
#plt.show()
#sys.exit()

dfall = pd.read_table("summary%s.dat" % teststr, sep = "\s+")

# read test comparison
dftest = pd.read_table("summary_percentiles%s.out" % compstr, sep = '\s+', index_col = 0, comment = "#")
dftestsum = pd.read_table("summary%s.dat" % compstr, sep = '\s+', index_col = 0)

# banned files
banned_test3 = [10, 162, 226, 263, 65]
if compstr == "_test3":
    for i in banned_test3:
        label = "test3_%i" % i
        dftest.drop([label], inplace = True)
        dftestsum = dftestsum[dftestsum['SN'] != label]

# plot simulated vs inferred        
fig, ax = plt.subplots(ncols = 5, figsize = (20, 4))
HiTSdir = "../HiTS/LCs"
dfsimall = pd.DataFrame()
log10mdotdiffs = []
betadiffs = []
massdiffs = []
zdiffs = []
Avdiffs = []
for SN in dftest.index:
    sn_mjd, sn_mjdref, sn_flux, sn_e_flux, sn_filters, fixz, zcmb, texp0 = readSNdata("HiTS", SN)
    dt = max(sn_mjd) - min(sn_mjd)
    SNR = sn_flux / sn_e_flux
    if dt < 7 or np.sum(SNR >= 5) < 2:
        continue
    dfsim = pd.read_table("%s/%s.pars" % (HiTSdir, SN), sep = "\s+", comment = "#")
    #print(SN, float(dfsim['beta']), float(dftest.loc[SN].loc['betap5']), float(dftest.loc[SN].loc['betap50']), float(dftest.loc[SN].loc['betap95']))
    yerr = [[dftest.loc[SN].loc['log10mdotp50'] - dftest.loc[SN].loc['log10mdotp5']], [dftest.loc[SN].loc['log10mdotp95'] - dftest.loc[SN].loc['log10mdotp50']]]
    yerr = [[dftest.loc[SN].loc['log10mdotp50'] - dftest.loc[SN].loc['log10mdotp16']], [dftest.loc[SN].loc['log10mdotp68'] - dftest.loc[SN].loc['log10mdotp50']]]
    ax[0].errorbar(dfsim['log10mdot'], dftest.loc[SN].loc["log10mdotp50"], yerr = yerr, alpha = 0.5, marker = 'o')
    ax[0].plot([-8, -2], [-8, -2])
    log10mdotdiffs.append(dfsim['log10mdot'] - dftest.loc[SN].loc["log10mdotp50"])
    ax[0].set_xlabel('log10mdot sim')
    ax[0].set_ylabel('log10mdot est')
    yerr = [[dftest.loc[SN].loc['betap50'] - dftest.loc[SN].loc['betap5']], [dftest.loc[SN].loc['betap95'] - dftest.loc[SN].loc['betap50']]]
    yerr = [[dftest.loc[SN].loc['betap50'] - dftest.loc[SN].loc['betap16']], [dftest.loc[SN].loc['betap68'] - dftest.loc[SN].loc['betap50']]]
    ax[1].errorbar(dfsim['beta'], dftest.loc[SN].loc["betap50"], yerr = yerr, alpha = 0.5, marker = 'o')
    ax[1].plot([1, 5], [1, 5])
    betadiffs.append(dfsim['beta'] - dftest.loc[SN].loc["betap50"])
    ax[1].set_xlabel('beta sim')
    ax[1].set_ylabel('beta est')
    yerr = [[dftest.loc[SN].loc['massp50'] - dftest.loc[SN].loc['massp5']], [dftest.loc[SN].loc['massp95'] - dftest.loc[SN].loc['massp50']]]
    ax[2].errorbar(dfsim['mass'], dftest.loc[SN].loc["massp50"], yerr = yerr, alpha = 0.5, marker = 'o')
    ax[2].plot([12, 16], [12, 16])
    massdiffs.append(dfsim['mass'] - dftest.loc[SN].loc["massp50"])
    ax[2].set_xlabel('mass sim')
    ax[2].set_ylabel('mass est')
    print(SN, float(np.exp(dfsim['logz'])), float(dftest.loc[SN].loc['zp5']), float(dftest.loc[SN].loc['zp50']), float(dftest.loc[SN].loc['zp95']))
    yerr = [[dftest.loc[SN].loc['zp50'] - dftest.loc[SN].loc['zp5']], [dftest.loc[SN].loc['zp95'] - dftest.loc[SN].loc['zp50']]]
    ax[3].plot([0, 0.5], [0, 0.5])
    ax[3].set_ylim(0, 0.5)
    ax[3].errorbar(np.exp(dfsim['logz']), dftest.loc[SN].loc["zp50"], yerr = yerr, alpha = 0.5, marker = 'o')
    zdiffs.append(np.exp(dfsim['logz']) - dftest.loc[SN].loc["zp50"])
    ax[3].set_xlabel('z sim')
    ax[3].set_ylabel('z est')
    yerr = [[dftest.loc[SN].loc['Avp50'] - dftest.loc[SN].loc['Avp5']], [dftest.loc[SN].loc['Avp95'] - dftest.loc[SN].loc['Avp50']]]
    ax[4].plot([0, 0.5], [0, 0.5])
    ax[4].set_ylim(0, 0.5)
    ax[4].errorbar(np.exp(dfsim['logAv']), dftest.loc[SN].loc["Avp50"], yerr = yerr, alpha = 0.5, marker = 'o')
    Avdiffs.append(np.exp(dfsim['logAv']) - dftest.loc[SN].loc["Avp50"])
    ax[4].set_xlabel('Av sim')
    ax[4].set_ylabel('Av est')
    dfsimall = pd.concat([dfsimall, dfsim])

ax[0].set_title("%f %f" % (np.std(log10mdotdiffs), np.median(dftest['log10mdotp95'] - dftest['log10mdotp50'])))
ax[1].set_title("%f %f" % (np.std(betadiffs), np.median(dftest['betap95'] - dftest['betap50'])))
ax[2].set_title("%f %f" % (np.std(massdiffs), np.median(dftest['massp95'] - dftest['massp50'])))
ax[3].set_title("%f %f" % (np.std(zdiffs), np.median(dftest['zp95'] - dftest['zp50'])))
ax[4].set_title("%f %f" % (np.std(Avdiffs), np.median(dftest['Avp95'] - dftest['Avp50'])))
plt.show()

# change names for plotting purposes
print(df.columns)
dfsel = df[['log10mdotp50', 'betap50', 'massp50', 'energyp50', 'zp50', 'Avp50']]
print(df["betap50"])
dfsel.columns = [r"$\log_{10} \dot M$", r"$\beta$", 'mass', 'energy', r'$z$', r'$A_V$']

print(dfsel.columns)

#sns.pairplot(dfsel, kind = 'reg', diag_kind = 'kde')

import seaborn as sns

# plot correlation matrix
print(df.index)
fig, ax = plt.subplots()
print(dfsel.corr())
sns.heatmap(dfsel.corr(), cmap = sns.diverging_palette(220, 10, as_cmap = True), square = True, ax = ax)
plt.savefig("plots/Corr_matrix%s.png" % teststr)
plt.show()

# mdot, beta, mass plot
dm = df['massp95'] - df['massp5']
mass = df['massp50']
beta = df['betap50']
norm_mass = matplotlib.colors.Normalize(vmin=min(mass), vmax=max(mass))
norm_beta = matplotlib.colors.Normalize(vmin=min(beta), vmax=max(beta))
c_m = matplotlib.cm.jet
s_m_mass = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm_mass)
s_m_beta = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm_beta)
s_m_mass.set_array([])
s_m_beta.set_array([])

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
print(ms_sel)

fig, ax = plt.subplots(ncols = 3, figsize = (20, 8))
for SN in df.index:
    print(SN)
    xerr = [[df.loc[SN].loc['log10mdotp50'] - df.loc[SN].loc['log10mdotp5']], [df.loc[SN].loc['log10mdotp95'] - df.loc[SN].loc['log10mdotp50']]]
    yerr = [[df.loc[SN].loc['betap50'] - df.loc[SN].loc['betap5']], [df.loc[SN].loc['betap95'] - df.loc[SN].loc['betap50']]]
    yyerr = [[df.loc[SN].loc['Avp50'] - df.loc[SN].loc['Avp5']], [df.loc[SN].loc['Avp95'] - df.loc[SN].loc['Avp50']]]
    zerr = [[df.loc[SN].loc['massp50'] - df.loc[SN].loc['massp5']], [df.loc[SN].loc['massp95'] - df.loc[SN].loc['massp50']]]
    #xerr = [[df.loc[SN].loc['log10mdotp50'] - df.loc[SN].loc['log10mdotp16']], [df.loc[SN].loc['log10mdotp68'] - df.loc[SN].loc['log10mdotp50']]]
    #yerr = [[df.loc[SN].loc['betap50'] - df.loc[SN].loc['betap16']], [df.loc[SN].loc['betap68'] - df.loc[SN].loc['betap50']]]
    #zerr = [[df.loc[SN].loc['massp50'] - df.loc[SN].loc['massp16']], [df.loc[SN].loc['massp68'] - df.loc[SN].loc['massp50']]]
    limbeta = 9
    limlog10mdot = 9
    limdm = 9
    if df.loc[SN].loc['betap95'] - df.loc[SN].loc['betap5'] < limbeta and df.loc[SN].loc['log10mdotp95'] - df.loc[SN].loc['log10mdotp5'] < limlog10mdot and dm[SN] < limdm:
        ax[2].errorbar(df.loc[SN].loc['log10mdotp50'], df.loc[SN].loc['Avp50'], marker = 'o', xerr = xerr, yerr = yyerr, color = s_m_mass.to_rgba(mass.loc[SN]))#, markersize = 5 * dm[SN])
        ax[1].errorbar(df.loc[SN].loc['log10mdotp50'], df.loc[SN].loc['betap50'], marker = 'o', xerr = xerr, yerr = yerr, color = s_m_mass.to_rgba(mass.loc[SN]))#, markersize = 5 * dm[SN])
        ax[0].errorbar(df.loc[SN].loc['massp50'], df.loc[SN].loc['log10mdotp50'], marker = 'o', xerr = zerr, yerr = xerr, color = s_m_beta.to_rgba(beta.loc[SN]))#, markersize = 5 * dm[SN])

ax[1].set_xlabel(r"$\log_{10}\ \dot M\ [M_\odot/yr]$", fontsize = 16)
ax[1].set_ylabel(r"$\beta$", fontsize = 16)
ax[0].set_ylabel(r"$\log_{10}\ \dot M\ [M_\odot/yr]$", fontsize = 16)
ax[0].set_xlabel(r"$Mass\ [M_\odot]$", fontsize = 16)
cbar = plt.colorbar(s_m_mass)
cbar.set_label(r"$Mass\ \ [M_\odot]$", fontsize = 16)
plt.savefig("log10mdotbetamass%s.png" % teststr)
plt.show()


fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize = (20, 10))

cumulative = True
for idx, label in enumerate(['beta', 'mass', 'log10mdot', 'energy']):

    ax[0, idx].hist(df['%sp50' % label], alpha = 0.5, normed = True, cumulative = cumulative, label = "medians")
    ax[0, idx].hist(dfall[label], alpha = 0.5, normed = True, cumulative = cumulative, bins = 100, label = "data posterior")
    ax[0, idx].hist(dftestsum[label], alpha = 0.5, normed = True, cumulative = cumulative, bins = 100, label = "test posterior")
    D, p = ks_2samp(dfall[label], dftestsum[label])
    ax[0, idx].set_title("%f %e" % (D, p))
    ax[0, idx].set_xlabel(label)
    print(label)
    ps = np.array(list(map(lambda i: ks_2samp(np.random.choice(dfall[label], size = len(df.index)), np.random.choice(dftestsum[label], size = len(df.index)))[1], range(1000))))
    ax[0, idx].legend(fontsize = 8)
    ax[1, idx].hist(np.log10(ps), cumulative = cumulative, normed = True, bins = 100)
        

##ax[1].hist(df['massp50'], alpha = 0.5, normed = True)
##ax[1].hist(dfsimall['mass'], alpha = 0.5, normed = True)
#ax[1].hist(dfall['mass'], alpha = 0.5, normed = True, cumulative = True)
#ax[1].hist(dftestsum['mass'], alpha = 0.5, normed = True, cumulative = True)
#D, p = scipy.stats.ks_2samp(dfall['mass'], dftestsum['mass'])
#ax[0].set_title("%f %e" % D, p)
#ax[1].set_xlabel('mass')
#ax[1].set_ylabel('mass')
#
##ax[2].hist(df['log10mdotp50'], alpha = 0.5, normed = True)
##ax[2].hist(np.log10(dfsimall['mdot']), alpha = 0.5, normed = True)
#ax[2].hist(dfall['log10mdot'], alpha = 0.5, normed = True, cumulative = True)
#ax[2].hist(dftestsum['log10mdot'], alpha = 0.5, normed = True, cumulative = True)
#D, p = scipy.stats.ks_2samp(dfall['beta'], dftestsum['beta'])
#ax[0].set_title("%f %e" % D, p)
#ax[2].set_xlabel('log10mdot')
#ax[2].set_ylabel('log10mdot')
#
##ax[3].hist(df['energyp50'], alpha = 0.5, normed = True)
##ax[3].hist(dfsimall['energy'], alpha = 0.5, normed = True)
#ax[3].hist(dfall['energy'], alpha = 0.5, normed = True, cumulative = True)
#ax[3].hist(dftestsum['energy'], alpha = 0.5, normed = True, cumulative = True)
#ax[3].set_xlabel('energy')
#ax[3].set_ylabel('energy')

plt.show()

