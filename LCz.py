#! /usr/bin/python2.7
# CC by F. Forster -----------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import sys, re
sys.path.append("../cos_calc")
import cos_calc

#from iminuit import Minuit

#from pylab import *

sys.path.append("../surveysim")
from constants import *
from extinction import *

import random
import sys

from scipy.interpolate import interp1d # interpolation
from scipy.optimize import curve_fit

# black body flux given frequency, radius and temperature
# Note that nu, R and T are vectors, R and T should have the same dimensions
def BB(nu, R, T):
    
    area = 4. * np.pi * np.atleast_2d(R).T**2 # R should be in cm
    surfaceflux = 2. * np.pi * h * np.atleast_2d(nu)**3 / cspeed**2 * 1. / (np.exp(h * np.atleast_2d(nu) / (k * np.atleast_2d(T).T)) - 1.) # ergs / s / Hz / cm2
    
    return area * surfaceflux # erg / s / Hz

# class that redshifts a series of time evolving spectra (designed for SN light curves)
class LCz(object):

    # object variables required for what follows are
    # modelname, ntimes, nlambda, times (ntimes) [days], lambda (nlambda) [AA], flux (ntimes x nlambda) [erg/s/AA] and doplot

    def __init__(self, modelname, times, lambdas, flux, doplot = False):
        
        self.modelname = modelname
        self.times = times  # days
        self.lambdas = lambdas # AA
        self.flux = flux # erg/s/AA 2 dimensional tensor (time, lambda)
        self.doplot = doplot

        self.ntimes = len(times)
        self.nlambdas = len(lambdas)
        self.extmodel = "CCM89+O94"
    
    def luminosity(self, **kwargs):

        # compute delta times
        self.nu = cspeedAAs / self.lambdas
        self.dtimes = np.zeros(self.ntimes)
        self.dtimes[1:self.ntimes] = self.times[1: self.ntimes] - self.times[0: self.ntimes - 1]
        
        # compute nu, dnu, dlambda
        self.dnu = np.zeros(self.nlambdas)
        self.dlambda = np.zeros(self.nlambdas)
        self.dlambda[1:self.nlambdas] = self.lambdas[1: self.nlambdas] - self.lambdas[0: self.nlambdas - 1]
        self.dnu[1:self.nlambdas]     = -(self.nu[1: self.nlambdas] - self.nu[0: self.nlambdas - 1])
        
        # compute total luminosity
        self.L = np.zeros(self.ntimes)

        for i in range(self.ntimes):

            self.L[i] = np.sum(self.flux[i] * self.dlambda)

        if self.doplot:
            print("Plotting integrated energy")

            plt.clf()
            ax = fig.add_subplot(1,1,1)
            ax.set_xlabel(r'$t$ [days]')
            ax.set_ylabel(r'$\log_{10}$ $L$ [erg s$^{-1}$]')
                
            ax.plot(self.times, np.log10(self.L))
            ax.text(self.times[np.argmax(np.log10(self.L))], max(np.log10(self.L)), "Total energy radiated: %7.2e [erg]" % (np.sum(self.L * self.dtimes) * yr2sec), fontsize = 8)
            
            ax.set_xlim(min(self.times) - 1, max(self.times))
            ax.set_ylim(min(np.log10(self.L)) - 1, max(np.log10(self.L)) + 1)
            plt.grid()
            plt.savefig("plots/%s_Luminosity.png" % (self.modelname))
            

    # add extinction at restframe (uses class extinction)
    def attenuate(self, **kwargs):
        
        self.Av = kwargs["Av"]
        self.Rv = kwargs["Rv"]

        if self.Av != 0:
            self.dopeaks = False

        ext = extinction(Av = self.Av, Rv = self.Rv, model = self.extmodel)
        (x, self.attenuation) = ext.dm(lambdasAA = self.lambdas)
        self.fluxatt = self.flux * 10**(- self.attenuation / 2.5)
        
    # redshift spectra at different redshifts
    def redshift(self, **kwargs):

        # arguments
        self.zs = kwargs["zs"]
        self.DL = kwargs["DL"]

        # derived quantities
        self.nz = len(self.zs)
        self.fluxz = np.zeros((self.nz, self.ntimes, self.nlambdas))
        self.timesz = np.zeros((self.nz, self.ntimes))
        
        self.lambdaz = np.zeros((self.nz, self.nlambdas))
        self.nuz = np.zeros((self.nz, self.nlambdas))
        self.dlambdaz = np.zeros((self.nz, self.nlambdas))
        self.dnuz = np.zeros((self.nz, self.nlambdas))
        self.dlognuz = np.zeros((self.nz, self.nlambdas))
        self.Lz = np.zeros((self.nz, self.ntimes))
        
        for i in range(self.nz):

            self.timesz[i] = self.times * (1. + self.zs[i])
            self.lambdaz[i] = self.lambdas * (1. + self.zs[i])
            self.nuz[i] = cspeedAAs / self.lambdaz[i]
            
            self.dlambdaz[i][1:self.nlambdas] = self.lambdaz[i][1: self.nlambdas] - self.lambdaz[i][0: self.nlambdas - 1]
            self.dnuz[i][1:self.nlambdas] = -(self.nuz[i][1: self.nlambdas] - self.nuz[i][0: self.nlambdas - 1])
            
            self.dlognuz[i] = self.dnuz[i] / self.nuz[i]
            
            for j in range(self.ntimes):
                
                # use attenuated flux if it exists
                if hasattr(self, "fluxatt"):
                    obsflux = self.fluxatt[j]
                else:
                    obsflux = self.flux[j]
                self.fluxz[i, j] = obsflux / 4. / np.pi / (self.DL[i] * 1e6 * pc2cm)**2 / (1. + self.zs[i])
                self.Lz[i, j] = np.sum(self.fluxz[i, j] * self.dlambdaz[i])

    # get filter transmission around given band
    def dofilter(self, **kwargs):

        self.filtername = kwargs["filtername"]
                
        # read transmission curve and create interpolation function
        # --------------------------------------------------------
            
        ugrizy = ['u', 'g', 'r', 'i', 'z', 'Y']
        UBVRI = ['U', 'B', 'V', 'R', 'I']

        if self.filtername in ugrizy:

            bandfilter = np.loadtxt('/home/fforster/Work/surveysim/filters/DECam_transmission_total.txt').transpose()
            iband = 0
            for i in range(len(ugrizy)):
                if ugrizy[i] == self.filtername:
                    iband = i
            lfilter = bandfilter[0] * 10. # AA
            tfilter = bandfilter[iband + 1]  # fraction

        elif self.filtername in UBVRI:

            bandfilter = np.loadtxt('/home/fforster/Work/surveysim/filters/Bessel_%s-1.txt' % self.filtername).transpose()
            lfilter = bandfilter[0] * 10. # AA
            tfilter = bandfilter[1] / 100. # fraction
            idxsort = np.argsort(lfilter)
            lfilter = lfilter[idxsort]
            tfilter = tfilter[idxsort]

        elif self.filtername == "Kepler":

            bandfilter = np.loadtxt('filters/kepler_response.dat').transpose()
            lfilter = bandfilter[0] * 10. # AA
            tfilter = bandfilter[1] # fraction
            idxsort = np.argsort(lfilter)
            lfilter = lfilter[idxsort]
            tfilter = tfilter[idxsort]

        i1 = 0
        i2 = 0
        for i in range(len(lfilter)):
            if tfilter[i] > 0.01:
                i1 = i
                break
        
        for i in np.arange(len(lfilter) - 1, 0, -1):
            if tfilter[i] > 0.01:
                i2 = i
                break
        
        bandf = interp1d(lfilter[i1:i2], tfilter[i1:i2])  # AA
        maxlband = max(lfilter[i1:i2]) # AA
        minlband = min(lfilter[i1:i2]) # AA
        
        # create array of lambdas around the position of the filter
        # ---------------------------------------------------------
        
        self.bandlambda = np.arange(minlband, maxlband, 1.) # step of 1 AA
        self.bandSnu = bandf(self.bandlambda)
        
        self.nnu = len(self.bandlambda)
        
        bandnu = cspeedAAs / self.bandlambda
        dbandnu = np.zeros(self.nnu)
        dbandnu[1:self.nnu] = -(bandnu[1:self.nnu] - bandnu[0:self.nnu-1])
        self.dlognu = dbandnu / bandnu
        
    # compute observed magnitudes at previously set band
    def mags(self, **kwargs):
        
        self.Dm = kwargs["Dm"]

        self.bandmag = np.zeros((self.nz, self.ntimes))
        self.Snuz = np.zeros((self.nz, self.nlambdas))

        # map fluxes on bandlambda creating interpolating functions
        # -------------------------------------------------------------
        
        self.bandfluxz = np.zeros((self.nz, self.ntimes, self.nnu))
        
        for i in range(self.nz):
        
            for j in range(self.ntimes):

                fluxzf = interp1d(self.lambdaz[i], self.fluxz[i, j])
                self.bandfluxz[i, j] = fluxzf(self.bandlambda)

        if self.doplot:
            
            print("Plotting redshifted spectra and transmission filter...")
            fig, ax = plt.subplots()
            ax.set_xlabel(r'$\lambda$ [\AA]')
            ax.set_ylabel(r'$\log_{10}$ flux [erg s$^{-1}$ \AA$^{-1}$]')
            ax2 = ax.twinx()
            ax2.set_ylabel(r'Transmission')
            ax2.plot(self.bandlambda, self.bandSnu)
            
            # color map as a function of time step
            jet = cm = plt.get_cmap('jet') 
            cNorm  = colors.Normalize(vmin = 0, vmax = self.nz)
            scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = jet)
            
            for i in range(self.nz):
                
                colorVal = scalarMap.to_rgba(i)
                ax.plot(self.bandlambda, np.log10(self.bandfluxz[i, np.argmin(np.abs(self.timesz[i] - 1.))]), color = colorVal, label = "z: %5.3f" % self.zs[i])
                
            ax.legend(fancybox = False, prop = {'size':8}, loc = 1)
            ax.set_title("Time: 1 day rest frame")
            plt.grid()
            plt.savefig("plots/%s_%s_fluxz.png" % (self.modelname, self.filtername))
            
        
        # integrate bands
        # ---------------
        
        self.absmagz = np.zeros(np.shape(self.bandmag))
        for i in range(self.nz):
            for j in range(self.ntimes):
                self.bandmag[i, j] = -2.5 * np.log10(np.sum(self.bandfluxz[i, j] * (self.bandlambda**2 / cspeedAAs) * self.bandSnu * self.dlognu) / np.sum(self.bandSnu * self.dlognu)) - 48.6
                if not np.isfinite(self.bandmag[i, j]):
                    self.bandmag[i, j] = 40.
            # compute effective absolute magnitudes
            self.absmagz[i] = self.bandmag[i] - self.Dm[i]

        # find maximum, SBO peak, peak end, and different days post SBO in restframe time
        # --------------------------------------------------------------------------------
        if self.dopeaks:

            self.tmax = np.zeros(self.nz)
            self.tstartpeak = np.zeros(self.nz)
            self.tpeak = np.zeros(self.nz)
            self.tendpeak = np.zeros(self.nz)
            self.tday1 = np.zeros(self.nz)
            self.tday2 = np.zeros(self.nz)
            self.tday4 = np.zeros(self.nz)
            self.tday6 = np.zeros(self.nz)
        
            for i in range(self.nz):
        
                # mag time derivative
                dbandmag = np.zeros(np.shape(self.bandmag[i]))
                dbandmag[0] = 0
                dbandmag[1:-1] = (self.bandmag[i][1:-1] - self.bandmag[i][0:-2])
                
                # find start of peak based on derivative and absolute flux (< -12)
                maxlim = 6
                mask = (self.timesz[i] > 0) & (self.timesz[i] < maxlim) & (self.bandmag[i] - self.Dm[i] < -12) & (dbandmag < -1e-8)# last constraint is on absolute magnitude
                self.tstartpeak[i] = np.min(self.timesz[i][mask])
                
                # find peak based on tstartpeak and derivative
                mask = (self.timesz[i] > self.tstartpeak[i]) & (dbandmag > 1e-8)
                ipeak = np.argmin(self.timesz[i][mask])
                self.tpeak[i] = self.timesz[i][mask][ipeak]
                bandmagpeak = self.bandmag[i][mask][ipeak]
                # find end of peak based on tpeak
                mask = (self.timesz[i] > self.tpeak[i]) & (dbandmag < -1e-8)
                iendpeak = np.argmin(self.timesz[i][mask])
                self.tendpeak[i] = self.timesz[i][mask][iendpeak]
                bandmagendpeak = self.bandmag[i][mask][iendpeak]
                deltamag = np.abs(bandmagendpeak - bandmagpeak)
                mask = (self.timesz[i] >= self.tpeak[i]) & (self.timesz[i] <= self.tendpeak[i]) & (self.bandmag[i] >= bandmagpeak + 0.9 * deltamag)
                if np.sum(mask) > 0:
                    self.tendpeak[i] = np.min(self.timesz[i][mask])
                # time one and two days after first peak
                self.tday1[i] = self.timesz[i][np.argmin(np.abs(self.times - self.tpeak[i] - 1.))]
                self.tday2[i] = self.timesz[i][np.argmin(np.abs(self.times - self.tpeak[i] - 2.))]
                self.tday4[i] = self.timesz[i][np.argmin(np.abs(self.times - self.tpeak[i] - 4.))]
                self.tday6[i] = self.timesz[i][np.argmin(np.abs(self.times - self.tpeak[i] - 6.))]
                # find max post SBO
                mask = (self.timesz[i] > self.tendpeak[i])
                self.tmax[i] = self.timesz[i][mask][np.argmin(self.bandmag[i][mask])]
                print("Redshift: %4.2f, start of peak: %f obs. days, peak: %f obs. days, end of peak: %f obs. days, 1 day after end of peak: %f obs. days, maximum: %f obs. days" % (self.zs[i], self.tstartpeak[i], self.tpeak[i], self.tendpeak[i], self.tday1[i], self.tmax[i]))
                #if self.tpeak[i] > maxlim:
                #    print("WARNING: tpeak looks too high")
                #    sys.exit()
        else:
            
            if self.modelname[:4] == "NS10":
                
                # some variables used for SBO detection metric
                self.tstartpeak = np.zeros_like(self.zs)
                self.tpeak = self.SN.t0() / self.SN.ct0R() / 24. * (1. + self.zs)# SBO peak after time delay
                self.tendpeak = self.SN.ts() / 24. * (1. + self.zs) # beginning of planar phase
                self.tday1 = 1. * (1. + self.zs)# time for SBO to end
                self.tday2 = 2. * (1. + self.zs)# time for SBO to end
                self.tday4 = 4. * (1. + self.zs)# time for SBO to end
                self.tday6 = 6. * (1. + self.zs)# time for SBO to end
                self.tmax = 10. * self.SN.R500**0.69 * (self.SN.E51 / self.SN.M15)**0.19 * (1. + self.zs)

        if self.doplot:

            # plot absolute magnitudes as seen at different redshifts
            # -------------------------------------------------------

            print("Plotting absolute redshifted magnitudes...")

            plt.clf()
            ax = fig.add_subplot(1,1,1)
            ax.set_xlabel(r'$t$ [days]')
            ax.set_ylabel(r'%s' % self.filtername)
            
            jet = cm = plt.get_cmap('jet') 
            cNorm  = colors.Normalize(vmin = 0, vmax = self.nz)
            scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = jet)
            for i in range(self.nz):

                colorVal = scalarMap.to_rgba(i)

                ax.plot(self.timesz[i], self.absmagz[i], color = colorVal, label = "z: %5.3f" % self.zs[i])
                idx90 = np.argmin(np.abs(self.times - 90.))
                ax.text(self.timesz[i, idx90], self.absmagz[i, idx90], "z: %4.2f" % self.zs[i], color = colorVal, fontsize = 8) # show label at 90 days restframe
        
                if self.dopeaks:
                    ax.axvline(self.tstartpeak[i], c = colorVal, alpha = 0.2, ls = '-')
                    ax.axvline(self.tpeak[i], c = colorVal, alpha = 0.2, lw = 2, ls = '-')
                    ax.axvline(self.tendpeak[i], c = colorVal, alpha = 0.2, ls = '-.')
                    ax.axvline(self.tday1[i], c = 'gray', alpha = 0.2, ls = ':')
                    ax.axvline(self.tmax[i], c = colorVal, alpha = 0.2, ls = '--')
                    ax.axvline(90)
                    
            ax.legend(fancybox = False, prop = {'size':8}, loc = 1)
        
            ax.set_xlim(np.min(self.timesz), np.max(self.timesz) * 1.5)
            
            ax.set_xlim(np.min(self.timesz), min(150, np.max(self.timesz)))
            ax.set_ylim(min(self.absmagz.flatten()) + 10, min(self.absmagz.flatten()) - 0.5)
            ax.axhline(23.5)
            
            plt.grid()
            ax.legend(fancybox = False, prop = {'size':8}, loc = 4, framealpha = 0.6)

            plt.savefig("plots/%s_%s_absmag.png" % (self.modelname, self.filtername))
            plt.savefig("plots/%s_%s_absmag.pdf" % (self.modelname, self.filtername))

            if self.dopeaks:
                ax.set_ylim(min(self.absmagz.flatten()) + 5, min(self.absmagz.flatten()) - 0.5)
                ax.set_xlim(np.min(self.timesz), max(self.tmax))
                plt.savefig("plots/%s_%s_absmag_premax.png" % (self.modelname, self.filtername))
                
                ax.set_xlim(min(self.tstartpeak) - 0.05, max(self.tendpeak) + 0.05)
                plt.savefig("plots/%s_%s_absmag_peak.png" % (self.modelname, self.filtername))
                
        
            # plot apparent magnitude for different redshifts
            # -----------------------------------------------
        
            print("Plotting apparent magnitudes at different redshifts...")

            plt.clf()
            ax = fig.add_subplot(1,1,1)
        
            jet = cm = plt.get_cmap('jet') 
            cNorm  = colors.Normalize(vmin = 0, vmax = self.nz)
            scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = jet)
            
            for i in range(self.nz):
        
                colorVal = scalarMap.to_rgba(i)
                
                idx90 = np.argmin(np.abs(self.times - 90.))
                ax.plot(self.timesz[i], self.bandmag[i], color = colorVal, label = "z: %5.3f" % self.zs[i])
                ax.text(self.timesz[i, idx90], self.bandmag[i, idx90], "z: %4.2f" % self.zs[i], color = colorVal, fontsize = 8) # show label at 90 days restframe
        
                if self.dopeaks:
                    ax.axvline(self.tpeak[i], c = colorVal, alpha = 0.2)
                    ax.axvline(self.tendpeak[i], c = colorVal, alpha = 0.2)
                    ax.axvline(self.tday1[i], c = 'gray', alpha = 0.2)
                    ax.axvline(self.tmax[i], c = colorVal, alpha = 0.2)
        
            ax.legend(fancybox = False, prop = {'size':8}, loc = 1)
            ax.set_xlabel(r'$t$ [days]')
            ax.set_ylabel(r'%s' % self.filtername)
            ax.set_ylim(min(self.bandmag.flatten()) + 10, min(self.bandmag.flatten()) - 0.5) 

            ax.set_xlim(np.min(self.timesz), min(np.max(self.timesz), 150))
            plt.savefig("plots/%s_%s_mag.png" % (self.modelname, self.filtername))
            plt.savefig("plots/%s_%s_mag.pdf" % (self.modelname, self.filtername))
            
            if self.dopeaks:
                ax.set_ylim(min(self.bandmag.flatten()) + 6, min(self.bandmag.flatten()) - 0.5) 
                ax.set_xlim(0, max(self.tmax))
                plt.savefig("plots/%s_%s_mag_premax.png" % (self.modelname, self.filtername))
                
                print(self.tpeak, self.tendpeak)
                ax.set_xlim(min(self.tpeak), max(self.tendpeak))
                plt.savefig("plots/%s_%s_mag_peak.png" % (self.modelname, self.filtername))


# Model based on output from the Stella code
class StellaModel(LCz):

    def __init__(self, **kwargs):

        self.dir = kwargs["dir"]
        self.modelfile = kwargs["modelfile"]
        self.modelname = self.modelfile
        self.doplot = False
        if "doplot" in kwargs.keys():
            self.doplot = kwargs["doplot"]
        self.extmodel = "CCM89+O94"

        self.dopeaks = False
        if "dopeaks" in kwargs.keys():
            self.dopeaks = kwargs["dopeaks"]
        
        # read file
        try:
            filename = "%s/%s" % (self.dir, self.modelfile)
            print("Opening model file %s" % filename)
            data = open(filename, 'r')
        except:
            print("Cannot open file %s" % self.modelfile)
            sys.exit()
            
        # extract wavelengths
        self.lambdas = data.readline()
        self.lambdas = np.array(self.lambdas.split()[3:], dtype = float)
        self.nlambdas = np.shape(self.lambdas)[0]

        # extract all spectra
        specall = re.findall("(.*)\n", data.read())
        self.ntimes = np.shape(specall)[0]

        # initialize flux and time arrays
        self.flux = np.zeros((self.ntimes, self.nlambdas))
        self.times = np.zeros(self.ntimes)

        # fill flux for given time
        for i in range(self.ntimes):
            fluxes = np.array(specall[i].split(), dtype = float)
            self.times[i] = fluxes[0]
            self.flux[i, :] = 10**(-(fluxes[1:] + 48.6) / 2.5)  # dI/dnu [erg/s/cm2/Hz]
            self.flux[i, :] = self.flux[i, :] * (cspeedAAs / self.lambdas**2) # [erg/s/cm2/AA]
            self.flux[i, :] = self.flux[i, :] * (4. * np.pi * (10. * pc2cm)**2) # [erg/s/AA]

# Hsiao model based on columns time, lambdas and flux
class Hsiao(LCz):

    def __init__(self, **kwargs):

        self.dir = kwargs["dir"]
        self.modelfile = kwargs["modelfile"]
        self.modelname = self.modelfile
        self.doplot = False
        if "doplot" in kwargs.keys():
            self.doplot = kwargs["doplot"]
        self.extmodel = "CCM89+O94"
        self.dopeaks = False
        
        # read file
        try:
            filename = "%s/%s" % (self.dir, self.modelfile)
            print("Opening model file %s" % filename)
            data = np.loadtxt(filename).transpose()
        except:
            print("Cannot open file %s" % self.modelfile)
            sys.exit()

        # extract times
        self.times = np.unique(data[0])
        self.times = self.times - min(self.times)
        self.ntimes = np.shape(self.times)[0]
        
        # extract wavelengths
        self.lambdas = np.unique(data[1])
        self.nlambdas = np.shape(self.lambdas)[0]

        # extract fluxes
        scale = 10**47.802 # this gives -19.46 mag at max in B
        self.flux = np.reshape(scale * data[2], (self.ntimes, self.nlambdas))


# empirical evolution of SN light curves from Tominaga et al.
class T11(StellaModel):

    def __init__(self, **kwargs):

        print(kwargs.keys())
        

        super(StellaModel, self).__init__(self, **kwargs)
        self.modelname = "T11_%s" % self.modelname

# empirical evolution of SN light curves from Yoon
class Yoon(StellaModel):

    def __init__(self, **kwargs):

        kwargs["modelfile"] = "yoon%s" % kwargs["modelfile"]
        kwargs["dir"] = "%s/%s" % (kwargs["dir"], "yoon12msun")
        super(StellaModel, self).__init__(self, kwargs)
        

# Type Ia supernova light curve from Dado & Dar 2013
class SNIaDadoDar(object):

    def __init__(self, **kwargs):

        self.Mc = float(kwargs["Mc"])        # Msun
        self.M56Ni = float(kwargs["M56Ni"])  # Msun
        self.fe = 0.3
        if "fe" in kwargs.keys():
            self.fe = float(kwargs["fe"]) # ~0.3 fraction of free (ionized) electrons
        self.Ae = 0.05
        if "Ae" in kwargs.keys():
            self.Ae = float(kwargs["Ae"]) # ~0.05 ratio between energy released in the decay of 56Co as positron kinetic energy and gamma ray energy 

        self.tsec = np.linspace(0, 300, 10000) * (24. * 3600.)  # s
        self.tdays = self.tsec / 3600. / 24.

        self.sigmaCompton56Ni = 9.5e-26 # cm2
        self.sigmaCompton56Co = 8.7e-26 # cm2

    def Ek(self):
        
        return 1.47e51 * self.Mc # erg

    def V(self):
        
        return np.sqrt(10. / 3. * self.Ek() / (self.Mc * Msun)) # cm / s
    
    def tr(self):
        
        return np.sqrt(3. * self.Mc * Msun * self.fe * sigmaT / (8. * np.pi * mp * cspeed * self.V())) # sec

    def tgamma(self, sigmaCompton):
        
        return np.sqrt(3. * self.Mc * Msun * sigmaCompton / (8. * np.pi * mp * self.V()**2)) # sec

    def Agamma(self, sigmaCompton):
        
        return 1. - np.exp(-self.tgamma(sigmaCompton)**2 / self.tsec**2) # dimensionless

    def Edotgamma(self):

        return  1e43 * self.M56Ni * (7.78 * self.Agamma(self.sigmaCompton56Ni) * np.exp(-self.tdays / 8.76) 
                                     + 1.5 * self.Agamma(self.sigmaCompton56Co) * (np.exp(-self.tdays / 111.27) - np.exp(-self.tdays / 8.76))) # erg / s

    def Edotpositron(self):
        
        return 1e43 * self.M56Ni * self.Ae * (np.exp(-self.tdays / 111.27) - np.exp(-self.tdays / 8.76)) # erg / s

    def Edot(self):
        
        return self.Edotgamma() + self.Edotpositron()  # erg / s

    def Lbolf(self):
        
        Lbol = self.tsec * np.exp(self.tsec**2 / (2. * self.tr()**2)) * self.Edot()  # s * erg/s = erg
        Lbol[1:] = (self.tsec[1:] - self.tsec[:-1]) * (Lbol[:-1] + Lbol[1:]) / 2.
        Lbol[1:] = np.cumsum(Lbol[1:])
        Lbol[0] = 0
        Lbol = np.exp(-self.tsec**2 / (2. * self.tr()**2)) / self.tr()**2 * Lbol
        return interp1d(self.tdays, Lbol, bounds_error = False, fill_value = 0)
         
        return Lbolf

    def Vph(self):

        factor = (1. - 2. * self.V() * self.tsec**2 / (3. * cspeed * self.tr()**2))
        mask = factor > 0.2
        return self.V() * factor[mask]
                           
    def Rphf(self):
        
        factor = (1. - 2. * self.V() * self.tsec**2 / (3. * cspeed * self.tr()**2))
        mask = factor > 0.2
        return interp1d(self.tdays[mask], self.V() * self.tsec[mask] * factor[mask], bounds_error = False, fill_value = 0)

    def Tf(self):
        
        return interp1d(self.tdays, ((1. - np.exp(-self.tsec**2 / (2. * self.tr()**2))) * self.Edot() / (4. * np.pi * self.V()**2 * self.tsec**2 * sigmaSB))**0.25, bounds_error = False, fill_value = 0)


# early evolution SN light curves from Nakar & Sari 2010
class SNIINakarSari(object):

    def __init__(self, **kwargs):

        self.M15 = kwargs["M15"]
        self.E51 = kwargs["E51"]
        self.R500 = kwargs["R500"]

    def E0(self):

        return 9e47 * self.M15**-0.43 * self.R500**1.74 * self.E51**0.56 # erg

    def t0(self):
        
        return 300. / 3600. * self.M15**0.21 * self.R500**2.16 * self.E51**-0.79 # hr

    def L0(self):

        return self.E0() / (3600. * self.t0()) # erg / s

    def ct0R(self):

        return 0.25 * self.M15**0.21 * self.R500**1.16 * self.E51**-0.79 # dimensionless

    def Tobs0(self):
        
        return 25. * self.M15**-0.3 * self.R500**-0.65 * self.E51**0.5 #  K

    # transition between planar and spherical phases
    def ts(self):

        return 14. * self.M15**0.43 * self.R500**1.26 * self.E51**-0.56 # hr


    # eta0 (< 1 implies radiation in thermal emission)
    def eta0(self, **kwargs):

        return 0.06 * self.M15**-1.72 * self.R500**-0.76 * self.E51**2.16
        
    # luminosity during precursor, SBO, planar and spherical phases
    def L(self, **kwargs):
        
        thr = kwargs["thr"]

        # time in days
        td = thr / 24.

        # masks
        maskpos = (thr > 0)
        maskprecursor = maskpos & (thr <= self.t0())
        maskSBO = maskpos & (maskprecursor == False) & (thr <= self.t0() / self.ct0R())
        maskplanar = maskpos & (maskprecursor == False) & (maskSBO == False) & (thr <= self.ts())
        maskspherical = maskpos &(maskprecursor == False) & (maskSBO == False) & (maskplanar == False)
        
        L = np.zeros(np.shape(thr))
        
        # precursor
        L[maskprecursor] = self.L0() * np.exp(1. - self.t0() / thr[maskprecursor])

        # luminosity during planar phase
        L[maskSBO | maskplanar] = 1.1e44 * self.M15**-0.37 * self.R500**2.46 * self.E51**0.3 * thr[maskSBO | maskplanar]**(-4./3.) # erg / s

        # luminosity during spherical phase
        L[maskspherical] = 3e42 * self.M15**-0.87 * self.R500 * self.E51**0.96 * td[maskspherical]**-0.17 # erg / s
        #L[maskpos] = 3e42 * self.M15**-0.87 * self.R500 * self.E51**0.96 * td[maskpos]**-0.17 # erg / s

        return L

    # temperature during precursor, SBO, planar and spherical phases
    def T(self, **kwargs):

        # time in hours
        thr = kwargs["thr"]
    
        # time in days
        td = thr / 24.

        maskpos = (thr > 0)
        maskprecursor = maskpos & (thr <= self.t0())
        maskSBO = maskpos & (maskprecursor == False) & (thr <= self.t0() / self.ct0R())
        maskplanar = maskpos & (maskprecursor == False) & (maskSBO == False) & (thr <= self.ts())
        maskspherical = maskpos & (maskprecursor == False) & (maskSBO == False) & (maskplanar == False)
        
        T = np.zeros(np.shape(thr))

        # temperature at precursor and SBO
        T[maskprecursor] = self.Tobs0()

        # temperature during planar phase
        T[maskSBO | maskplanar] = 10.4 * self.M15**-0.22 * self.R500**0.12 * self.E51**0.23 * thr[maskSBO | maskplanar]**-0.36 # eV

        # temperature during spherical phase
        T[maskspherical] = 3. * self.M15**-0.13 * self.R500**0.38 * self.E51**0.11 * td[maskspherical]**-0.56 # eV
        #T[maskpos] = 3. * self.M15**-0.13 * self.R500**0.38 * self.E51**0.11 * td[maskpos]**-0.56 # eV

        return T * eV / k # K

    def R(self, **kwargs):
        
        # time in hours
        thr = kwargs["thr"]

        return np.sqrt(self.L(thr = thr) / (4. * np.pi * sigmaSB * self.T(thr = thr)**4)) # cm

    # predict supernova flux during rise
    def flux(self, nu, MJD):
        
        # requires MJDs, MJDexp, scale and factor to be defined
        flux = np.zeros(np.shape(MJD))
        thr = (MJD - self.MJDexp) * 24.
        maskpos = (thr > 0)
        flux[maskpos] = self.factor * self.scale * BB(nu, self.R(thr = thr[maskpos]), self.T(thr = thr[maskpos]))
        return flux

    # chi2
    def chi2(self, MJDexp, factor, sqrtR500):
        
        self.R500 = sqrtR500**2
        self.MJDexp = MJDexp
        self.factor = factor
        ADUmodel = self.flux(self.nufit, self.MJDs)
        MJDfit = (ADUmodel >= self.ADUs) | (self.MJDs <= self.maxMJD)
        return np.sum((ADUmodel[MJDfit] - self.ADUs[MJDfit])**2 / self.e_ADUs[MJDfit]**2)

    # fit parameters to observed data
    def fit(self, **kwargs):
    
        # arguments
        self.MJDs = kwargs["MJDs"] # times
        self.ADUs = kwargs["ADUs"] # ADUs
        self.e_ADUs = kwargs["e_ADUs"] # error in ADUs
        self.nufit = kwargs["nu"] # fitting band

        # find maximum time to consider and typical scale ratio between data and ADUs
        mask = self.ADUs > 3. * self.e_ADUs
        self.maxMJD = int(self.MJDs[0]) + 2. * (self.MJDs[mask][np.argmax(self.ADUs[mask] + 2. * self.e_ADUs[mask])] - int(self.MJDs[0]))
        self.factor = 1.
        self.scale = 1.
        self.MJDexp = int(self.MJDs[0])
        self.scale = np.max(self.ADUs) / np.max(self.flux(self.nufit, self.MJDs))
        
        # minimize using Minuit
        m = Minuit(self.chi2, MJDexp = int(self.MJDs[0]), factor = 1., sqrtR500 = 1., error_MJDexp = 0.1, error_factor = 0.1, error_sqrtR500 = 0.01, limit_factor = (1e-2, 1e2), limit_sqrtR500 = (5e-2, 3), print_level = 0, errordef = 1.)
        m.migrad()
        (self.MJDexp, self.factor, self.sqrtR500) = (m.values["MJDexp"], m.values["factor"], m.values["sqrtR500"])
        (self.e_MJDexp, self.e_factor, self.e_sqrtR500) = (m.errors["MJDexp"], m.errors["factor"], m.errors["sqrtR500"])
        self.R500 = self.sqrtR500**2
        self.e_R500 = 0.5 / self.sqrtR500 * self.e_sqrtR500

# early evolution SN light curves from Rabinak & Waxman 2011
class SNIIRabinakWaxman(object):

    def __init__(self, **kwargs):
        
        self.Msun = kwargs["Msun"]
        self.E51 = kwargs["E51"]
        self.R13 = kwargs["R13"]
        self.kappa34 = kwargs["kappa34"] # opacity in units of 0.34 cm2/g
        self.q = kwargs["q"]  # core mass fraction 

        self.frho = (12.62 - 12.17 * self.q) * (0.37 - 0.18 * self.q + 0.096 * self.q**2)**2.5 * 3. / (4. * np.pi)

    # minimum time of validity
    def tmin(self):
        return 5e3 * self.Msun**0.6 * self.R13**1.4 * self.kappa34**-0.2 * self.E51**-0.9 # sec

    # maximum time of validity
    def tmax(self):
        
        return 4.5e6 * (self.frho / 0.07)**0.69 * self.Msun * self.kappa34**0.5 * self.E51**-0.5 # sec

    # radius evolution with time for convective envelope
    def R(self, **kwargs):
        
        t5 = kwargs["t5"]
        maskpos = (t5 > 0)
        R = np.zeros(np.shape(t5))
        R[maskpos] = 3.3e14 * self.frho**-0.062 * self.E51**0.41 * self.kappa34**0.093 / self.Msun**0.31 * t5[maskpos]**0.81 # cm
        return R 

    # temperature evolution with time for convective envelopes
    def T(self, **kwargs):

        t5 = kwargs["t5"]
        maskpos = (t5 > 0)
        T = np.zeros(np.shape(t5))
        T[maskpos] = 1.6 * self.frho**-0.037 * self.E51**0.027 * self.R13**0.25 / self.Msun**0.054 / self.kappa34**0.28 * t5[maskpos]**-0.45 * eV / k # K
        return T

    # predict supernova flux during rise
    def flux(self, nu, MJD):
        
        # requires MJDs, MJDexp, scale and factor to be defined
        flux = np.zeros(np.shape(MJD))
        t5 = (MJD - self.MJDexp) * 24. * 3600. / 1e5
        maskpos = (t5 > 0)
        flux[maskpos] = self.factor * self.scale * BB(nu, self.R(t5 = t5[maskpos]), self.T(t5 = t5[maskpos]))
        return flux

    # chi2
    def chi2(self, MJDexp, factor, sqrtR500):
        
        self.MJDexp = MJDexp
        self.factor = factor
        self.R13 = sqrtR500**2 * 500. * Rsun / 1e13
        ADUmodel = self.flux(self.nufit, self.MJDs)
        MJDfit = (ADUmodel >= self.ADUs) | (self.MJDs <= self.maxMJD)
        return np.sum((ADUmodel[MJDfit] - self.ADUs[MJDfit])**2 / self.e_ADUs[MJDfit]**2)

    # fit parameters to observed data
    def fit(self, **kwargs):
    
        # arguments
        self.MJDs = kwargs["MJDs"] # times
        self.ADUs = kwargs["ADUs"] # ADUs
        self.e_ADUs = kwargs["e_ADUs"] # error in ADUs
        self.nufit = kwargs["nu"] # fitting frequency

        # find maximum time to consider and typical scale ratio between data and ADUs
        mask = self.ADUs > 3. * self.e_ADUs
        self.maxMJD = int(self.MJDs[0]) + 2. * (self.MJDs[mask][np.argmax(self.ADUs[mask] + 2. * self.e_ADUs[mask])] - int(self.MJDs[0]))
        self.factor = 1.
        self.scale = 1.
        self.MJDexp = int(self.MJDs[0])
        self.scale = np.max(self.ADUs) / np.max(self.flux(self.nufit, self.MJDs))
        
        # minimize using Minuit
        m = Minuit(self.chi2, MJDexp = int(self.MJDs[0]) - 5, factor = 1., sqrtR500 = 1., error_MJDexp = 0.1, error_factor = 0.1, error_sqrtR500 = 0.01, limit_factor = (1e-2, 1e2), limit_sqrtR500 = (5e-2, 3), print_level = 0, errordef = 1.)
        m.migrad()
        (self.MJDexp, self.factor, self.sqrtR500) = (m.values["MJDexp"], m.values["factor"], m.values["sqrtR500"])
        (self.e_MJDexp, self.e_factor, self.e_sqrtR500) = (m.errors["MJDexp"], m.errors["factor"], m.errors["sqrtR500"])
        self.R13 = self.sqrtR500**2 * 500. * Rsun / 1e13
        self.e_R13 = 0.5 / self.sqrtR500 * self.e_sqrtR500 * 500. * Rsun / 1e13

# early evolution of SNe Ia from Piro et al. 2016
class SNPiro16(object):

    def __init__(self, **kwargs):
        
        self.mixing = kwargs["mixing"]

        self.modeldir = "models/Piro16/Ia/ia_lum_teff"
        self.lumfile = "%s/lum_%s.dat" % (self.modeldir, self.mixing)
        self.Tefffile = "%s/T_eff_%s.dat" % (self.modeldir, self.mixing)
        
    def readmodel(self):
        
        timeL, log10L  = np.loadtxt(self.lumfile).transpose()
        timeT, T  = np.loadtxt(self.Tefffile).transpose()
        if len(log10L) != len(T) or np.max(timeL - timeT) != 0:
            print("WARNING: model L and Teff have different dimensions or times are not consistent")
            sys.exit()
        mask = np.isfinite(log10L) & np.isfinite(T)
        self.timemodel = timeL[mask] / days2sec # days
        self.log10Lmodel = log10L[mask] # log10 erg/s
        self.Teffmodel = T[mask] # K
        self.log10Linterpf = interp1d(self.timemodel, self.log10Lmodel, fill_value = -99, bounds_error = False) # erg/s
        self.Teffinterpf = interp1d(self.timemodel, self.Teffmodel, fill_value = 0, bounds_error = False) # K

    def R(self, **kwargs):
        
        # time in hours
        tdays = kwargs["tdays"]

        R = np.zeros_like(tdays)
        mask = (tdays > 0) & (self.Teffinterpf(tdays) > 0)
        R[mask] = np.sqrt(10**self.log10Linterpf(tdays[mask]) / (4. * np.pi * sigmaSB * self.Teffinterpf(tdays[mask])**4)) # cm
        return R

    # predict supernova flux during rise
    def flux(self, nu, MJD):
        
        # requires MJDs, MJDexp, scale and factor to be defined
        flux = np.zeros(np.shape(MJD))
        tdays = (MJD - self.MJDexp)
        maskpos = (tdays > 0)
        flux[maskpos] = self.factor * self.scale * BB(nu, self.R(tdays = tdays[maskpos]), self.Teffinterpf(tdays[maskpos]))
        return flux


# Piro16 at any redshift
class SNPiro16z(LCz):

    def __init__(self, **kwargs):

        # kwargs
        self.mixing = float(kwargs["mixing"])
        self.times = kwargs["tdays"] # days
        self.doplot = False
        if "doplot" in kwargs.keys():
            self.doplot = kwargs["doplot"]

        # do not look for peaks
        self.dopeaks = False

        # extinction model
        self.extmodel = "CCM89+O94"

        # create lambdas array in optical range
        self.lambdas = np.logspace(3, 4, 160) # 160 bins between 1e3 and 1e4 AA logarithmically spaced 
        self.nlambdas = len(self.lambdas)

        # derived quantities
        self.ntimes = len(self.times)
        self.nu = cspeedAAs / self.lambdas

        # create SN model
        self.modelname = "Piro16_%5.3fmix" % (self.mixing)
        self.SN = SNPiro16(mixing = self.mixing)
        self.SN.readmodel()

        # compute BB flxu
        self.flux = BB(self.nu, self.SN.R(tdays = self.times), self.SN.Teffinterpf(self.times)) * (cspeedAAs / np.atleast_2d(self.lambdas)**2)  # erg/s/AA


# Dado & Dar at any redshift
class SNDDz(LCz):
    
    def __init__(self, **kwargs):

        self.Mc = float(kwargs["Mc"])        # Msun
        self.M56Ni = float(kwargs["M56Ni"])  # Msun
        self.fe = 0.3
        if "fe" in kwargs.keys():
            self.fe = float(kwargs["fe"]) # ~0.3 fraction of free (ionized) electrons
        self.Ae = 0.05
        if "Ae" in kwargs.keys():
            self.Ae = float(kwargs["Ae"]) # ~0.05 ratio between energy released in the decay of 56Co as positron kinetic energy and gamma ray energy 

        self.extmodel = "CCM89+O94"
                
        self.modelname = "DD_Mc%4.2f_M56Ni%4.2f_fe%4.2f_Ae%4.2f" % (self.Mc, self.M56Ni, self.fe, self.Ae)

        self.times = np.array(kwargs["tdays"]) # days
        self.dopeaks = False

        self.doplot = False
        if "doplot" in kwargs.keys():
            self.doplot = kwargs["doplot"]

        # create lambdas array in optical range
        self.lambdas = np.logspace(3, 4, 160) # 160 bins between 1e3 and 1e4 AA logarithmically spaced 
        self.nlambdas = len(self.lambdas)

        # derived quantities
        self.ntimes = len(self.times)
        self.nu = cspeedAAs / self.lambdas

        # create SN model
        self.SN = SNIaDadoDar(Mc = self.Mc, M56Ni = self.M56Ni, fe = self.fe, Ae = self.Ae)
        
        # compute BB flux
        self.flux = BB(self.nu, self.SN.Rphf()(self.times), self.SN.Tf()(self.times)) * (cspeedAAs / np.atleast_2d(self.lambdas)**2)  # erg/s/AA

# Nakar & Sari at any redshift
class SNNSz(LCz):

    def __init__(self, **kwargs):
        
        self.M15 = float(kwargs["M15"])
        self.E51 = float(kwargs["E51"])
        self.R500 = float(kwargs["R500"])
        self.modelname = "NS10_%iMsun_%04.1ffoe_%iRsun" % (self.M15 * 15., self.E51, self.R500 * 500)

        self.times = np.array(kwargs["tdays"]) # days
        self.doplot = False
        if "doplot" in kwargs.keys():
            self.doplot = kwargs["doplot"]
        dotimedelay = False
        if "dotimedelay" in kwargs.keys():
            dotimedelay = kwargs["dotimedelay"]
        self.dopeaks = False


        # create lambdas array in optical range
        self.lambdas = np.logspace(3, 4.5, 160) # 160 bins between 1e3 and 1e4 AA logarithmically spaced 
        self.nlambdas = len(self.lambdas)

        # derived quantities
        self.ntimes = len(self.times)
        self.nu = cspeedAAs / self.lambdas
        thr = self.times * 24.

        # create SN model
        self.SN = SNIINakarSari(M15 = self.M15, E51 = self.E51, R500 = self.R500)

        # compute BB flux
        if dotimedelay:
            tlight = 500. * self.R500 * Rsun / cspeed / 3600. # hr
            mask = self.times * 24. > 20. * tlight
            self.flux = np.zeros((self.ntimes, self.nlambdas))
            self.flux[mask, :] = BB(self.nu, self.SN.R(thr = thr[mask]), self.SN.T(thr = thr[mask])) * (cspeedAAs / np.atleast_2d(self.lambdas)**2)
            ndt = 10
            for i in range(np.sum(np.invert(mask))):
                t = self.times[i] * 24. # hr
                dts = np.linspace(0, min(tlight, t), ndt + 1) # hr
                timehrs = t - dts # hr
                for timehr in timehrs[:-1]:
                    self.flux[i, :] = self.flux[i, :] + BB(self.nu, self.SN.R(thr = timehr), self.SN.T(thr = timehr)) * (cspeedAAs / np.atleast_2d(self.lambdas)**2) / ndt * (timehrs[0] - timehrs[-1]) / tlight # erg/s/AA
        else:
            self.flux = BB(self.nu, self.SN.R(thr = thr), self.SN.T(thr = thr)) * (cspeedAAs / np.atleast_2d(self.lambdas)**2)


# Rabinak & Waxman at any redshift
class SNRWz(LCz):

    def __init__(self, **kwargs):
        
        self.M15 = float(kwargs["M15"])
        self.E51 = float(kwargs["E51"])
        self.R500 = float(kwargs["R500"])
        self.modelname = "RW11_%iMsun_%04.1ffoe_%iRsun" % (self.M15 * 15., self.E51, self.R500 * 500)

        self.times = kwargs["tdays"] # days
        self.doplot = False
        if "doplot" in kwargs.keys():
            self.doplot = kwargs["doplot"]
        self.dopeaks = False

        # create lambdas array in optical range
        self.lambdas = np.logspace(3, 4, 160) # 160 bins between 1e3 and 1e4 AA logarithmically spaced 
        self.nlambdas = len(self.lambdas)

        # derived quantities
        self.ntimes = len(self.times)
        self.nu = cspeedAAs / self.lambdas

        # create SN model
        self.SN = SNIIRabinakWaxman(Msun = 15. * self.M15, E51 = self.E51, R13 = 500. * self.R500 * Rsun / 1e13, kappa34 = 1., q = 0.4)
        
        # compute BB flux
        self.flux = BB(self.nu, self.SN.R(t5 = self.times * 24. * 3600. / 1e5), self.SN.T(t5 = self.times * 24. * 3600. / 1e5)) * (cspeedAAs / np.atleast_2d(self.lambdas)**2)  # erg/s/AA


# -------------------------------
# main program
# -------------------------------

if __name__ == "__main__":

    
    from matplotlib import rc
    rc('text', usetex=True)
    rc('font', family='sans-serif')
    
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    
    fig, ax = plt.subplots()

    filtername = sys.argv[1]
    print(filtername)

    # times
    times = np.logspace(-4, 2.4, 200) * 24. * 60. * 60. # sec
    t5 = times * 1e-5
    thr = times / 3600.
    tdays = thr / 24.

    # Cosmology
    dz = 0.02
    zs = np.array(np.arange(dz, 0.44, dz))
    dz = 0.001
    zs = np.array(np.arange(dz, 0.002, dz))
    nz = len(zs)
    DL = np.zeros(nz)
    Dc = np.zeros(nz)
    Dm = np.zeros(nz)
    h100, omega_m, omega_k, omega_lambda = 0.71, 0.27, 0., 0.73
    for i in range(nz):
        cosmo =  cos_calc.fn_cos_calc(h100, omega_m, omega_k, omega_lambda, self.zs[i])
        Dc[i] = cosmo[1] # Mpc
        DL[i] = cosmo[4] # Mpc
        Dm[i] = cosmo[5] # mag

    ## test SN Ia model
    #SNIa = SNDDz(Mc = 1.1, M56Ni = 0.84, fe = 0.5, Ae = 0.18, tdays = tdays)
    ##fig, ax = plt.subplots()
    ##nugreen = 590e12 # green
    ##nured = 460e12 # green
    ##ax.plot(tdays, BB(nugreen, SNIa.Rphf()(tdays), SNIa.Tf()(tdays)), c = 'g')
    ##ax.plot(tdays, BB(nured, SNIa.Rphf()(tdays), SNIa.Tf()(tdays)), c = 'r')
    ##ax.set_xlim(0, 50)
    ##ax.set_ylim(0, 1e16)
    ##plt.savefig("plots/SNIa.png")
    ##SNIa = SNDDz(Mc = 1.1, M56Ni = 0.84, fe = 0.5, Ae = 0.18, tdays = tdays, doplot = True)
    ##SNIa.luminosity()
    ##SNIa.redshift(zs = zs, DL = DL)
    #sys.exit()
    

    # Test attenuation
    fig, ax = plt.subplots()
    NS = SNNSz(E51 = 1., M15 = 1., R500 = 1., doplot = False, tdays = tdays, dotimedelay = False)
    NS.luminosity()
    NS.redshift(zs = zs, DL = DL)
    ext = extinction(Av = 1, Rv = 2.75, model = NS.extmodel)
    (x, att) = ext.dm(lambdasAA = NS.lambdas)
    ax.plot(x, att, label = "Av = 1, Rv = 2.75")
    ext = extinction(Av = 1, Rv = 3.52, model = NS.extmodel)
    (x, att) = ext.dm(lambdasAA = NS.lambdas)
    ax.plot(x, att, label = "Av = 1, Rv = 3.52")
    ext = extinction(Av = 1, Rv = 5.3, model = NS.extmodel)
    (x, att) = ext.dm(lambdasAA = NS.lambdas)
    ax.plot(x, att, label = "Av = 1, Rv = 5.3")
    ax.axvline(1./0.55)
    ax.set_xlim(0, 3)
    ax.set_xlabel(r'1/$\lambda$ [$\mu$m$^{-1}$]')
    ax.set_ylabel("Attenuation")
    ax.legend(loc = 2)
    plt.savefig("plots/extinction.png")

    # get T11 parameters
    from T11modelpars import *
    T11pars = T11modelpars()

    fig, ax = plt.subplots(nrows = 2, sharex = True, figsize = (10, 12))
    # plot example NS10 model similar to Figure 3
    NS = SNNSz(E51 = 1., M15 = 1., R500 = 1., doplot = False, tdays = tdays, dotimedelay = False)
    NS.luminosity()
    NS.redshift(zs = zs, DL = DL)
    NS.mags(Dm = Dm, filtername = 'V')
    NS.mags(Dm = Dm, filtername = 'V')
    ax[0].plot(tdays, NS.absmagz[0], c = 'k', label = "$NS10 ~M_{15}: %3.1f,~ E_{51}: %3.1f,~ R_{500}: %3.1f,~ V$, no time delay" % (NS.M15, NS.E51, NS.R500), ls = '--')
    ax[1].plot(tdays, NS.SN.L(thr = thr), c = 'k', label = 'L')
    ax[1].set_yscale('log')
    ax[1].set_ylabel('L [erg/s]')
    ax[1].set_ylim(1e42, 1e46)
    ax[1].legend(loc = 2)
    ax2 = ax[1].twinx()
    ax2.plot(tdays, NS.SN.T(thr = thr), c = 'r', label = 'T')
    ax2.set_yscale('log')
    ax2.set_ylabel('T [K]')
    ax2.set_ylim(1e4, 1e6)
    ax2.legend(loc = 1)


    #NS.mags(Dm = Dm, filtername = 'u')
    #ax.plot(tdays, NS.absmagz[0], c = 'b', label = "$NS ~M_{15}: %3.1f,~ E_{51}: %3.1f,~ R_{500}: %3.1f,~ u$, no time delay" % (NS.SN.M15, NS.SN.E51, NS.SN.R500), ls = '--')
    #NS.mags(Dm = Dm, filtername = 'g')
    #ax.plot(tdays, NS.absmagz[0], c = 'g', label = "$NS ~M_{15}: %3.1f,~ E_{51}: %3.1f,~ R_{500}: %3.1f,~ g$, no time delay" % (NS.SN.M15, NS.SN.E51, NS.SN.R500), ls = '--')
    #NS.mags(Dm = Dm, filtername = 'r')
    #ax.plot(tdays, NS.absmagz[0], c = 'r', label = "$NS ~M_{15}: %3.1f,~ E_{51}: %3.1f,~ R_{500}: %3.1f,~ r$, no time delay" % (NS.SN.M15, NS.SN.E51, NS.SN.R500), ls = '--')

    NS = SNNSz(E51 = 1., M15 = 1., R500 = 1., doplot = False, tdays = tdays, dotimedelay = True)
    NS.luminosity()
    NS.redshift(zs = zs, DL = DL)
    NS.mags(Dm = Dm, filtername = 'V')
    u = NS.absmagz[0]
    ax[0].plot(tdays, NS.absmagz[0], c = 'k', label = "$NS10 ~M_{15}: %3.1f,~ E_{51}: %3.1f,~ R_{500}: %3.1f,~ V$" % (NS.SN.M15, NS.SN.E51, NS.SN.R500))


    RW = SNRWz(E51 = 1., M15 = 1., R500 = 1., doplot = False, tdays = tdays, dotimedelay = False)
    RW.luminosity()
    RW.redshift(zs = zs, DL = DL)
    RW.mags(Dm = Dm, filtername = 'V')
    ax[0].plot(tdays, RW.absmagz[0], c = 'k', label = "$RW11 ~M_{15}: %3.1f,~ E_{51}: %3.1f,~ R_{500}: %3.1f,~ V$, no time delay" % (RW.M15, RW.E51, RW.R500), ls = ':')

    #NS.mags(Dm = Dm, filtername = 'u')
    #u = NS.absmagz[0]
    #ax.plot(tdays, NS.absmagz[0], c = 'b', label = "$NS ~M_{15}: %3.1f,~ E_{51}: %3.1f,~ R_{500}: %3.1f,~ u$" % (NS.SN.M15, NS.SN.E51, NS.SN.R500))
    #NS.mags(Dm = Dm, filtername = 'g')
    #g = NS.absmagz[0]
    #ax.plot(tdays, NS.absmagz[0], c = 'g', label = "$NS ~M_{15}: %3.1f,~ E_{51}: %3.1f,~ R_{500}: %3.1f,~ g$" % (NS.SN.M15, NS.SN.E51, NS.SN.R500))
    #NS.mags(Dm = Dm, filtername = 'r')
    #r = NS.absmagz[0]
    #ax.plot(tdays, NS.absmagz[0], c = 'r', label = "$NS ~M_{15}: %3.1f,~ E_{51}: %3.1f,~ R_{500}: %3.1f,~ r" % (NS.SN.M15, NS.SN.E51, NS.SN.R500))
    #ax.plot(tdays, g - 0.59 * (g - r) - 0.01, label = "g - 0.59 * (g - r) - 0.01 $\sim$ V", c = 'k')

    #NS = SNNSz(E51 = 1., M15 = 1., R500 = 1., doplot = False, tdays = tdays, dotimedelay = True)
    #NS.luminosity()
    #NS.attenuate(Av = 1, Rv = 3.1)
    #NS.redshift(zs = zs, DL = DL)
    #NS.mags(Dm = Dm, filtername = 'u')
    #ax.plot(tdays, NS.absmagz[0], c = 'b', label = "$NS ~M_{15}: %3.1f,~ E_{51}: %3.1f,~ R_{500}: %3.1f,~ u, A_{V} = 1, Rv = 3.1$" % (NS.SN.M15, NS.SN.E51, NS.SN.R500), ls = ':')
    #NS.mags(Dm = Dm, filtername = 'g')
    #ax.plot(tdays, NS.absmagz[0], c = 'g', label = "$NS ~M_{15}: %3.1f,~ E_{51}: %3.1f,~ R_{500}: %3.1f,~ g, A_{V} = 1, R_{V} = 3.1$" % (NS.SN.M15, NS.SN.E51, NS.SN.R500), ls = ':')
    #NS.mags(Dm = Dm, filtername = 'r')
    #ax.plot(tdays, NS.absmagz[0], c = 'r', label = "$NS ~M_{15}: %3.1f,~ E_{51}: %3.1f,~ R_{500}: %3.1f,~ r, A_{V} = 1, R_{V} = 3.1$" % (NS.SN.M15, NS.SN.E51, NS.SN.R500), ls = ':')
    ##
    #print(T11pars.models[9])
    #SN = T11(dir = "models", modelfile = "15z002E1.dat", doplot = False)
    #SN.luminosity()
    #SN.redshift(zs = zs, DL = DL)
    #SN.mags(Dm = Dm, filtername = 'u')
    #ax.plot(SN.timesz[0] - SN.tstartpeak[0], SN.absmagz[0], c = 'k', label = "$T11 ~M_{15}: %3.1f,~ E_{51}: %3.1f,~ R_{500}: %3.2f,~ u$" % (T11pars.Mzams[9] / 15., T11pars.E51[9], T11pars.Rpresn[9] / 500.), ls = '-')
    #SN.mags(Dm = Dm, filtername = 'g')
    #ax.plot(SN.timesz[0] - SN.tstartpeak[0], SN.absmagz[0], c = 'k', label = "$T11 ~M_{15}: %3.1f,~ E_{51}: %3.1f,~ R_{500}: %3.2f,~ g$" % (T11pars.Mzams[9] / 15., T11pars.E51[9], T11pars.Rpresn[9] / 500.), ls = '-')
    #SN.mags(Dm = Dm, filtername = 'r')
    #ax.plot(SN.timesz[0] - SN.tstartpeak[0], SN.absmagz[0], c = 'k', label = "$T11 ~M_{15}: %3.1f,~ E_{51}: %3.1f,~ R_{500}: %3.2f,~ r$" % (T11pars.Mzams[9] / 15., T11pars.E51[9], T11pars.Rpresn[9] / 500.), ls = '-')
    #SN = T11(dir = "models", modelfile = "15z002E1.dat", doplot = False)
    #SN.luminosity()
    #SN.attenuate(Av = 1, Rv = 3.4)
    #SN.redshift(zs = zs, DL = DL)
    #SN.mags(Dm = Dm, filtername = 'u')
    #ax.plot(SN.timesz[0] - SN.tstartpeak[0], SN.absmagz[0], c = 'k', label = "$T11 ~M_{15}: %3.1f,~ E_{51}: %3.1f,~ R_{500}: %3.2f,~ u, A_{V} = 1, R_{V} = 3.1$" % (T11pars.Mzams[9] / 15., T11pars.E51[9], T11pars.Rpresn[9] / 500.), ls = ':')
    #SN.mags(Dm = Dm, filtername = 'g')
    #ax.plot(SN.timesz[0] - SN.tstartpeak[0], SN.absmagz[0], c = 'k', label = "$T11 ~M_{15}: %3.1f,~ E_{51}: %3.1f,~ R_{500}: %3.2f,~ g, A_{V} = 1, R_{V} = 3.1$" % (T11pars.Mzams[9] / 15., T11pars.E51[9], T11pars.Rpresn[9] / 500.), ls = ':')
    #SN.mags(Dm = Dm, filtername = 'r')
    #ax.plot(SN.timesz[0] - SN.tstartpeak[0], SN.absmagz[0], c = 'k', label = "$T11 ~M_{15}: %3.1f,~ E_{51}: %3.1f,~ R_{500}: %3.2f,~ r, A_{V} = 1, R_{V} = 3.1$" % (T11pars.Mzams[9] / 15., T11pars.E51[9], T11pars.Rpresn[9] / 500.), ls = ':')
    
    ax[0].axvline(NS.SN.t0() / 24.)
    ax[0].axvline(NS.SN.t0() / NS.SN.ct0R() / 24.)
    ax[1].set_xlabel("t [days]")
    ax[0].set_ylabel("$M_{V}$")
    ax[0].set_xscale('log')
    ax[0].set_xlim(1e-4, 1e1)
    ax[0].set_ylim(-5, -18)
    ax[0].legend(loc = 4, fontsize = 10, framealpha = 0.3)
    fig.subplots_adjust(wspace = 0.01, hspace = 0.01)
    plt.savefig("plots/NS10_fig3.png")

    #print("Done")

    # plot example NS10 model similar to Figure 7
    fig, ax = plt.subplots(nrows = 2, sharex = True)
    NS = SNNSz(E51 = 1.2, M15 = 1.2, R500 = 1.6, doplot = False, tdays = tdays, dotimedelay = True)
    NS.luminosity()
    ax[0].plot(tdays * 24. * 3600., NS.SN.L(thr = thr) / 1.4, c = 'k')
    ax[1].plot(tdays * 24. * 3600., NS.SN.T(thr = thr), c = 'k')
    ax[0].axvline(NS.SN.t0() * 3600.)
    ax[0].axvline(NS.SN.t0() / NS.SN.ct0R() * 3600.)
    ax[1].axvline(NS.SN.t0() * 3600.)
    ax[1].axvline(NS.SN.t0() / NS.SN.ct0R() * 3600.)
    ax[0].set_xscale('log')
    ax[0].set_xlim(1e3, 1e6)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel("Time [sec]")
    ax[0].set_ylabel("L [erg/s]")
    ax[1].set_ylabel("T[K]")
    ax[0].set_ylim(1e42, 3e45)
    ax[1].set_ylim(9e3, 4e5)
    ax[1].axvline(NS.SN.t0() / 24.)
    plt.savefig("plots/NS10_fig7.png")

    #sys.exit()

    # Compare rising light curves for T11 vs NS and RW
    ##################################################
    fig, ax = plt.subplots(ncols = 4, nrows = 4, figsize = (16, 10), sharex = True, sharey = True)
    ix = 0
    iy = 0
    for SN, Mzams, Rpresn, E51 in zip(T11pars.models, T11pars.Mzams, T11pars.Rpresn, T11pars.E51):
    
        # line width
        lw = 0.5 #(Rpresn / 250. / 2.)**2
    
        # initialize examples of Tominaga et al. models
        SN = T11(dir = "models", modelfile = SN, doplot = False)
        SN.luminosity()
        SN.redshift(zs = zs, DL = DL)
        SN.mags(filtername = filtername, Dm = Dm, doplot = False)
        ax[ix, iy].plot((SN.timesz[0] - SN.tstartpeak[0]), SN.absmagz[0], label = "T11 %4.1f Msun, %i Rsun, %3.1f foe" % (Mzams, Rpresn, E51), lw = 1, c = 'k', zorder = 1000)
    
        # independent parameters for SN10 and RW11
        v2 = E51 / Mzams
        rv2 = Rpresn * v2
        vkms = np.sqrt(E51 * 1e51 / (Mzams * Msun)) * 1e-8 # 1e3 km/s
    
        for Mzams in range(8, 30, 5):
            
            label = ''
            labelmin = ''
            labelmax = ''
            label5e3 = ''
            label1ev = ''

            E51 = v2 * Mzams
            # NS10 model
            NS = SNNSz(M15 = Mzams / 15., E51 = E51, R500 = Rpresn / 500., tdays = tdays, dotimedelay = True)
            NS.luminosity()
            NS.redshift(zs = zs, DL = DL)
            NS.mags(Dm = Dm, filtername = filtername)
            if Mzams == 8:
                label = "NS10 %i Rsun, v: %3.1f 1e3 kms/s" % (Rpresn, vkms)
                label5e3 = "$T_{RSG} = 5000~ K$"
                label1ev = "$T_{RSG} = 1~ eV$"

            ax[ix, iy].plot(tdays, NS.absmagz[0], label = label, lw = lw, c = 'r')
            recomb = tdays > 5
            T = NS.SN.T(thr = thr[recomb])
            ax[ix, iy].axvline(tdays[recomb][np.argmin(np.abs(T - 5e3))], lw = 3, c = 'r', alpha = 0.3, label = label5e3)
            ax[ix, iy].axvline(tdays[recomb][np.argmin(np.abs(T - eV / k))], lw = 1, ls = ':', c = 'r', alpha = 0.3, label = label1ev)
    
            # RW11 model
            RW = SNRWz(M15 = Mzams / 15., E51 = E51, R500 = Rpresn / 500., tdays = tdays)
            RW.luminosity()
            RW.redshift(zs = zs, DL = DL)
            RW.mags(Dm = Dm, filtername = filtername)
            if Mzams == 8:
                label = "RW11 %i Rsun, v: %3.1f 1e3 kms/s" % (Rpresn, vkms)
                label5e3 = "$T_{RSG} = 5000~ K$"
                label1ev = "$T_{RSG} = 1~ eV$"
                labelmin = "SBO dominated"
                labelmax = "curvature effects"
            ax[ix, iy].plot(tdays, RW.absmagz[0], label = label, lw = lw, c = 'b')
            T = RW.SN.T(t5 = t5[recomb])
            ax[ix, iy].axvline(tdays[recomb][np.argmin(np.abs(T - 5e3))], lw = 3, c = 'b', alpha = 0.3, label = label5e3)
            ax[ix, iy].axvline(tdays[recomb][np.argmin(np.abs(T - eV / k))], lw = 1, ls = ':', c = 'b', alpha = 0.3, label = label1ev)
            ax[ix, iy].axvline(RW.SN.tmin() / 3600. / 24., lw = 1, ls = '--', c = 'b', alpha = 0.3, label = labelmin)
            #ax[ix, iy].axvline(RW.SN.tmax() / 3600. / 24., lw = 1, ls = '-.', c = 'b', alpha = 0.3, label = labelmax)
    
        ax[ix, iy].set_ylim(-8, -20.2)
        ax[ix, iy].set_xscale('log')
        ax[ix, iy].set_xlim(1e-3, 1e2)
        if iy == 0:
            ax[ix, iy].set_ylabel(r'$M_%s$' % filtername, fontsize = 10)
        if ix == 3:
            ax[ix, iy].set_xlabel("Time [days]", fontsize = 10)
        ax[ix, iy].legend(fontsize = 6, loc = 4, framealpha = 0.6)
        
        print(ix, iy)
        ix = ix + 1
        if np.mod(ix, 4) == 0:
            iy = iy + 1
            ix = 0
    
    fig.subplots_adjust(wspace = 0.01, hspace = 0.01)
    plt.savefig("plots/T11-NS-RW_%s.png" % filtername)

    #sys.exit()

    # best-fitting power law indices
    bT11 = 0.56
    bNS = 0.69
    bRW = 0.56
    bfT11 = -2.1
    bfNS = -2.3
    bfRW = -2.3

    # get characteristic parameters to sort light curves and initialize color map
    fv2 = []
    rv2 = []
    stretch = []
    for SN, Mzams, Rpresn, E51 in zip(T11pars.models, T11pars.Mzams, T11pars.Rpresn, T11pars.E51):
        #fv2i = 24. * 3600. * np.sqrt(E51 * 1e51 / (Mzams * Msun)) / (1000. * Rsun)
        #rv2i = Rpresn * E51 / Mzams
        fv2i = np.sqrt(E51 / (Mzams / 15.))
        rv2i = (Rpresn / 500.) * E51 / (Mzams / 15.)
        fv2.append(fv2i)
        rv2.append(rv2i)
        stretch.append(fv2i / rv2i**bT11)
    rv2 = np.array(rv2)
    idx = np.argsort(rv2)
    jet = cm = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin = np.log10(min(rv2)), vmax = np.log10(max(rv2)))
    scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = jet)

    # compare empirical light curves scaling the time axis
    fig, ax = plt.subplots(ncols = 2, nrows = 2, figsize = (15, 10))
    riseT11 = []
    riseNS = []
    riseRW = []
    maxT11 = []
    maxNS = []
    maxRW = []

    RsNS = []
    RsRW = []

    for SN, Mzams, Rpresn, E51 in zip(T11pars.models[idx], T11pars.Mzams[idx], T11pars.Rpresn[idx], T11pars.E51[idx]):

        print("E51", E51, "Msun", Mzams, "Rsun", Rpresn)
        v2 = E51 / (Mzams / 15.) # E51/M15
        rv2 = (Rpresn / 500.) * v2 # R500 R51 / M15
        fv2 = np.sqrt(v2)
        vkms = np.sqrt(E51 * 1e51 / (Mzams * Msun)) * 1e-8 # 1e3 km/s
        lw = 0.5 #(Rpresn / 250. / 2.)**2

        # get color
        colorVal = scalarMap.to_rgba(np.log10(rv2))

        # initialize examples of Tominaga et al. models
        SN = T11(dir = "models", modelfile = SN, doplot = False)
        SN.luminosity()
        SN.redshift(zs = zs, DL = DL)
        SN.mags(filtername = filtername, Dm = Dm, doplot = False)

        # plot
        flux = 10**(-SN.absmagz[0] / 2.5)
        ax[0, 0].plot((SN.timesz[0] - SN.tstartpeak[0]), SN.absmagz[0], label = "T11 %4.1f Msun, %i Rsun, %3.1f foe, %i km/s" % (Mzams, Rpresn, E51, vkms * 1000), lw = 1, c = colorVal, zorder = 1000)
        ax[0, 1].plot((SN.timesz[0] - SN.tstartpeak[0]) * fv2, flux / max(flux), label = "T11 %4.1f Msun, %i Rsun, %3.1f foe, %i km/s" % (Mzams, Rpresn, E51, vkms * 1000), lw = 1, c = colorVal, zorder = 1000)
        arg = np.argmax(flux)
        riseT11.append([rv2, SN.timesz[0][arg] * fv2, SN.absmagz[0][arg], Mzams, Rpresn, E51])

        # overplot light curve from Nakar & Sari
        NS = SNNSz(M15 = Mzams / 15., E51 = E51, R500 = Rpresn / 500., tdays = tdays, dotimedelay = True)
        NS.luminosity()
        NS.redshift(zs = zs, DL = DL)
        NS.mags(filtername = filtername, Dm = Dm, doplot = False)
        ax[0, 0].plot(tdays, NS.absmagz[0], lw = 1, ls = ':', c = colorVal, zorder = 1000)

        ## fit light curve using RW and NS model
        #for delta in range(15):
        #    phase = np.random.random() * 12 - 6
        #    dates = np.array([0., 0.06864565, 0.11572704, 0.84329641, 0.91077998, 0.97881667, 1.0489714, 1.11795946, 1.84005084, 1.90740449, 1.97716794, 2.0459449, 2.11505815, 2.83737369, 3.83473416, 4.83249566, 4.89986671, 4.96802681, 5.03690001, 5.10556929, 6.88464669, 6.94614565, 7.95347101, 9.87601682, 9.9441805, 20.00324065, 24.94381797])
        #    datescomp = dates
        #    #tfit = SN.timesz[0] - SN.tpeak[0] + phase
        #    tfit = SN.timesz[0] + phase
        #    masktime = np.ones(np.shape(tfit), dtype = bool)
        #    for i in range(np.sum(masktime)):
        #        if len(datescomp) > 0 and tfit[i] > datescomp[0]:
        #            datescomp = np.delete(datescomp, 0)
        #        else:
        #            masktime[i] = False
        #    masktime[0] = False
        #    maskpreexp = dates < min(tfit[masktime])
        #    if np.sum(maskpreexp) > 0:
        #        tfit = np.hstack([dates[maskpreexp], tfit[masktime]])[1:]
        #        fluxfit = np.hstack([np.zeros(np.sum(maskpreexp)), flux[masktime]])[1:]
        #    else:
        #        tfit = tfit[masktime][1:]
        #        fluxfit = flux[masktime][1:]
        #    dozeroflux = False
        #    if dozeroflux:
        #        fluxfit = fluxfit - fluxfit[0]
        #    if max(fluxfit) == fluxfit[0]:
        #        continue
        #    NS = SNIINakarSari(M15 = 1., E51 = 1., R500 = 1.)
        #    NS.fit(MJDs = tfit, ADUs = fluxfit, e_ADUs = np.ones(np.shape(tfit)) * max(fluxfit) * 0.001, nu = nus[1])
        #    RsNS.append([phase, Rpresn, E51, Mzams, NS.MJDexp, NS.R500 * 500., 1., 15.])
        #    RW = SNIIRabinakWaxman(Msun = 15., E51 = 1., R13 = 500. * Rsun / 1e13, kappa34 = 1., fp = 0.1)
        #    RW.fit(MJDs = tfit, ADUs = fluxfit, e_ADUs = np.ones(np.shape(tfit)) * max(fluxfit) * 0.001, nu = nus[1])
        #    RsRW.append([phase, Rpresn, E51, Mzams, RW.MJDexp, RW.R13 * 1e13 / Rsun, 1., 15.])
        #    fluxRW = RW.flux(nus[1], tfit)
        #    ax[0, 0].scatter(tfit - phase, fluxRW, facecolor = colorVal, edgecolor = 'none', alpha = 0.5, zorder = 1000)
        #    ax[0, 1].scatter((tfit - phase) * fv2, fluxRW / max(fluxRW), facecolor = colorVal, edgecolor = 'none', alpha = 0.5, zorder = 1000)
        
        # independent parameters for SN10 and RW11
        for Mzams in np.arange(8, 30, 5):
            
            label = ''
            E51 = v2 * Mzams
            # NS10 model
            NS = SNNSz(M15 = Mzams / 15., E51 = E51, R500 = Rpresn / 500., tdays = tdays, dotimedelay = True)
            NS.luminosity()
            NS.redshift(zs = zs, DL = DL)
            NS.mags(Dm = Dm, filtername = filtername)
            if Mzams == 8:
                label = "NS10 %i Rsun, v: %3.1f 1e3 kms/s" % (Rpresn, vkms)
            
            flux = 10**(-NS.absmagz[0] / 2.5)
            ax[1, 0].plot(tdays * fv2, flux / max(flux[tdays > 3]), label = label, lw = lw, c = colorVal)
            arg = np.argmax(flux)
            riseNS.append([rv2, tdays[arg] * fv2, NS.absmagz[0][arg], Mzams, Rpresn, E51])

            # RW11 model
            RW = SNRWz(M15 = Mzams / 15., E51 = E51, R500 = Rpresn / 500., tdays = tdays)
            RW.luminosity()
            RW.redshift(zs = zs, DL = DL)
            RW.mags(Dm = Dm, filtername = filtername)
            if Mzams == 8:
                label = "RW11 %i Rsun, v: %3.1f 1e3 kms/s" % (Rpresn, vkms)

            flux = 10**(-RW.absmagz[0] / 2.5)
            ax[1, 1].plot(tdays * fv2, flux / max(flux), label = label, lw = lw, c = colorVal)
            arg = np.argmax(flux)
            riseRW.append([rv2, tdays[arg] * fv2, RW.absmagz[0][arg], Mzams, Rpresn, E51])
                
    ax[0, 0].set_ylabel("$M_%s$" % filtername, fontsize = 10)
    ax[0, 1].set_ylabel("normalized %s flux" % filtername, fontsize = 10)
    ax[1, 0].set_ylabel("normalized %s flux" % filtername, fontsize = 10)
    ax[1, 1].set_ylabel("normalized %s flux" % filtername, fontsize = 10)
    ax[0, 0].set_ylim(-13, -20.2)
    ax[0, 0].set_xlim(-1, 30)
    ax[0, 1].set_xlim(-1, 30)
    ax[1, 0].set_xlim(-1, 30)
    ax[1, 1].set_xlim(-1, 30)
    ax[0, 0].set_xlabel("$t$ [days]", fontsize = 10)
    ax[0, 1].set_xlabel("$t \sqrt{E51/M15}$ [days]", fontsize = 10)
    ax[1, 0].set_xlabel("$t \sqrt{E51/M15}$ [days]", fontsize = 10)
    ax[1, 1].set_xlabel("$t \sqrt{E51/M15}$ [days]", fontsize = 10)
    ax[0, 1].legend(fontsize = 6, loc = 4)
    ax[0, 0].grid()
    ax[0, 1].grid()
    ax[1, 0].grid()
    ax[1, 1].grid()
    ax[0, 0].set_title("T11")
    ax[0, 1].set_title("T11 scaled")
    ax[1, 0].set_title("NS10 scaled")
    ax[1, 1].set_title("RW11 scaled")
    plt.savefig("plots/T11-NS-RW_scaled_%s.png" % filtername)

    ## plot results of radius fitting
    #RsNS = np.array(RsNS).transpose()
    #RsRW = np.array(RsRW).transpose()
    #T11exp = RsNS[0]
    #NSexp = RsNS[4]
    #RWexp = RsRW[4]
    #T11parNS = RsNS[1] * (RsNS[2] / RsNS[3])**(1. - 0.5 / 0.68)
    #parNS = RsNS[5] * (RsNS[6] / RsNS[7])**(1. - 0.5 / 0.68)
    #T11parRW = RsRW[1] * (RsRW[2] / RsRW[3])**(1. - 0.5 / 0.56)
    #parRW = RsRW[5] * (RsRW[6] / RsRW[7])**(1. - 0.5 / 0.56)
    ## first parameter
    #fig, ax = plt.subplots(ncols = 2, figsize = (12, 6), sharey = True)
    #ax[0].scatter(np.log10(T11parNS), parNS / T11parNS, facecolor = RsNS[2], edgecolor = 'none', label = 'NS', alpha = 0.5, s = (RsNS[6] + 5) * 4)
    #ax[1].scatter(np.log10(T11parRW), parRW / T11parRW, facecolor = RsRW[2], edgecolor = 'none', label = 'RW', alpha = 0.5, s = (RsRW[6] + 5) * 4)
    #ax[0].axhline(1, c = 'gray', ls = ':')
    #ax[0].set_xlabel("T11 R* = R x (E51 / M15)^0.26")
    #ax[0].set_ylabel("Best-fitting R* / T11 R*")
    #ax[1].axhline(1, c = 'gray', ls = ':')
    #ax[1].set_xlabel("T11 R* = R x (E51 / M15)^0.11")
    #ax[0].set_title("NS (average: %4.2f, rms %4.2f)" % (np.average(parNS / T11parNS), np.std(parNS / T11parNS)))
    #ax[1].set_title("RW (average: %4.2f, rms %4.2f)" % (np.average(parRW / T11parRW), np.std(parRW / T11parRW)))
    #fig.subplots_adjust(wspace = 0, hspace = 0)
    #plt.savefig("plots/T11-NS-RW_radiistar_zeroflux%s.png" % dozeroflux)     
    ## radii parameter
    #fig, ax = plt.subplots(ncols = 2, figsize = (12, 6), sharey = True)
    #ax[0].scatter(np.log10(RsNS[1]), RsNS[5] / RsNS[1], facecolor = RsNS[2], edgecolor = 'none', label = 'NS', alpha = 0.5, s = (RsNS[6] + 5) * 4)
    #ax[1].scatter(np.log10(RsRW[1]), RsRW[5] / RsRW[1], facecolor = RsRW[2], edgecolor = 'none', label = 'RW', alpha = 0.5, s = (RsRW[6] + 5) * 4)
    #ax[0].axhline(1, c = 'gray', ls = ':')
    #ax[0].set_xlabel("T11 R [Rsun]")
    #ax[0].set_ylabel("Best-fitting R / T11 R")
    #ax[1].axhline(1, c = 'gray', ls = ':')
    #ax[1].set_xlabel("T11 R [Rsun]")
    #ax[0].set_title("NS (average: %4.2f, rms %4.2f)" % (np.average(RsNS[5] / RsNS[1]), np.std(RsNS[5] / RsNS[1])))
    #ax[1].set_title("RW (average: %4.2f, rms %4.2f)" % (np.average(RsRW[5] / RsRW[1]), np.std(RsRW[5] / RsRW[1])))
    #fig.subplots_adjust(wspace = 0, hspace = 0)
    #plt.savefig("plots/T11-NS-RW_radii_zeroflux%s.png" % dozeroflux)    
    ## explosion times
    #fig, ax = plt.subplots(ncols = 2, figsize = (12, 6), sharey = True)
    #def ticks(x, pos):
    #    return "%i" % (np.round(np.sinh(x)))
    #ax[0].scatter(T11exp, np.arcsinh(NSexp - T11exp), edgecolor = 'none', label = 'NS', alpha = 0.5, facecolor = 'k', zorder = 1e9)
    #ax[1].scatter(T11exp, np.arcsinh(RWexp - T11exp), edgecolor = 'none', label = 'RW', alpha = 0.5, facecolor = 'k', zorder = 1e9) 
    #ax[0].set_xlabel("Explosion date [days]")
    #ax[0].set_ylabel("Best-fitting explosion date - explosion date[days]")
    #ax[1].set_xlabel("Explosion date [days]")
    #ax[0].set_title("NS")
    #ax[1].set_title("RW")
    #ax[0].set_yticks(np.arcsinh([-2000, -500, -200, -50, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 50, 200, 500, 2000, 5000]))
    #ax[1].set_yticks(np.arcsinh([-2000, -500, -200, -50, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 50, 200, 500, 2000, 5000]))
    #ax[0].yaxis.set_major_formatter(plt.FuncFormatter(ticks))
    #ax[1].yaxis.set_major_formatter(plt.FuncFormatter(ticks))
    #fig.subplots_adjust(wspace = 0, hspace = 0)
    #plt.savefig("plots/T11-NS-RW_exptime_zeroflux%s.png" % dozeroflux)    
    #
    #sys.exit()

    # analyze rise time and maximum flux
    riseT11 = np.array(riseT11).transpose()
    riseNS = np.array(riseNS).transpose()
    riseRW = np.array(riseRW).transpose()
    fig, ax = plt.subplots(ncols = 2, figsize = (15, 8))
    line = np.array([1e-3, 1e5])
    def linearfit(x, a, b):
        return a + b * x
    ax[0].scatter(riseT11[0], riseT11[1], edgecolor = 'none', facecolor = 'k', marker = '*', label = 'T11', alpha = 0.5, s = 30)
    (popt, pcov) = curve_fit(linearfit, np.log10(riseT11[0]), np.log10(riseT11[1]), p0 = (0, 1))
    (aT11, bT11) = popt
    ax[0].plot(riseT11[0], 10**aT11 * riseT11[0]**bT11, c = 'k')
    ax[0].scatter(riseNS[0], riseNS[1], edgecolor = 'none', facecolor = 'r', marker = 'o', label = 'NS10', alpha = 0.5, s = 20)
    (popt, pcov) = curve_fit(linearfit, np.log10(riseNS[0]), np.log10(riseNS[1]), p0 = (0, 1))
    (aNS, bNS) = popt
    ax[0].plot(riseNS[0], 10**aNS * riseNS[0]**bNS, c = 'r')
    ax[0].scatter(riseRW[0], riseRW[1], edgecolor = 'none', facecolor = 'b', marker = 'o', label = 'RW11', alpha = 0.5, s = 20)
    (popt, pcov) = curve_fit(linearfit, np.log10(riseRW[0]), np.log10(riseRW[1]), p0 = (0, 1))
    (aRW, bRW) = popt
    ax[0].plot(riseRW[0], 10**aRW * riseRW[0]**bRW, c = 'b')
    print(bT11, bNS, bRW)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].legend(loc = 2)
    ax[0].set_xlabel(r"$R_{500} E_{51} / M_{15}$")
    ax[0].set_ylabel(r"$t_{rise} \sqrt{E_{51}/M_{15}}$ [days]")

    ax[1].scatter(riseT11[0], riseT11[2], edgecolor = 'none', facecolor = 'k', marker = '*', label = 'T11', alpha = 0.5, s = 30)
    (popt, pcov) = curve_fit(linearfit, np.log10(riseT11[0]), riseT11[2], p0 = (-17, 1))
    (afT11, bfT11) = popt
    ax[1].plot(riseT11[0], afT11 + bfT11 * np.log10(riseT11[0]), c = 'k')
    ax[1].scatter(riseNS[0], riseNS[2], edgecolor = 'none', facecolor = 'r', marker = 'o', label = 'NS10', alpha = 0.5, s = 20)
    (popt, pcov) = curve_fit(linearfit, np.log10(riseNS[0]), riseNS[2], p0 = (-17, 1))
    (afNS, bfNS) = popt
    ax[1].plot(riseNS[0], afNS + bfNS * np.log10(riseNS[0]), c = 'r')
    ax[1].scatter(riseRW[0], riseRW[2], edgecolor = 'none', facecolor = 'b', marker = 'o', label = 'RW11', alpha = 0.5, s = 20)
    (popt, pcov) = curve_fit(linearfit, np.log10(riseRW[0]), riseRW[2], p0 = (-17, 1))
    (afRW, bfRW) = popt
    ax[1].plot(riseRW[0], afRW + bfRW * np.log10(riseRW[0]), c = 'b')
    print(bfT11, bfNS, bfRW)
    ax[1].set_xscale('log')
    ax[1].legend(loc = 2)
    ax[1].set_xlabel("$R_{500} E_{51} / M_{15}$")
    ax[1].set_ylabel("$M_%s$ @ max" % filtername)
    plt.savefig("plots/rise_%s.png" % filtername)

    # look at the dispersion at maximum light
    fig, ax = plt.subplots(ncols = 3, figsize = (18, 6))
    maxT11 = riseT11[2] - bfT11 * np.log10(riseT11[0])
    maxNS = riseNS[2] - bfNS * np.log10(riseNS[0])
    maxRW = riseRW[2] - bfRW * np.log10(riseRW[0])
    ax[0].hist(maxT11)
    ax[1].hist(maxNS)
    ax[2].hist(maxRW)
    ax[0].set_xlabel(r'$M_%s @ max + %4.2f \log_{10} ~R_{500} E_{51} / M_{15}$' % (filtername, -bfT11), fontsize = 10)
    ax[1].set_xlabel(r'$M_%s @ max + %4.2f \log_{10} ~R_{500} E_{51} / M_{15}$' % (filtername, -bfNS), fontsize = 10)
    ax[2].set_xlabel(r'$M_%s @ max + %4.2f \log_{10} ~R_{500} E_{51} / M_{15}$' % (filtername, -bfRW), fontsize = 10)
    ax[0].set_title("$T11 (rms: %4.2f)$" % np.std(maxT11))
    ax[1].set_title("$NS (rms: %4.2f)$" % np.std(maxNS))
    ax[2].set_title("$RW (rms: %4.2f)$" % np.std(maxRW))
    plt.savefig("plots/histograms_%s.png" % filtername)
    
    # plot parameters
    
    fig, ax = plt.subplots()
    cax = ax.scatter(riseT11[4], riseT11[5], c = riseT11[3], label = 'RW', edgecolor = 'none')
    ax.set_xlabel("Radius")
    ax.set_ylabel("Mass")
    ax.set_xlabel("Radius [Rsun]")
    ax.set_ylabel("Energy [foe]")
    ax.set_title("T11 parameters")
    ax.set_xscale("log")
    ax.set_yscale("log")
    cbar = plt.colorbar(cax)
    cbar.set_label("Mass [Msun]")
    plt.savefig("plots/pars_T11.png")

    fig, ax = plt.subplots()
    cax = ax.scatter(riseNS[4], riseNS[5], c = riseNS[3], label = 'RW', edgecolor = 'none')
    ax.set_xlabel("Radius [Rsun]")
    ax.set_ylabel("Energy [foe]")
    ax.set_title("RW & NS parameters")
    ax.set_xlabel("Radius [Rsun]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    cbar = plt.colorbar(cax)
    cbar.set_label("Mass [Msun]")
    plt.savefig("plots/pars_RWNS.png")


    # use newly learnt relations to scale both x and y axis
    fig, ax = plt.subplots(ncols = 2, nrows = 2, figsize = (15, 10))
    for SN, Mzams, Rpresn, E51 in zip(T11pars.models[idx], T11pars.Mzams[idx], T11pars.Rpresn[idx], T11pars.E51[idx]):

        print(E51, Mzams, Rpresn)
        v2 = E51 / (Mzams / 15.)
        rv2 = (Rpresn / 500.) * v2
        fv2 = np.sqrt(E51 / (Mzams / 15.))
        fT11 = 1. / rv2**bT11
        fNS = 1. / rv2**bNS
        fRW = 1. / rv2**bRW
        vkms = np.sqrt(E51 * 1e51 / (Mzams * Msun)) * 1e-8 # 1e3 km/s
        lw = 0.5 #(Rpresn / 250. / 2.)**2

        # get color
        colorVal = scalarMap.to_rgba(np.log10(rv2))

        # initialize examples of Tominaga et al. models
        SN = T11(dir = "models", modelfile = SN, doplot = False)
        SN.luminosity()
        SN.redshift(zs = zs, DL = DL)
        SN.mags(filtername = filtername, Dm = Dm, doplot = False)
        # plot
        flux = 10**(-SN.absmagz[0] / 2.5)
        ax[0, 0].plot((SN.timesz[0] - SN.tstartpeak[0]), SN.absmagz[0], label = "T11 %4.1f Msun, %i Rsun, %3.1f foe, %i km/s" % (Mzams, Rpresn, E51, vkms * 1000), lw = 1, c = colorVal, zorder = 1000)
        ax[0, 1].plot((SN.timesz[0] - SN.tstartpeak[0]) * fv2 * fT11, SN.absmagz[0] - bfT11 * np.log10(rv2), label = "T11 %4.1f Msun, %i Rsun, %3.1f foe, %i km/s" % (Mzams, Rpresn, E51, vkms * 1000), lw = 1, c = colorVal, zorder = 1000)
        
        # independent parameters for SN10 and RW11
        for Mzams in np.arange(8, 30, 5):
            
            label = ''
            E51 = v2 * Mzams
            # NS10 model
            NS = SNNSz(M15 = Mzams / 15., E51 = E51, R500 = Rpresn / 500., tdays = tdays, dotimedelay = True)
            NS.luminosity()
            NS.redshift(zs = zs, DL = DL)
            NS.mags(Dm = Dm, filtername = filtername)
            if Mzams == 8:
                label = "NS10 %i Rsun, v: %3.1f 1e3 kms/s" % (Rpresn, vkms)
            
            ax[1, 0].plot(tdays * fv2 * fNS, NS.absmagz[0] - bfNS * np.log10(rv2), label = label, lw = lw, c = colorVal)
            recomb = tdays > 5
            T = NS.SN.T(thr = thr[recomb])
            ax[1, 0].axvline(tdays[recomb][np.argmin(np.abs(T - 5e3))] * fv2 * fNS, lw = lw, c = colorVal)
            ax[1, 0].axvline(tdays[recomb][np.argmin(np.abs(T - eV / k))] * fv2 * fNS, lw = 3, c = colorVal)

            # RW11 model
            RW = SNRWz(M15 = Mzams / 15., E51 = E51, R500 = Rpresn / 500., tdays = tdays)
            RW.luminosity()
            RW.redshift(zs = zs, DL = DL)
            RW.mags(Dm = Dm, filtername = filtername)
            if Mzams == 8:
                label = "RW11 %i Rsun, v: %3.1f 1e3 kms/s" % (Rpresn, vkms)

            ax[1, 1].plot(tdays * fv2 * fRW, RW.absmagz[0] - bfRW * np.log10(rv2), label = label, lw = lw, c = colorVal)
            T = RW.SN.T(t5 = t5[recomb])
            ax[1, 1].axvline(tdays[recomb][np.argmin(np.abs(T - 5e3))] * fv2 * fNS, lw = lw, c = colorVal)
            ax[1, 1].axvline(RW.SN.tmin() / 3600. / 24. * fv2 * fNS, lw = 1, c = colorVal)
            ax[1, 1].axvline(RW.SN.tmax() / 3600. / 24. * fv2 * fNS, lw = 3, c = colorVal)
                
    ax[0, 0].set_xlim(-1, 80)
    ax[0, 1].set_xlim(-1, 80)
    ax[1, 0].set_xlim(-1, 80)
    ax[1, 1].set_xlim(-1, 80)
    ax[0, 0].set_ylim(-13, -20.2)
    ax[0, 1].set_ylim(-13, -18)
    ax[1, 0].set_ylim(-15, -20)
    ax[1, 1].set_ylim(-15, -20)
    ax[0, 0].set_xlabel("t [days]", fontsize = 10)
    ax[0, 1].set_xlabel(r'$t\ R_{500}^{%4.2f}  (E_{51} / M_{15})^{%4.2f}$ [days]' % (-bT11, 0.5 - bT11), fontsize = 12)
    ax[1, 0].set_xlabel(r'$t\ R_{500}^{%4.2f}  (E_{51} / M_{15})^{%4.2f}$ [days]' % (-bNS, 0.5 - bNS), fontsize = 12)
    ax[1, 1].set_xlabel(r'$t\ R_{500}^{%4.2f}  (E_{51} / M_{15})^{%4.2f}$ [days]' % (-bRW, 0.5 - bRW), fontsize = 12)
    ax[0, 0].set_ylabel(r'$M_%s$' % filtername, fontsize = 10)
    ax[0, 1].set_ylabel(r'$M_%s + %4.2f\ \log_{10}\ R_{500} E_{51} / M_{15}$' % (filtername, -bfT11), fontsize = 12)
    ax[1, 0].set_ylabel(r'$M_%s + %4.2f\ \log_{10}\ R_{500} E_{51} / M_{15}$' % (filtername, -bfNS), fontsize = 12)
    ax[1, 1].set_ylabel(r'$M_%s + %4.2f\ \log_{10}\ R_{500} E_{51} / M_{15}$' % (filtername, -bfRW), fontsize = 12)
    ax[0, 1].legend(fontsize = 6, loc = 4)
    ax[0, 0].grid()
    ax[0, 1].grid()
    ax[1, 0].grid()
    ax[1, 1].grid()
    ax[0, 0].set_title("T11")
    ax[0, 1].set_title("T11 scaled")
    ax[1, 0].set_title("NS10 scaled")
    ax[1, 1].set_title("RW11 scaled")
    plt.savefig("plots/T11-NS-RW_scaledxy_%s.png" % filtername)

    # use newly learnt relations to scale both x and y axis and plot Rabinak & Waxman light curves
    fig, ax = plt.subplots()
    for SN, Mzams, Rpresn, E51 in zip(T11pars.models[idx], T11pars.Mzams[idx], T11pars.Rpresn[idx], T11pars.E51[idx]):

        print(E51, Mzams, Rpresn)
        v2 = E51 / (Mzams / 15.)
        rv2 = (Rpresn / 500.) * v2
        fv2 = np.sqrt(E51 / (Mzams / 15.))
        fT11 = 1. / rv2**bT11
        fNS = 1. / rv2**bNS
        fRW = 1. / rv2**bRW
        vkms = np.sqrt(E51 * 1e51 / (Mzams * Msun)) * 1e-8 # 1e3 km/s
        lw = 0.5 #(Rpresn / 250. / 2.)**2

        # get color
        colorVal = scalarMap.to_rgba(np.log10(rv2))

        ## initialize examples of Tominaga et al. models
        #SN = T11(dir = "models", modelfile = SN, doplot = False)
        #SN.luminosity()
        #SN.redshift(zs = zs, DL = DL)
        #SN.mags(filtername = filtername, Dm = Dm, doplot = False)
        ## plot
        #flux = 10**(-SN.absmagz[0] / 2.5)
        #ax[0, 0].plot((SN.timesz[0] - SN.tstartpeak[0]), SN.absmagz[0], label = "T11 %4.1f Msun, %i Rsun, %3.1f foe, %i km/s" % (Mzams, Rpresn, E51, vkms * 1000), lw = 1, c = colorVal, zorder = 1000)
        #ax[0, 1].plot((SN.timesz[0] - SN.tstartpeak[0]) * fv2 * fT11, SN.absmagz[0] - bfT11 * np.log10(rv2), label = "T11 %4.1f Msun, %i Rsun, %3.1f foe, %i km/s" % (Mzams, Rpresn, E51, vkms * 1000), lw = 1, c = colorVal, zorder = 1000)
        
        # independent parameters for SN10 and RW11
        for Mzams in np.arange(8, 30, 5):
            
            label = "%i Rsun, %2i Msun, %3.1f foe (v: %3.1f 1e3 km/s)" % (Rpresn, Mzams, E51, vkms)
            if Mzams == 8:
                label = "%i Rsun (v: %3.1f 1e3 km/s)" % (Rpresn, vkms)
            else:
                label = ''

            E51 = v2 * Mzams
            
            # RW11 model
            RW = SNRWz(M15 = Mzams / 15., E51 = E51, R500 = Rpresn / 500., tdays = tdays)
            RW.luminosity()
            RW.redshift(zs = zs, DL = DL)
            RW.mags(Dm = Dm, filtername = filtername)

            ax.plot(tdays * fv2 * fRW, RW.absmagz[0] - bfRW * np.log10(rv2), label = label, lw = lw, c = colorVal)
            T = RW.SN.T(t5 = t5[recomb])
            #ax.axvline(tdays[recomb][np.argmin(np.abs(T - 5e3))] * fv2 * fNS, lw = lw, c = colorVal)
            #ax.axvline(RW.SN.tmin() / 3600. / 24. * fv2 * fNS, lw = 1, c = colorVal)
            #ax.axvline(RW.SN.tmax() / 3600. / 24. * fv2 * fNS, lw = 3, c = colorVal)
                
    ax.set_xlim(-1, 20)
    ax.set_ylim(-18, -21)
    ax.set_xlabel(r'$t\ R_{500}^{%4.2f}  (E_{51} / M_{15})^{%4.2f}$ [days]' % (-bRW, 0.5 - bRW), fontsize = 12)
    ax.set_ylabel(r'$M_%s + %4.2f\ \log_{10}\ R_{500} E_{51} / M_{15}$' % (filtername, -bfRW), fontsize = 12)
    ax.set_title("RW11 scaled")
    ax.legend(framealpha = 0.5, loc = 4, fontsize = 8)
    plt.savefig("plots/RW11_scaledxy_%s.png" % filtername)

