import sys, os
import re
import numpy as np
import ephem
from astropy.time import Time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# observatory information
from observatory import *
from ETC import *
from skymag import *

# DECam specific modues
#sys.path.append("../HiTS-public")
#from DECam_tools import *

class obsplan(object):

    def __init__(self, **kwargs):


        self.obsname = kwargs['obsname']  # Blanco-DECam, Subaru-HSC, LSST
        if 'band' in kwargs.keys():
            self.band = kwargs['band'] # ugrizy or multiband
        self.mode = kwargs['mode'] # custom, some file prefix (TODO)
        doload = False
        if 'doload' in kwargs.keys():
            doload = kwargs['doload']
        dosave = False
        if 'dosave' in kwargs.keys():
            dosave = kwargs['dosave']
        if doload:
            dosave = False
        
        if 'bandcolors' in kwargs.keys():
            self.bandcolors = kwargs["bandcolors"]

        # only for lsst maf
        if self.mode == 'maf':
            self.MJDs = kwargs["MJDs"]
            self.limmag = kwargs["limmag"]
            self.bands = kwargs["bands"]
            self.uniquebands = np.sort(np.unique(self.bands))
            self.nfields = 1.
            self.obs = observatory(observatory = self.obsname)
            #self.exptime = kwargs["visitExpTime"]
            #self.airmasses = kwargs["airmass"]
            #self.skymags = kwargs["fieldSkyBrightness"]
            self.planname = "maf"
            return

        # create observatory
        self.obs = observatory(observatory = self.obsname)

        # initialize exposure time calculator
        seeing_r_CTIO = 0.75
        self.ETC = ETC(observatory = self.obs, seeing_r_arcsec = seeing_r_CTIO)

        # plot
        if "doplot" in kwargs.keys():
            self.doplot = kwargs["doplot"]
        else:
            self.doplot = False

        # main parameters
        self.nfields = kwargs["nfields"]
        self.nread = kwargs["nread"] # number of reads per epoch
        self.nepochspernight = kwargs["nepochspernight"]
        self.nightfraction = kwargs["nightfraction"]
        
        # night length and Moon period
        synodicmoonperiod = 29.5
        self.nightlength = 8.25 / 24.
        
        # custom observational plan (assume next new moon plus starting moon phase as initial MJD)
        if self.mode == 'custom':
        
            self.ncontnights = kwargs["ncontnights"]
            self.nnights = kwargs["nnights"]
            self.startmoonphase = kwargs["startmoonphase"]
            if "maxmoonphase" in kwargs.keys():
                self.maxmoonphase = kwargs["maxmoonphase"]
            else:
                self.maxmoonphase = 1e9
                
            # custom plan name
            self.planname = "%s-nf%i-ne%i-nr%i-nc%i-nn%i_%s_%s" % (self.mode, self.nfields, self.nepochspernight, self.nread, self.ncontnights, self.nnights, self.obsname, self.band)
            print("Observation plan name:", self.planname)
        
            # starting date, new and full moon days
            ephemdate = ephem.Date((Time(Time.now().mjd, format = 'mjd', scale = 'utc').isot).replace("T", " "))
            ephemnextnewmoon = ephem.next_new_moon(ephemdate)
            starttime = Time(str(ephemnextnewmoon).replace("/", "-").replace(" ", "T"), format = 'isot', scale = 'utc').mjd + self.startmoonphase
            
            # MJD times
            dtobs = self.nightlength * self.nightfraction / self.nepochspernight
            dates = []
            for i in range(self.ncontnights):
                for j in range(self.nepochspernight):
                    dates.append(i + j * dtobs)
            self.MJDs = np.array(dates) + starttime

        elif self.mode == "file-csv":

            self.obsplanfile = kwargs["inputfile"]

            params = {}
            input = open("%s/obsplans/%s" % (os.environ["SURVEYSIM_PATH"], self.obsplanfile), 'r')
            for line in input.readlines():
                key, val = line.split()
                params[key] = val

                
            if "dates" in params.keys():
                self.MJDs = []
                for lims in params["dates"].split(","):
                    ab = lims.split(":")
                    if len(ab) == 1:
                        Time(str(ephem.Date(ab[0])).replace("/", "-").replace(" ", "T"), format = 'isot', scale = 'utc').mjd
                        self.MJDs = np.hstack([self.MJDs, mjds])
                    elif len(ab) == 2:
                        mjds = list(map(lambda x: Time(str(ephem.Date(x)).replace("/", "-").replace(" ", "T"), format = 'isot', scale = 'utc').mjd, [ab[0], ab[1]]))
                        self.MJDs = np.hstack([self.MJDs, np.linspace(mjds[0], mjds[1], mjds[1] - mjds[0])])
                    else:
                        print("WARNING: observational plan incorrectly formatted")
                        sys.exit()
            elif "mjds" in params.keys():
                self.MJDs = []
                for mjd in params["mjds"].split(","):
                    self.MJDs = np.hstack([self.MJDs, float(mjd)])
            else:
                print("WARNING: observation times not specified")
                sys.exit()

            # make sure there are no repeated observations
            self.MJDs = np.unique(np.sort(self.MJDs))
            dates = self.MJDs - self.MJDs[0]

            # custom plan name
            self.planname = "%s-nf%i-ne%i-nr%i-nn%i_%s_%s" % (self.obsplanfile[:-4], self.nfields, self.nepochspernight, self.nread, len(self.MJDs), self.obsname, self.band)
            print("Observation plan name:", self.planname)

        elif self.mode == "file-cols":

            self.obsplanfile = kwargs["inputfile"]

            import pandas as pd
            df = pd.read_csv("%s/obsplans/%s" % (os.environ["SURVEYSIM_PATH"], self.obsplanfile), sep = "\s+", comment = '#')

            if 'MJD' in df.columns:
                self.MJDs = np.array(df['MJD'])
            if 'DATE' in df.columns:
                self.MJDs = df['DATE']
                for idx, MJD in enumerate(self.MJDs):
                    if '/' in MJD:
                        self.MJDs[idx] = Time(str(ephem.Date(MJD)).replace("/", "-").replace(" ", "T"), format = 'isot', scale = 'utc').mjd
                self.MJDs = np.array(self.MJDs, dtype = float)

            # eliminate repeated dates and sort
            self.MJDs, idxu = np.unique(self.MJDs, return_index = True)
            idxsorted = np.argsort(self.MJDs)
            self.MJDs = self.MJDs[idxsorted]
            
            if 'FILTER' in df.columns:
                self.bands = np.array(df['FILTER'])
                self.bands = self.bands[idxu][idxsorted]
            if 'AIRMASS' in df.columns:
                self.airmasses = np.array(df['AIRMASS'])
                self.airmasses = self.airmasses[idxu][idxsorted]
            if 'EXPTIME' in df.columns:
                self.exptimes = np.array(df['EXPTIME'])
                self.exptimes = self.exptimes[idxu][idxsorted]
            if 'SKYMAG' in df.columns:
                self.skymags = np.array(df['SKYMAG'])
                self.skymags = self.skymags[idxu][idxsorted]
            if 'MOONPHASE' in df.columns:
                self.moonphases = np.array(df['MOONPHASE'])
                self.moonphases = self.moonphases[idxu][idxsorted]
                
            # find unique bands
            self.bands = self.bands[np.array(idxu, dtype = int)]
            self.uniquebands = np.sort(np.unique(self.bands))
            self.band = "".join(self.uniquebands)

            # find relative dates
            dates = self.MJDs - self.MJDs[0]

            # plan name
            self.planname = "%s-nf%i-ne%i-nr%i-nn%i_%s_%s" % (self.obsplanfile[:-4], self.nfields, self.nepochspernight, self.nread, len(self.MJDs), self.obsname, self.band)
            print("Observation plan name:", self.planname)

        # overhead time
        self.overhead = max(self.obs.readouttime, self.obs.slewtime)
        
        # compute exposure time
        if not hasattr(self, "exptimes"):
            self.exptime = self.nightlength * self.nightfraction / self.nfields / self.nepochspernight * 24. * 60. * 60. - self.overhead - self.obs.readouttime * (self.nread - 1) # seconds
            if self.exptime < 0:
                print("Negative exposure time")
                sys.exit()
            print("Number of fields to visit: %i, exposure time: %s" % (self.nfields, self.exptime))
        else:
            self.exptime = np.median(self.exptimes)

        if dosave or doload:
            outputfile = "%s/obsplans/%s.out" % (os.environ["SURVEYSIM_PATH"], self.obsplanfile)

        if doload:
            self.MJDs, self.bands, self.limmag, self.skymags, self.airmasses, self.moonphases = np.loadtxt(outputfile, dtype = str).transpose()
            self.MJDs = np.array(self.MJDs, dtype = float)
            self.airmasses = np.array(self.airmasses, dtype = float)
            self.limmag = np.array(self.limmag, dtype = float)
            self.moonphases = np.array(self.moonphases, dtype = float)
            self.skymags = np.array(self.skymags, dtype = float)

            return

        if not hasattr(self, "airmasses"):
            # use simple linear model to predict airmass (TODO: alternatively, provide RA DEC list?)
            if self.band == 'u':
                minairmass = 1.0
                maxairmass = 1.5
            elif self.band == 'g' and self.nepochspernight == 4:
                minairmass = 1.2
                maxairmass = 1.8
            elif self.band == 'g' and self.nepochspernight == 5:
                minairmass = 1.1
                maxairmass = 1.7
            else:
                minairmass = 1.1
                maxairmass = 1.7
            
            t_minairmass = self.nightfraction / 2. - self.nightfraction / self.nepochspernight / 2.
            if self.nepochspernight > 1:
                self.airmasses = minairmass + (np.abs(np.mod(dates, 1.) / self.nightlength - t_minairmass) / t_minairmass)**2 * (maxairmass - minairmass)
            else:
                self.airmasses = np.ones_like(dates) * minairmass
            
        # compute moon phases and sky magnitude if necessary
        if not hasattr(self, "moonphases") or not hasattr(self, "skymags"):
            self.moonphases = np.zeros_like(self.MJDs)
            self.skymags = np.zeros_like(self.MJDs)
            for band in self.uniquebands:
                print("Computing moon phase and sky magnitudes for band %s" % band)
                mask = np.array(self.bands == band)
                skymodel = sky(band = band, MJDs = self.MJDs[mask])
                moonphases, skymags = skymodel.skymags()
                self.moonphases[mask] = moonphases
                self.skymags[mask] = skymags

        # using sky magnitudes, airmasses, exposure time, SNR lim, compute limiting magnitudes
        print("Computing limiting magnitudes for all bands (this may take some time)")
        if hasattr(self, "exptimes"):
            self.limmag = np.array(list(map(lambda band, exptime, airmass, skymag: self.ETC.findmag(band=band, SNRin=5., exptime=exptime, airmass=airmass, skymode="mag", skymag=skymag, nread = self.nread), self.bands, self.exptimes, self.airmasses, self.skymags)))
        else:
            self.limmag = np.array(list(map(lambda band, airmass, skymag: self.ETC.findmag(band=band, SNRin=5., exptime=self.exptime, airmass=airmass, skymode="mag", skymag=skymag, nread = self.nread), self.bands, self.airmasses, self.skymags)))

        
        # fix moonphases in custom mode
        if self.mode == 'custom':
            # mask moon phase
            mask = (self.moonphases <= self.maxmoonphase)
            self.totalepochs = np.sum(mask)
            self.MJDs = self.MJDs[mask]
            self.airmasses = self.airmasses[mask]
            self.limmag = self.limmag[mask]
            self.moonphases = self.moonphases[mask]
            self.skymags = self.skymags[mask]


        # save
        if dosave:
            vars2save = np.savetxt(outputfile, np.vstack([self.MJDs, self.bands, self.limmag, self.skymags, self.airmasses, self.moonphases]).transpose(), fmt = "%10s")

            
        # plot
        if self.doplot:

            print("Plotting observational plan")
            
            fig, ax = plt.subplots(figsize = (13, 7))
            
            ax.set_ylabel("%s mag" % self.band)
            
            # plot observing dates
            for i in self.MJDs:
                ax.axvline(i, c = 'k', alpha = 0.01)
                
            # sky magnitude and limiting magnitude
            for band in self.uniquebands:
                mask = (self.bands == band)
                ax.plot(self.MJDs[mask], self.skymags[mask], lw = 4, label = "Sky brightness (%s) [mag / arcsec2]" % band, c = self.bandcolors[band], alpha = 0.8)
                ax.plot(self.MJDs[mask], self.limmag[mask], label = "%s (%s)" % (self.obsname, band), zorder = 1000, c = self.bandcolors[band], alpha = 0.8, marker = '.')
                ax.plot(self.MJDs[mask], self.limmag[mask] - 2.5 * np.log10(np.sqrt(2.)), ls = ':', label = "%s (%s, diff.)" % (band, self.obsname), zorder = 2000, c = self.bandcolors[band], alpha = 0.8)
                #ax.scatter(self.MJDs[mask], self.limmag[mask], marker = 'o', lw = 0, zorder = 1000, c = self.bandcolors[band], alpha = 0.5)

            l1, l2 = ax.get_ylim()
            # plot airmasses, moon phase
            ax.plot(self.MJDs, l2 - (self.airmasses - 1) * (l2 - l1) , c = 'orange', label = "airmass (scale from 1 to 2)", lw = 4, alpha = 0.5)
            ax.plot(self.MJDs, l2 - self.moonphases / (synodicmoonperiod / 2.) * (l2 - l1), c = 'k', lw = 4, label = "Moon phase / half synodic period (0 to 1)", alpha = 0.3)

            # show more dates for reference
            for i in np.arange(int(min(self.MJDs)), int(max(self.MJDs)) + 20):
                ax.axvspan(-10 + self.nightlength + i , -10 + i + 1, alpha = 0.1, color = 'gray', lw = 0)
                
            # function used for plot ticks
            def YYYYMMDD(date, pos):
                return Time(date, format = 'mjd', scale = 'utc').iso[:11]
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(YYYYMMDD))
            
            # final touches
            ax.set_title("%i nights, %i sec median open shutter/epoch (%i readouts/epoch)" % (np.size(self.MJDs) / self.nepochspernight, self.exptime, self.nread))
            ax.set_xlim(min(self.MJDs), max(self.MJDs) + 2)
            ax.legend(loc = 4, fontsize = 8, framealpha = 0.5)
            plt.xticks(rotation = 90, ha = 'left')
            plt.xticks(np.arange(min(self.MJDs) - 2, max(self.MJDs) + 2, max(1, int((max(self.MJDs) - min(self.MJDs)) / 70.))))
            plt.tick_params(pad = 0)
            plt.tick_params(axis='x', which='major', labelsize=10)
            plt.tight_layout()
            ax.set_ylim(ax.get_ylim()[::-1])
            plt.savefig("plots/%s_plan.png" % self.planname)


        ######################################################################
        ## empirical based observational plan, read parameters from fits files
        #######################################################################
        #elif self.mode[0:5] == 'Blind':
        #    
        #    # number of fields
        #    if self.mode[0:8] == "Blind15A":
        #        self.nfields = 50
        #    elif self.mode[0:8] == "Blind14A":
        #        self.nfields = 40
        #    elif self.mode[0:8] == "Blind13A":
        #        self.nfields = 40
        #
        #    # necessary to extract files
        #    self.magdir = kwargs["magdir"]
        #    self.reference = kwargs["reference"]
        #    
        #    # plan name
        #    self.planname = "%s_%s_%s" % (self.mode, self.depthmodel, self.band)
        #    print "Observation plan name:", self.planname
        #
        #    # prepare for plotting
        #    if self.doplot and self.depthmodel in ["obs", "m50"]:
        #        fig, ax = plt.subplots(figsize = (12, 6))
        #
        #    # files with empirical magnitudes
        #    files = np.sort(os.listdir("%s/%s" % (self.magdir, self.mode)))
        #    
        #    # start arrays with relevant data
        #    self.MJDs = []
        #    self.airmasses = []
        #    self.skyADUs = []
        #    self.FWHMs = []
        #    self.exptime = []
        #
        #    # loop over all files to look for reference
        #    for file in files:
        #        if not re.search("%s_%02i_%s_zp.npy" % (self.mode, self.reference, self.band), file):
        #            continue
        #        dataref = np.load("%s/%s/%s" % (self.magdir, self.mode, file))
        #
        #    # loop over all files to compute efficiencies
        #    for file in files:
        #
        #        if not re.search("%s_\d\d_%s_zp.npy" % (self.mode, self.band), file):
        #            continue
        #        else:
        #            epoch = int(re.findall("%s_(\d\d)_%s_zp.npy" % (self.mode, self.band), file)[0])
        #    
        #        # extract basic data
        #        data = np.load("%s/%s/%s" % (self.magdir, self.mode, file))
        #
        #        self.airmasses.append(data["AIRMASS"])
        #        self.FWHMs.append(data["SEEING"])
        #        self.skyADUs.append(data["BACK_LEVEL"])
        #        self.MJDs.append(data['MJD'])
        #        self.exptime.append(data["EXP_TIME"])
        #
        #        if self.depthmodel == "obs" or self.depthmodel == "m50":
        #            
        #            # array used for interpolation
        #            xs = np.linspace(26, 21, 100)
        #
        #            # extract fata for fitting efficiency relations
        #            dataeff = np.load("%s/%s/%s" % (self.magdir, self.mode, file.replace("%s_zp.npy" % self.band, "ratio_complet_zp.npy")))
        #            mask = (dataeff[1] > 0.02) & (dataeff[1] < 0.9) & (dataeff[0] > 5e1) & (dataeff[0] < 1e4) 
        #            ADUs = dataeff[0][mask]
        #    
        #            # convert ADUs to reference frame
        #            if epoch != self.reference:
        #                # find files to convert to correct magnitudes
        #                matchfiles = os.listdir("%s/%s/match" % (self.magdir, self.mode))
        #                afluxfinal = 0
        #                e_afluxfinal = 0
        #                for matchfile in matchfiles:  # loop over many CCDs
        #                    if re.match("match_%s_.*?_%02i-%02i.npy" % (self.mode, epoch, self.reference), matchfile):
        #                        datamatch = np.load("%s/%s/match/%s" % (self.magdir, self.mode, matchfile))
        #                        (aflux, e_aflux) = [datamatch[0], datamatch[1]]
        #                        afluxfinal += aflux / e_aflux
        #                        e_afluxfinal += 1. / e_aflux
        #                if afluxfinal != 0 and e_afluxfinal != 0:
        #                    afluxfinal = afluxfinal / e_afluxfinal
        #                    e_afluxfinal = 1. / np.sqrt(e_afluxfinal)
        #                #print "Average conversion factor between science and reference fluxes among different CCDs: %s +- %s" % (afluxfinal, e_afluxfinal)
        #    
        #            else:
        #                afluxfinal = 1
        #        
        #            # convert ADUs to reference frame ADUs and then use the zero points from there to convert to magnitudes
        #            mags = DT.ADU2mag_avg(ADUs / afluxfinal, dataref["EXP_TIME"], dataref["AIRMASS"], self.band)  
        #    
        #            # the median difference between the magnitudes derived using the zero points in the science image and the previously derived magnitudes (from the reference frame)
        #            # any differences would be due to (presumably) clouds
        #            deltamag = np.median(DT.ADU2mag_avg(ADUs, data["EXP_TIME"], data["AIRMASS"], self.band) - mags)
        #
        #            # interpolate efficiencies to estimate 50% completeness and solve for model parameters
        #            effinterp = interp1d(dataeff[1][mask], mags)
        #            offsettry = effinterp(0.5)
        #            sol = curve_fit(DT.efficiency_mag, mags, dataeff[1][mask], p0 = [offsettry, 0.5])
        #        
        #            if not hasattr(self, "OFFSET"):
        #                self.OFFSET = np.array([sol[0][0]])
        #                self.DELTA = np.array([sol[0][1]])
        #                self.deltamag_sciref = np.array([deltamag])
        #            else:
        #                self.OFFSET = np.hstack([self.OFFSET, sol[0][0]])
        #                self.DELTA = np.hstack([self.DELTA, sol[0][1]])
        #                self.deltamag_sciref = np.hstack([self.deltamag_sciref, deltamag])
        #                
        #            # plot efficiency relations
        #            if self.doplot:
        #                ax.plot(xs, DT.efficiency_mag(xs, sol[0][0], sol[0][1]), c = 'k')
        #                if self.mode[0:8] == "Blind14A":
        #                    ax.text(25.5, 0.9, "b)", fontsize = 30)
        #                if self.mode[0:8] == "Blind15A":
        #                    ax.text(25.5, 0.9, "c)", fontsize = 30)
        #                if self.mode[0:8] == "Blind13A":
        #                    ax.text(25.5, 0.9, "a)", fontsize = 30)
        #
        #    
        #    # numpify and sort
        #    self.MJDs = np.array(self.MJDs)
        #    self.airmasses = np.array(self.airmasses)
        #    self.FWHMs = np.array(self.FWHMs)
        #    self.skyADUs = np.array(self.skyADUs)
        #    self.exptime = np.array(self.exptime)
        #    idxMJDs = np.argsort(self.MJDs)
        #    self.MJDs = self.MJDs[idxMJDs]
        #    self.airmasses = self.airmasses[idxMJDs]
        #    self.FWHMs = self.FWHMs[idxMJDs]
        #    self.skyADUs = self.skyADUs[idxMJDs]
        #    self.exptime = self.exptime[idxMJDs]
        #    self.epochs = np.array(range(len(idxMJDs))) + 2
        #    if self.depthmodel == "obs" or self.depthmodel == "m50":
        #        self.deltamag_sciref = self.deltamag_sciref[idxMJDs]
        #        self.OFFSET = self.OFFSET[idxMJDs]
        #        self.DELTA = self.DELTA[idxMJDs]
        #
        #        # save plotted efficiencies
        #        if self.doplot:
        #            ax.axhline(1., c = 'gray')
        #            ax.set_ylabel("efficiency", fontsize = 20)
        #            ax.set_xlabel("%s mag" % self.band, fontsize = 20)
        #            ax.set_xlim(21, 26)
        #            ax.set_ylim(0, 1)
        #            plt.savefig("plots/obs/%s_%s_effs_ref.png" % (self.mode, self.band))
        #            plt.savefig("plots/obs/%s_%s_effs_ref.pdf" % (self.mode, self.band))
        #            
        #    # compute moon phases, limiting magnitudes
        #    skymodel = sky(band = self.band, MJDs = self.MJDs)
        #    self.moonphases, self.skymags = np.array(skymodel.skymags())
        #    if self.depthmodel[:3] == "sim":
        #        self.limmag = map(lambda x, y, z: self.ETC.findmag(band=self.band, SNRin=5., exptime=x, airmass=y, skymode="mag", skymag=z), self.exptime, self.airmasses, self.skymags)
        #    elif self.depthmodel[:3] == "ETC":
        #        self.limmag = map(lambda x, y, z, zz: self.ETC.findmag(band=self.band, SNRin=5., exptime=x, airmass=y, skymode="ADU-FWHM", skyADU=z, fwhm=zz), self.exptime, self.airmasses, self.skyADUs, self.FWHMs)
        #    elif self.depthmodel == "m50":
        #        self.limmag = self.OFFSET
        #    elif self.depthmodel == 'obs':
        #        self.limmag = map(lambda x, y: interp1d(xs, DT.efficiency_mag(xs, x, y), bounds_error = False, fill_value = (1, 0)), self.OFFSET, self.DELTA)
        #        self.interpmag = xs


        
if __name__ == '__main__':

    survey = sys.argv[1]
    
    #KMNTNet17B = obsplan(obsname = "KMTNet", band = 'B', mode = 'custom', nfields = 5, nepochspernight = 3, nightfraction = 0.045, nread = 1, ncontnights = 180, nnights = 180, startmoonphase = 3, maxmoonphase = 15, doplot = True)

    if survey == "KMTNet17B":
        #plan = obsplan(obsname = "KMTNet", mode = 'file-cols', inputfile = "KMTNet_17B_BVRI.dat", nfields = 200, nepochspernight = 1, nightfraction = 1., nread = 1, doplot = True, bandcolors = {'g': 'g', 'r': 'r', 'i': 'brown', 'z': 'k'})
        plan = obsplan(obsname = "KMTNet", mode = 'file-cols', inputfile = "KMTNet_17B_BVRI.dat", nfields = 76, nepochspernight = 1, nightfraction = 1., nread = 1, doplot = True, bandcolors = {'g': 'k', 'B': 'b', 'V': 'g', 'R': 'r', 'I': 'brown'})
        plan = obsplan(obsname = "KMTNet", mode = 'file-cols', inputfile = "KMTNet_17B_mod1.dat", nfields = 76, nepochspernight = 1, nightfraction = 1., nread = 1, doplot = True, bandcolors = {'g': 'k', 'B': 'b', 'V': 'g', 'R': 'r', 'I': 'brown'})
        plan = obsplan(obsname = "KMTNet", mode = 'file-cols', inputfile = "KMTNet_17B_mod1b.dat", nfields = 76, nepochspernight = 1, nightfraction = 1., nread = 1, doplot = True, bandcolors = {'g': 'k', 'B': 'b', 'V': 'g', 'R': 'r', 'I': 'brown'})
        plan = obsplan(obsname = "KMTNet", mode = 'file-cols', inputfile = "KMTNet_17B_mod2.dat", nfields = 76, nepochspernight = 1, nightfraction = 1., nread = 1, doplot = True, bandcolors = {'g': 'k', 'B': 'b', 'V': 'g', 'R': 'r', 'I': 'brown'})
    elif survey == "HiTS15B":
        obsplan(obsname = "CFHT-MegaCam", mode = 'file-cols', inputfile = "SNLS_bands.dat", nfields = 1, nepochspernight = 1, nightfraction = 0.045, nread = 5, doplot = True, bandcolors = {'g': 'g', 'r': 'r', 'i': 'brown', 'z': 'k'})
    
    #plan = obsplan(obsname = "KMTNet", band = 'g', mode = 'file', inputfile = "KMTNet17B.dat", nfields = 12, nepochspernight = 1, nightfraction = 0.5, nread = 3, doplot = True)

    #filtername = "g"
    #plan = obsplan(obsname = "CFHT-MegaCam", band = filtername, mode = 'file', inputfile = "SNLS_%s.dat" % filtername, nfields = 1, nepochspernight = 1, nightfraction = 0.045, nread = 5, doplot = True)

    #customplan = obsplan(obsname = "Blanco-DECam", band = 'g', mode = 'custom', nfields = 50, nepochspernight = 1, ncontnights = 30, nnights = 30, nightfraction = 0.5, nread = 2, startmoonphase = 0, maxmoonphase = 7, doplot = True)

