import re, os, sys
import numpy as np
import pandas as pd
from collections import defaultdict

from constants import *

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.io import ascii


def readSNdata(project, SNname, maxairmass = 1.7):
            
    #########################
    # Observational data
    #########################

    # make sure to return numpy arrays and not pandas data frames
    #############################################################
    
    # DES SNe
    # -----------------------------

    if project == 'DES':
        
        DESdir = "%s/../LCs/DES" % (os.environ["SURVEYSIM_PATH_INPUT"])
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

    elif project == "HiTS":
        #SNname = "SNHiTS15A"
        #SNname = "SNHiTS15P"
        #SNname = "SNHiTS15D"
        #SNname = "SNHiTS15aw"
        #SNname = "SNHiTS15K"
        #SNname = "SNHiTS14B"
        #SNname = "SNHiTS15B"

        #(MJDs, MJDrefs, airmass, ADUs, e_ADUs, mags, sn_filters) \
        #    = .loadtxt("../HiTS/LCs/%s.txt" % SNname, usecols = (0, 1, 2, 5, 6, 7, 10), dtype = str).transpose()
        df = pd.read_table("%s/../LCs/HiTS/LCs/%s.dat" % (os.environ["SURVEYSIM_PATH_INPUT"], SNname), sep = "\s+", comment = "#")
        
        sn_mjd = np.array(df["MJD"])
        sn_mjdref = np.array(df["MJDref"])
        airmass = np.array(df["airmasssci"])
        sn_adu = np.array(df["ADU"])
        sn_e_adu = np.array(df["e_ADU"])
        sn_mag = np.array(df["mag"])
        sn_filters = np.array(df["band"], dtype = str)
        
        sn_flux = np.array(sn_adu)
        sn_e_flux = np.array(sn_e_adu)

        mask = (airmass <= maxairmass)

        sn_mjd = sn_mjd[mask]
        sn_mjdref = sn_mjdref[mask]
        sn_adu = sn_adu[mask]
        sn_e_adu = sn_e_adu[mask]
        sn_mag = sn_mag[mask]
        sn_flux = sn_flux[mask]
        sn_e_flux = sn_e_flux[mask]
        sn_filters = np.array(sn_filters)[mask]
        
        maskg = (sn_filters == 'g')
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
        elif SNname == "SNHiTS14Q":
            sn_mjdref = np.hstack([sn_mjdref, sn_mjdref[-1]])
            sn_mjd = np.hstack([sn_mjd, sn_mjdref[-1] + 23.996])
            sn_flux = np.hstack([sn_flux, mag2flux(23.218)])
            sn_e_flux = np.hstack([sn_e_flux, mag2flux(23.218 - 0.12) - mag2flux(23.218)])
            sn_filters = np.hstack([sn_filters, 'g'])

            
        mask = sn_filters == 'g'
        texp0 = sn_mjd[mask][np.argmax(np.diff(sn_flux[mask]))]
        texp0 = sn_mjd[mask][0] + 3.#np.argmax(np.diff(sn_flux[mask]))]

        if SNname == "SNHiTS14A":
            zcmb = 0.2175
            fixz = True
        elif SNname == "SNHiTS14Y":
            zcmb = 0.108
            fixz = True
        elif SNname == "SNHiTS14C":
            zcmb = 0.084
            fixz = True
        elif SNname == "SNHiTS14D":
            zcmb = 0.135
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
        elif SNname == "SNHiTS15ad":
            zcmb = 0.055392
            fixz = True
        elif SNname == "SNHiTS15al":
            zcmb = 0.2
            fixz = True
        elif SNname == "SNHiTS15aw":
            zcmb = 0.0663
            fixz = True
        elif SNname == "SNHiTS15aq":
            zcmb = 0.11
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

    elif project == "PTF":
        
        df = pd.read_table("%s/../LCs/PTF/LCs/%s.dat" % (os.environ["SURVEYSIM_PATH_INPUT"], SNname), sep = "\s*\t\s*", comment = "#", engine = "python")
        JD0 = float(re.findall('.*?=(\d+.\d+).*?', df.columns[0])[0])
        MJDref = JD0 - 2400000.5 # in this case it is the explosion date minus 50 days
        df = df.rename(columns = {df.columns[0]: "MJD"})
        df["MJD"] = df["MJD"] + MJDref
        #df = df[(df['family'] == 'PTF') | ((df["family"] == "SDSS") & (df["band"] == 'g'))]
        df = df[(df["family"] == "SDSS") & ((df["band"] == 'g') | (df["band"] == 'r'))]

        sn_mjd = np.array(df["MJD"])
        sn_mjdref = np.array(MJDref * np.ones_like(sn_mjd)) - 1000. # assume reference was taken 1000 days into the past
        sn_mag = np.array(df["mag"])
        sn_e_mag = np.array(df["mag-err"])
        sn_filters = np.array(df["band"], dtype = str)
        
        sn_flux = mag2flux(sn_mag)
        sn_e_flux = mag2flux(sn_mag) - mag2flux(sn_mag + sn_e_mag)

        if SNname == "SN2013fs":
            zcmb = 0.011855
            fixz = True
            texp0 = MJDref

    elif project == "Kepler":

        filename = "%s/../LCs/Kepler/%s.txt" % (os.environ["SURVEYSIM_PATH_INPUT"], SNname)
        
        # find reference magnitude
        data = open(filename, 'r')
        line1 = data.readline()
        refmag = float(re.findall("flux.*(\d\d.\d\d)", line1)[0])
        refflux = mag2flux(refmag)
        print("Reference flux: %f" % refflux)
        
        df = pd.read_table(filename, sep = "\s+", comment = "#")

        sn_mjd = np.array(df["KJD"]) + 54832.5 # see Zheng 2016
        sn_mjdref = sn_mjd[0] - 1000
        texp0 = sn_mjd[0]
        sn_flux = np.array(df.flux) * refflux  # normalization to make SN appear close and make redshift correction negligible
        sn_e_flux = np.ones_like(sn_flux) * np.std(sn_flux[sn_mjd < sn_mjd[0] + 20])
        #sn_e_flux = np.array(df["LC_err"]) * 1e-25  # normalization to make SN appear close and make redshift correction negligible
        sn_filters = np.array(list(map(lambda x: "Kepler", sn_flux)), dtype = str)
        fixz = True
        if SNname[:6] == 'ksn11a':
            zcmb = 0.051
        elif SNname[:6] == 'ksn11d':
            zcmb = 0.087


    elif project == "ROTSEIII":

        df = pd.read_table("%s/../LCs/ROTSEIII/%s.txt" % (os.environ["SURVEYSIM_PATH_INPUT"], SNname), sep = "\s+", comment = "#")

        from datetime import datetime
        from astropy.time import Time
        import calendar
        
        year = df.year
        month = df.month
        daysf = df.day
        days = np.array(daysf, dtype = int)
        hours = np.array(24. * (daysf - days), dtype = int)
        minutes = np.array(60. * (24. * (daysf - days) - hours), dtype = int)
        seconds = np.array(60. * (60. * (24. * (daysf - days) - hours) - minutes), dtype = int)

        # dictionary to convert month to number
        month2n = {}
        for idx in range(1, 13):
            month2n[calendar.month_abbr[idx]] = idx
        sn_mjd = np.array(list(map(lambda yy, mm, dd, hh, mmm, ss: Time(datetime(yy, month2n[mm], dd, hh, mmm, ss), scale = "utc").mjd, year, month, days, hours, minutes, seconds)))
        sn_mjdref = sn_mjd[0] - 1000.
        texp0 = np.array(max(sn_mjd[df.date < 0])) # maximum MJD which has an explosion date less than zero
        sn_maglim = df.maglim
        sn_flux = mag2flux(df.mag)
        mask = np.isnan(sn_flux)
        sn_e_flux = mag2flux(df.mag - df.e_mag) - mag2flux(df.mag)  # normalization to make SN appear close and make redshift correction negligible
        # take into account non-detections
        sn_flux[mask] = 0
        sn_e_flux[mask] = mag2flux(sn_maglim[mask])
        sn_filters = np.array(list(map(lambda x: "ROTSEIII", sn_flux)), dtype = str)
        fixz = False

        if SNname == "SN2006bp":
            zcmb = 0.003510
            #fixz = True
        
    elif project == "PanSTARRS1":

        df = pd.read_table("%s/../LCs/PanSTARRS1/%s.txt" % (os.environ["SURVEYSIM_PATH_INPUT"], SNname), sep = "\s+", comment = "#")

        mask = (df.band == 'u') | (df.band == 'g') | (df.band == 'r') | (df.band == 'i') | (df.band == 'z')
        df = df[mask]
        sn_mjd = np.array(df.MJD)
        sn_mjdref = sn_mjd[0] - 1000.
        texp0 = np.array(min(sn_mjd[np.array(df.phase) > 0])) # maximum MJD which has an explosion date less than zero
        mask = np.isnan(df.e_mag)
        nmask = np.invert(mask)
        sn_mag = np.array(df.mag, dtype = float)
        sn_e_mag = np.array(df.e_mag, dtype = float)
        sn_flux = mag2flux(sn_mag)
        sn_e_flux = np.zeros_like(sn_flux)
        sn_e_flux[nmask] = mag2flux(sn_mag[nmask] - sn_e_mag[nmask]) - mag2flux(sn_mag[nmask])  # normalization to make SN appear close and make redshift correction negligible
        sn_e_flux[mask] = sn_flux[mask]
        sn_flux[mask] = 0
        sn_filters = df.band
        fixz = True

        if SNname == "PS1-13arp":
            zcmb = 0.1665


    elif project == "Swift":

        df = pd.read_table("%s/../LCs/Swift/%s.dat" % (os.environ["SURVEYSIM_PATH_INPUT"], SNname), sep = "\s+", comment = "#")

        # correct to AB magnitudes, see https://swift.gsfc.nasa.gov/analysis/uvot_digest/zeropts.html
        Vega2AB = {"V": -0.01, "B": -0.13, "U": 1.02, "UVW1": 1.51, "UVM2": 1.69, "UVW2": 0.8}
        for band in Vega2AB.keys():
            df.Mag[df.Filter == band] += Vega2AB[band]


        sn_mjd = np.array(df["MJD[days]"])
        sn_mjdref = sn_mjd[0] - 1000
        texp0 = sn_mjd[0]
        sn_flux = np.array(mag2flux(df.Mag))
        sn_e_flux = np.array(np.abs(mag2flux(df.Mag - df.MagErr) - mag2flux(df.Mag + df.MagErr)))
        sn_filters = np.array(df.Filter, dtype = str)
        
        mask = np.isfinite(sn_flux)
        sn_mjd = sn_mjd[mask]
        sn_flux = sn_flux[mask]
        sn_e_flux = sn_e_flux[mask]
        sn_filters = sn_filters[mask]            

        if SNname == "ASASSN-14jb":
            df = pd.read_table("%s/../LCs/Swift/%s_ASASSN.dat" % (os.environ["SURVEYSIM_PATH_INPUT"], SNname), sep = "\s+", comment = "#")
            MJDextra = np.array(df.HJD - 2400000.5)
            fluxextra = np.array(df["flux(mJy)"] * 1e-26) # 1 Jy = 1e-23 erg/s/cm2/Hz
            e_fluxextra = np.array(df["flux_err"] * 1e-26)
            filtersextra = np.array(list(map(lambda x: 'V', MJDextra)), dtype = str)

            sn_mjd = np.hstack([sn_mjd, MJDextra])
            sn_flux = np.hstack([sn_flux, fluxextra])
            sn_e_flux = np.hstack([sn_e_flux, e_fluxextra])
            sn_filters = np.hstack([sn_filters, filtersextra])

            idx = np.argsort(sn_mjd)
            sn_mjd = sn_mjd[idx]
            sn_flux = sn_flux[idx]
            sn_e_flux = sn_e_flux[idx]
            sn_filters = sn_filters[idx]

        fixz = True

        if SNname == "ASASSN-14jb":
            zcmb = 0.006031

    elif project == "ZTF":
        #needed to estimate airmass

        #### data = ascii.read("%s/../LCs/ZTF/LCs/%s.txt" % (os.environ["SURVEYSIM_PATH"], SNname), format="basic")  
        df = pd.read_csv("%s/../LCs/ZTF/LCs/%s.csv" % (os.environ["SURVEYSIM_PATH"], "sample_corrected"))
        #rename colnames to delete ","
        
        #Need to change this *************************************
        #df = df.replace("ZTF_r", "r")
        #df = df.replace("ZTF_g", "g")
        ##df.drop(df[df["forcediffimflux"] == "null"].index, inplace=True)
        ##df.drop(df[df["filter"] == "ZTF_i"].index, inplace=True)
        #mask to select SNe
        mask = (df.SN_Name == SNname) | (df.SN_Name == SNname[2:]) | (df.oid == SNname)
        df = df[mask] #select data from the SNe we want
        df = df.reset_index(drop=True)
        #to return 
        sn_mjd = np.array(df.mjd)
        sn_mjdref = np.array(df["refjdstart"]) - 2400000.5 
        sn_flux = np.array(df["corrflux_uJy"], dtype = float) * (10**-29)  #uJy to ergs
        sn_e_flux = np.array(df["sigma_flux_diff_uJy"], dtype = float) * (10**-29) #1-sigma
        sn_filters = np.array(df["filt"], dtype = str)
    
        #find index of first detection
        snt = 5 #threshold
        args_det = np.argwhere( (sn_flux/ sn_e_flux) > snt )
        first_det = args_det[0][0]

        texp0 = sn_mjd[first_det] 
        fixz = False
        zcmb = df["redshift"][0]
    
    elif project == "ATLAS":
        df = pd.read_csv("%s/../LCs/ATLAS/LCs/ATLAS_%s.csv" % (os.environ["SURVEYSIM_PATH"], SNname))
        sn_mjd = np.array(df.MJD, dtype = float)
        sn_mjdref = np.zeros(sn_mjd.shape[0])
        sn_mjdref = sn_mjdref + 57230
        sn_filters = np.array(df.F, dtype = str)
        sn_flux = np.array(df.uJy, dtype = float) * (10**-29)
        sn_e_flux = np.array(df.duJy, dtype = float) * (10**-29)
        fixz = False
        zcmb = 0.05
        
        #find index of first detection
        snt = 5 #threshold
        args_det = np.argwhere( (sn_flux/ sn_e_flux) > snt )
        first_det = args_det[0][0]

        texp0 = sn_mjd[first_det] - 5

    elif project == "ZTF+ATLAS":
        #ZTF
        df = pd.read_csv("%s/../LCs/ZTF/LCs/%s.csv" % (os.environ["SURVEYSIM_PATH"], "sample_corrected"))
        mask = (df.SN_Name == SNname) | (df.SN_Name == SNname[2:]) | (df.oid == SNname)
        df = df[mask] #select data from the SNe we want                                                                                                                                                  
        df = df.reset_index(drop=True)

        ztf_mjd = np.array(df.mjd)
        ztf_mjdref = np.array(df["refjdstart"]) - 2400000.5
        ztf_flux = np.array(df["corrflux_uJy"], dtype = float) * (10**-29)  #uJy to ergs                                                                                                                 
        ztf_e_flux = np.array(df["sigma_flux_diff_uJy"], dtype = float) * (10**-29) #1-sigma                                                                                                             
        ztf_filters = np.array(df["filt"], dtype = str)

        zcmb = df["redshift"][0]

        #ATLAS
        df = pd.read_csv("%s/../LCs/ATLAS/LCs/ATLASclean2.0.csv" % os.environ["SURVEYSIM_PATH"])
        mask =  (df.oid == SNname)
        df = df[mask] #select data from the SNe we want                                                                                                                                                  
        df = df.reset_index(drop=True)
        atlas_mjd = np.array(df.MJD, dtype = float)
        atlas_mjdref = np.zeros(atlas_mjd.shape[0])
        atlas_mjdref = atlas_mjdref + 57230
        atlas_filters = np.array(df.F, dtype = str)
        atlas_flux = np.array(df.uJy, dtype = float) * (10**-29)
        atlas_e_flux = np.array(df.duJy, dtype = float) * (10**-29)
        #Join ZTF+ATLAS
        sn_mjd = np.append(ztf_mjd, atlas_mjd)
        sn_mjdref = np.append(ztf_mjdref, atlas_mjdref)
        sn_flux = np.append(ztf_flux, atlas_flux)                                                                                                                   
        sn_e_flux = np.append(ztf_e_flux, atlas_e_flux)                                                                                                              
        sn_filters = np.append(ztf_filters, atlas_filters)

        #find index of first detection                                                                                                                                                                    
        snt = 5 #threshold                                                                                                                                                                               
        args_det = np.argwhere( (sn_flux/ sn_e_flux) > snt )
        first_det = args_det[0][0]
        texp0 = sn_mjd[first_det]
        fixz = False

    else:
        print("Define observations...")
        sys.exit()


    return sn_mjd, sn_mjdref, sn_flux, sn_e_flux, sn_filters, fixz, zcmb, texp0

    
if __name__ == "__main__":

    project = sys.argv[1]
    SNname = sys.argv[2]
    
    print(readSNdata(project, SNname))
