import re, os, sys
import numpy as np

from constants import *

def readSNdata(project, SNname):
    
    if project == "DES":
        dodes = True
        dehits = False
    elif project == "HiTS":
        dodes = False
        dohits = True
        
    #########################
    # Observational data
    #########################
    
    # DES SNe
    # -----------------------------

    if dodes:
        
        DESdir = "../DES"
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

    elif dohits:
        #SNname = "SNHiTS15A"
        #SNname = "SNHiTS15P"
        #SNname = "SNHiTS15D"
        #SNname = "SNHiTS15aw"
        #SNname = "SNHiTS15K"
        #SNname = "SNHiTS14B"
        #SNname = "SNHiTS15B"

        (MJDs, MJDrefs, ADUs, e_ADUs, mags, e1_mags, e2_mags, sn_filters) \
            = np.loadtxt("../HiTS/LCs/%s.txt" % SNname, usecols = (0, 1, 5, 6, 7, 8, 9, 10), dtype = str).transpose()

        sn_mjd = np.array(MJDs, dtype = float)
        sn_mjdref = np.array(MJDrefs, dtype = float)
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
        #texp0 = 57077.

        if SNname == "SNHiTS14B":
            sn_mjdref = np.hstack([sn_mjdref, sn_mjdref[-1], sn_mjdref[-1]])
            sn_mjd = np.hstack([sn_mjd, sn_mjd[0] + 9.01680793, sn_mjd[0] + 23.85])
            sn_flux = np.hstack([sn_flux, mag2flux(22.34), mag2flux(22.9)])
            sn_e_flux = np.hstack([sn_e_flux, sn_e_flux[-1], sn_e_flux[-1]])
            sn_filters = np.hstack([sn_filters, 'g', 'g'])
            
        mask = sn_filters == 'g'
        texp0 = sn_mjd[mask][np.argmax(np.diff(sn_flux[mask]))]

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


    else:
        print("Define observations...")
        sys.exit()


    return sn_mjd, sn_mjdref, sn_flux, sn_e_flux, sn_filters, fixz, zcmb, texp0

if __name__ == "__main__":

    project = sys.argv[1]
    SNname = sys.argv[2]
    
    print(readSNdata(project, SNname))
