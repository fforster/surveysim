import numpy as np
import sys, getopt
from observatory import *

# CC F. Forster, J. Martinez, J.C. Maureira, modified public DECam exposure time c

class ETC(object):

    # class constructor given seeing in r band in arcsec and ETC version as keyword arguments 'seeing_r_arcsec' and 'vETC'
    def __init__(self, **kwargs):

        # observatory
        self.observatory = kwargs['observatory']

        # seeing in r band
        self.seeing = {}
        if 'seeing_r_arcsec' in kwargs.keys():
            self.seeing['r'] = kwargs['seeing_r_arcsec'] # arcsec
        else:
            self.seeing['r'] = 0.75 # median value at CTIO

    def doseeing(self):
            
        # scaling between seeing at different filters
        self.seeing['g'] = self.seeing['r'] * (self.observatory.cwl['r'] / self.observatory.cwl['g'])**0.2
        self.seeing['u'] = 0.2 + self.seeing['g'] * (self.observatory.cwl['g'] / self.observatory.cwl['u'])**0.2
        self.seeing['i'] = self.seeing['r'] * (self.observatory.cwl['r'] / self.observatory.cwl['i'])**0.2
        self.seeing['z'] = self.seeing['i'] * (self.observatory.cwl['i'] / self.observatory.cwl['z'])**0.2
        self.seeing['Y'] = self.seeing['z'] * (self.observatory.cwl['z'] / self.observatory.cwl['Y'])**0.2

        # approximation, not sure how valid this is
        self.seeing['B'] = self.seeing['u']
        self.seeing['V'] = self.seeing['g']
        self.seeing['R'] = self.seeing['r']
        self.seeing['I'] = self.seeing['i']

    # compute FWHM given band and airmass as keyword arguments 'band' and 'airmass'
    def FWHM(self, **kwargs):

        # filter name
        band = kwargs['band']

        # compute seeing at all bands
        self.doseeing()
        
        # airmass effect
        fseeing_airmass = 1. / np.cos(np.arccos(1. / kwargs['airmass']))**(3./5.) # seeing correction factor due to airmass

        # add instrumental seeing
        FWHM_arcsec = np.sqrt((self.seeing[band] * fseeing_airmass)**2 + self.observatory.inst_seeing**2)
        
        return FWHM_arcsec

    # compute SNR given band, source magnitude, exposure time and airmass
    # different modes for sky/FWHM calculation
    # skymode = mag: sky given in mag per arcsec2 at zenith (skymag), FWHM derived from seeing in r band, band and airmass
    # skymode = ADU: use empirical sky in ADU (skyADU), FWHM derived from seeing in r band, band and airmass
    # skymode = mag-FWHM: sky given in mag per arcsec2 at zenith (skymag), use empirical FWHM (fwhm)
    # skymode = ADU-FWHM: use empirical sky in ADU (skyADU), use empirical FWHM (fwhm)
    def SNR(self, **kwargs):#, mag, exptime, airmass, skymode, skylevel):
    
        # non-optional arguments
        band = kwargs['band']
        mag = kwargs['mag']
        exptime = kwargs['exptime']
        airmass = kwargs['airmass']
        skymode = kwargs['skymode']
        if "nread" in kwargs.keys():
            nread = kwargs['nread']
        else:
            nread = 1
        if (skymode == 'mag' or skymode == 'mag-FWHM') and not 'skymag' in kwargs.keys():
            raise ValueError("Missing sky magnitude (skymag)")
        elif (skymode == 'ADU' or skymode == 'ADU-FWHM')  and not 'skyADU' in kwargs.keys():
            raise ValueError("Missing sky level in ADU (skyADU)")
        elif (skymode == 'ADU-FWHM' or skymode == 'mag-FWHM') and not 'fwhm' in kwargs.keys():
            raise ValueError("Missing FWHM (fwhm)")

        # optional arguments: skymag or skyADU, fwhm
        
        # speed of light
        cspeed = 2.99792458e17 # nm
            
        # filter width in Hz
        bandpass_Hz = cspeed / (self.observatory.cwl[band] - self.observatory.bandpass[band] / 2.) \
                      - cspeed / (self.observatory.cwl[band] + self.observatory.bandpass[band] / 2.)  # Hz
    
        # Planck's constant in MKS
        hPlanck_MKS = 6.62606957e-34 # m2 kg / s
        # hc / lambda in MKS (note cspeed and cwl_nm are in nm)
        hc_lambda = hPlanck_MKS * cspeed / self.observatory.cwl[band]
    
        # magnitude scale zero point flux
        zero_photrate = 3.631e-23 * bandpass_Hz / hc_lambda # photons / sec / m2  #  taken from ETC
    
        # photons from zero magnitude source
        zero_phot = zero_photrate * self.observatory.area * exptime  # photons
    
        # atmospheric transmission taking airmass into account
        atmosph_eff_airmass = self.observatory.atmosph_eff[band] * np.exp(1.) / np.exp(airmass)
            
        # derive FWHM from seeing and airmass or use empirical FWHM
        if skymode == 'mag' or skymode == 'ADU':
            fseeing_airmass = 1. / np.cos(np.arccos(1. / airmass))**(3./5.) # seeing correction factor due to airmass
            # airmass and optics effect on FWHM
            self.doseeing()
            FWHM_arcsec = np.sqrt((self.seeing[band] * fseeing_airmass)**2 + self.observatory.inst_seeing**2)
        elif skymode == 'mag-FWHM' or skymode == 'ADU-FWHM':
            # if FWHM is provided do not scale seeing
            FWHM_arcsec = kwargs['fwhm']
        else:
            raise ValueError("SNR: skymode %s not recognized" % skymode)
    
        # aperture
        aperture = np.pi * ((self.observatory.aperture_eff * FWHM_arcsec) / 2.)**2  # aperture in arcsec2
    
        # final throughput
        throughput = self.observatory.CCD_eff[band] * self.observatory.filter_eff[band] * \
                     self.observatory.corrector_eff[band] * self.observatory.primary_eff[band] * \
                     self.observatory.vig * atmosph_eff_airmass

        # signal from zero magnitude source
        zero_signal = zero_phot * throughput # electrons from a zero mag source

        # electrons from source
        source_electrons = zero_signal * 10**(mag / -2.5) # electrons from a source of the given magnitude
        
        # electrons from the sky: 1) sky_ADU is given (use gain, aperture, pixel scale), 2) sky_mag is given (use zero_signal, aperture and airmass)
        if skymode == 'ADU' or skymode == 'ADU-FWHM':
            sky_electrons = kwargs['skyADU'] * self.observatory.gain * aperture / self.observatory.pixscale**2 # electrons from the sky per aperture given the empirical sky per pixel in ADU
        elif skymode == 'mag' or skymode == 'mag-FWHM':
            sky_electrons = 10**(-kwargs['skymag'] / 2.5) * zero_signal * aperture * airmass # electrons from the sky per aperture
        else:
            raise ValueError("SNR: skymode %s not recognized" % skymode)
    
        # readout noise per aperture
        RON_aper = self.observatory.RON**2 * (aperture / self.observatory.pixscale**2) # electrons^2/pixel^2
            
        # surce signal to noise ratio
        SNRout = source_electrons / np.sqrt(source_electrons + sky_electrons + nread * RON_aper)
    
        return SNRout
    
    # find magnitude given target SNR (very stupid search for now)
    def findmag(self, **kwargs):
        
        magsarray = np.linspace(15., 27., 100000)
        if kwargs['skymode'] == 'mag':
            SNRs = self.SNR(band=kwargs['band'], mag=magsarray, exptime=kwargs['exptime'], nread=kwargs['nread'], airmass=kwargs['airmass'], skymode=kwargs['skymode'], skymag=kwargs['skymag'])
        elif kwargs['skymode'] == 'mag-FWHM':
            SNRs = self.SNR(band=kwargs['band'], mag=magsarray, exptime=kwargs['exptime'], nread=kwargs['nread'], airmass=kwargs['airmass'], skymode=kwargs['skymode'], skymag=kwargs['skymag'], fwhm=kwargs['fwhm'])
        elif kwargs['skymode'] == 'ADU':
            SNRs = self.SNR(band=kwargs['band'], mag=magsarray, exptime=kwargs['exptime'], nread=kwargs['nread'], airmass=kwargs['airmass'], skymode=kwargs['skymode'], skyADU=kwargs['skyADU'])
        elif kwargs['skymode'] == 'ADU-FWHM':
            SNRs = self.SNR(band=kwargs['band'], mag=magsarray, exptime=kwargs['exptime'], nread=kwargs['nread'], airmass=kwargs['airmass'], skymode=kwargs['skymode'], skyADU=kwargs['skyADU'], fwhm=kwargs['fwhm'])
        else:
            raise ValueError("findmag: wrong keyword arguments")

        if 'SNRin' not in kwargs.keys():
            raise ValueError("findmag: missing input SNR")
        else:
            return magsarray[np.argmin((SNRs - kwargs['SNRin'])**2)]
    
if __name__ == "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hts:e:o:b:t:n:m:a:f:M:A:S", ["help", "test", "seeing=", "ETC=", "skymode=", "band=", "exptime=", "nread=", "mag=", "airmass=", "fwhm=", "skymag=", "skyADU=", "SNR="])
    except getopt.GetoptError:
        print 'python ETC_DECam.py --help'
        sys.exit(1)
    for opt, arg in opts:

        if opt in ('-h', '--help'):
            print "\nCC F. Forster, J. Martinez, J.C. Maureira, modified DECam ETC.\n---------------------------------------------------\n"
            print "Please report any problems to francisco.forster@gmail.com"
            print "Options: help, test, seeing= [arcsec], ETC=, skymode=, band=, exptime= [sec], mag=, airmass=, fwhm= [arcsec], skymag= [mag/arcsec2], skyADU= [ADU/pix], SNR=\n"
            print "Code contains special class to call the ETC from inside python code, but can also be called from the command line."
            
            print "\nExample inside python code:\n"
            
            print """   # import module
   from ETC import *
   from observatory import *

   # initialize observatory
   obs = observatory(observatory = "Blanco-DECam")
            
   # initilize ETC
   ETCobs = ETC(observatory = obs, seeing_r_arcsec = 0.75)

   # Testing FWHM for an airmass vector...
   print "   u band, airmass between 1.0 and 1.6", ETCobs.FWHM(band='u', airmass=np.linspace(1.0, 1.6, 10))
   print "   g band, airmass between 1.2 and 1.9", ETCobs.FWHM(band='g', airmass=np.linspace(1.2, 1.9, 10))

   # Testing SNR and findmag with all skymodes with a 20 mag source, 173 sec exposure time in g band, airmass of 1.0...
   print ETCobs.SNR(band='g', mag=20, exptime=173, nread=1, airmass=1.0, skymode='mag', skymag=22)
   print ETCobs.findmag(band='g', SNRin=SNRtest, exptime=173, nread=1, airmass=1.0, skymode='mag', skymag=22.0)\n\n"""

            print "Command line has two basic modes: giving an input magnitude to get a signal to noise ratio (SNR) and viceversa, i.e. getting a limiting magnitude given a SNR.\n"
            print "Then, the skymode variabla controls how the environmental variables are defined:"
            print "mag: sky is given in mag/arcsec2 and the FWHM is derived from the airmass and seeing at zenith in r band (default is 0.75\" at zenith)"
            print "ADU: sky is given in ADU/pixel and the FWHM is derived from the airmass and seeing at zenith in r band (default is 0.75\" at zenith)"
            print "mag-FWHM: sky is given in mag/arcsec2 and FWHM is manually input in arcsec (airmass is also needed to compute extra extinction and atmosphere emission)"
            print "ADU-FWHM: sky is given in ADU/pixeland FWHM is manually input in arcsec (airmass is also needed to compute extra extinction and atmosphere emission)\n"
            
            print "Command line examples:\n"
            print """   python ETC.py --skymode mag --mag 20 --band g --exptime 173 --nread 1 --airmass 1.0 --skymag 22 
   python ETC.py --skymode ADU --mag 20 --band g --exptime 173 --nread 1 --airmass 1.0 --skyADU 200 
   python ETC.py --skymode mag-FWHM --mag 20 --band g --exptime 173 --nread 1 --airmass 1.0 --skymag 22 --fwhm 2
   python ETC.py --skymode ADU-FWHM --mag 20 --band g --exptime 173 --nread 1 --airmass 1.0 --skyADU 200 --fwhm 2
   
   python ETC.py --skymode mag --SNR 5 --band g --exptime 173 --nread 1 --airmass 1.0 --skymag 22 
   python ETC.py --skymode ADU --SNR 5 --band g --exptime 173 --nread 1 --airmass 1.0 --skyADU 200 
   python ETC.py --skymode mag-FWHM --SNR 5 --band g --exptime 173 --nread 1 --airmass 1.0 --skymag 22 --fwhm 2
   python ETC.py --skymode ADU-FWHM --SNR 5 --band g --exptime 173 --nread 1 --airmass 1.0 --skyADU 200 --fwhm 2"""
            print "\n\n\n"
            

            sys.exit(1)

        elif opt in ('-t', '--test'):

            print "\nInitializing ETC for Blanco-DECam with seeing_r=of 0.75\"..."
            
            # initialize observatory
            obs = observatory(observatory = "Blanco-DECam")
            
            # initilize ETC
            ETCobs = ETC(observatory = obs, seeing_r_arcsec = 0.75)
            
            print "\nTesting FWHM for an airmass vector..."
            print "   u band, airmass between 1.0 and 1.6", ETCobs.FWHM(band='u', airmass=np.linspace(1.0, 1.6, 10))
            print "   g band, airmass between 1.2 and 1.9", ETCobs.FWHM(band='g', airmass=np.linspace(1.2, 1.9, 10))
            print "   r band, airmass between 1.2 and 1.9", ETCobs.FWHM(band='r', airmass=np.linspace(1.2, 1.9, 10))
            print "   i band, airmass between 1.2 and 1.9", ETCobs.FWHM(band='i', airmass=np.linspace(1.2, 1.9, 10))
            print "   z band, airmass between 1.2 and 1.9", ETCobs.FWHM(band='z', airmass=np.linspace(1.2, 1.9, 10))
            print "   Y band, airmass between 1.2 and 1.9", ETCobs.FWHM(band='Y', airmass=np.linspace(1.2, 1.9, 10))
            
            print "\nTesting SNR and findmag with all skymodes with a 20 mag source, 173 sec exposure time in g band, airmass of 1.0..."
            SNRtest = ETCobs.SNR(band='g', mag=20, exptime=173, nread=1, airmass=1.0, skymode='mag', skymag=22)
            magtest = ETCobs.findmag(band='g', SNRin=SNRtest, exptime=173, nread=1, airmass=1.0, skymode='mag', skymag=22.0)
            print "   mag (skymag=22): SNR %f <-> mag %f" % (SNRtest, magtest)
            if np.abs(20. - magtest) > 1e-4:
                print "   mag not OK"
            else:
                print "   OK"
            SNRtest = ETCobs.SNR(band='g', mag=20, exptime=173, nread=1, airmass=1.0, skymode='mag-FWHM', skymag=22, fwhm=1.0)
            magtest = ETCobs.findmag(band='g', SNRin=SNRtest, exptime=173, nread=1, airmass=1.0, skymode='mag-FWHM', skymag=22, fwhm=1.0)
            print "   mag-FWHM (skymag=22, fwhm=1): SNR %f <-> mag %f"  % (SNRtest, magtest)
            if np.abs(20. - magtest) > 1e-4:
                print "   mag-FWHM not OK"
            else:
                print "   OK"
            SNRtest = ETCobs.SNR(band='g', mag=20, exptime=173, nread=1, airmass=1.0, skymode='ADU', skyADU=120)
            magtest = ETCobs.findmag(band='g', SNRin=SNRtest, exptime=173, nread=1, airmass=1.0, skymode='ADU', skyADU=120)
            print "   ADU (skyADU=120): SNR %f <-> mag %f" % (SNRtest, magtest)
            if np.abs(20. - magtest) > 1e-4:
                print "   ADU not OK"
            else:
                print "   OK"
            SNRtest = ETCobs.SNR(band='g', mag=20, exptime=173, nread=1, airmass=1.0, skymode='ADU-FWHM', skyADU=120, fwhm=1.0)
            magtest = ETCobs.findmag(band='g', SNRin=SNRtest, exptime=173, nread=1, airmass=1.0, skymode='ADU-FWHM', skyADU=120, fwhm=1.0)
            print "   ADU-FHWM (skyADU=120, fwhm=1): SNR %f <-> mag %f" % (SNRtest, magtest)
            if np.abs(20. - magtest) > 1e-4:
                print "   ADU-FWHM not OK"
            else:
                print "   OK"

            print "\nTesting findmag for input SNR of 5 and exposure time of 173 sec, assuming sky magnitudes of 22.8, 22.1, 21.1, 20.1, 18.7 and 18 in ugrizY..."
            print "   u band, mags at airmasses of 1.0, 1.2, 1.4, 1.6, 1.8, 2.0", map(lambda x: ETCobs.findmag(band='u', SNRin=5, exptime=173, nread=1, airmass=x, skymode='mag', skymag=22.8), [1., 1.2, 1.4, 1.6, 1.8, 2.0])
            print "   g band, mags at airmasses of 1.0, 1.2, 1.4, 1.6, 1.8, 2.0", map(lambda x: ETCobs.findmag(band='g', SNRin=5, exptime=173, nread=1, airmass=x, skymode='mag', skymag=22.1), [1., 1.2, 1.4, 1.6, 1.8, 2.0])
            print "   r band, mags at airmasses of 1.0, 1.2, 1.4, 1.6, 1.8, 2.0", map(lambda x: ETCobs.findmag(band='r', SNRin=5, exptime=173, nread=1, airmass=x, skymode='mag', skymag=21.1), [1., 1.2, 1.4, 1.6, 1.8, 2.0])
            print "   i band, mags at airmasses of 1.0, 1.2, 1.4, 1.6, 1.8, 2.0", map(lambda x: ETCobs.findmag(band='i', SNRin=5, exptime=173, nread=1, airmass=x, skymode='mag', skymag=20.1), [1., 1.2, 1.4, 1.6, 1.8, 2.0])
            print "   z band, mags at airmasses of 1.0, 1.2, 1.4, 1.6, 1.8, 2.0", map(lambda x: ETCobs.findmag(band='z', SNRin=5, exptime=173, nread=1, airmass=x, skymode='mag', skymag=18.7), [1., 1.2, 1.4, 1.6, 1.8, 2.0])
            print "   Y band, mags at airmasses of 1.0, 1.2, 1.4, 1.6, 1.8, 2.0", map(lambda x: ETCobs.findmag(band='Y', SNRin=5, exptime=173, nread=1, airmass=x, skymode='mag', skymag=18), [1., 1.2, 1.4, 1.6, 1.8, 2.0])

            print "\n\n\n\n"
            sys.exit()
            

        elif opt in ('-s', '--seeing'):
            seeing_r_arcsec = float(arg)
        elif opt in ('-e', '--ETC'):
            vETC = int(arg)
        elif opt in ('-o', '--skymode'):
            skymode = arg
        elif opt in ('-b', '--band'):
            band = arg
        elif opt in ('-t', '--exptime'):
            exptime = float(arg)
        elif opt in ('-n', '--nread'):
            nread = float(arg)
        elif opt in ('-m', '--mag'):
            mag = float(arg)
        elif opt in ('a', '--airmass'):
            airmass = float(arg)
        elif opt in ('-f', '--fwhm'):
            fwhm = float(arg)
        elif opt in ('-M', '--skymag'):
            skymag = float(arg)
        elif opt in ('-A', '--skyADU'):
            skyADU = float(arg)
        elif opt in ('-S', '--SNR'):
            SNR = float(arg)

    if 'seeing_r_arcsec' in locals():
        ETCobs = ETC(observatory = observatory(observatory = "Blanco-DECam"), seeing_r_arcsec=seeing_r_arcsec)
    else:
        ETCobs = ETC(observatory = observatory(observatory = "Blanco-DECam"))

    # manage exceptions, missing lots of them!
    if 'skymode' in locals():
        
        if 'SNR' in locals():

            if skymode == 'mag':
                print "Magnitude(SNR=%f): %f" % (SNR, ETCobs.findmag(band=band, SNRin=SNR, exptime=exptime, nread=nread, airmass=airmass, skymode='mag', skymag=skymag))
            elif skymode == 'mag-FWHM':
                print "Magnitude(SNR=%f): %f" % (SNR, ETCobs.findmag(band=band, SNRin=SNR, exptime=exptime, nread=nread, airmass=airmass, skymode='mag-FWHM', skymag=skymag, fwhm=fwhm))
            elif skymode == 'ADU':
                print "Magnitude(SNR=%f): %f" % (SNR, ETCobs.findmag(band=band, SNRin=SNR, exptime=exptime, nread=nread, airmass=airmass, skymode='ADU', skyADU=skyADU))
            elif skymode == 'ADU-FWHM':
                print "Magnitude(SNR=%f): %f" % (SNR, ETCobs.findmag(band=band, SNRin=SNR, exptime=exptime, nread=nread, airmass=airmass, skymode='ADU-FWHM', skyADU=skyADU, fwhm=fwhm))
        
        else:
            
            if skymode == 'mag':
                print "SNR(mag=%f): %f" % (mag, ETCobs.SNR(band=band, mag=mag, exptime=exptime, nread=nread, airmass=airmass, skymode='mag', skymag=skymag))
            elif skymode == 'mag-FWHM':
                print "SNR(mag=%f): %f" % (mag, ETCobs.SNR(band=band, mag=mag, exptime=exptime, nread=nread, airmass=airmass, skymode='mag-FWHM', skymag=skymag, fwhm=fwhm))
            elif skymode == 'ADU':
                print "SNR(mag=%f): %f" % (mag, ETCobs.SNR(band=band, mag=mag, exptime=exptime, nread=nread, airmass=airmass, skymode='ADU', skyADU=skyADU))
            elif skymode == 'ADU-FWHM':
                print "SNR(mag=%f): %f" % (mag, ETCobs.SNR(band=band, mag=mag, exptime=exptime, nread=nread, airmass=airmass, skymode='ADU-FWHM', skyADU=skyADU, fwhm=fwhm))

    else:
        
        print "Define skymode (mag, mag-FWHM, ADU, ADU-FWHM)"
        sys.exit()
        
    

