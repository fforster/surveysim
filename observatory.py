# CC by F. Forster
# telescope class

import os

try:
    import ephem
    ephemimport = True
except:
    print "WARNING: ephem cannot be imported"
    ephemimport = False

class observatory(object):

    def __init__(self, **kwargs):

        if "observatory" in kwargs.keys():
            
            # default filter central wavelengths and bandwidth
            self.cwl = {'u': 375, 'g': 473.5, 'r': 638.5, 'i': 775.5, 'z': 922.5, 'Y': 995} # nm
            self.bandpass = {'u': 50, 'g': 147, 'r': 141, 'i': 147, 'z': 147, 'Y': 50} # nm

            # default instrumental seeing
            self.inst_seeing = 0.63 # arcsec

            # atmosphere transmission
            self.atmosph_eff = {'u': 0.7, 'g': 0.8,'r': 0.9, 'i': 0.9, 'z': 0.9, 'Y': 0.9}

            # effective aperture radius for photometric measurement
            self.aperture_eff = 2.04 

            if kwargs["observatory"] == "Blanco-DECam":

                self.area = 9.7 # m2
                self.FoV = 3.0 # deg2
                self.pixscale = 0.264 # "
                self.RON = 7. # e-
                self.gain = 4 # e-/ADU
                self.filterchange = 8 # s
                self.readouttime = 20.6 # s
                self.slewtime = 30 # s # typical slew time
                self.vig = 1.0 # vignetting
                
                # mirror, lens, filter and CCD efficiencies at filters
                self.primary_eff = {'u': 0.89, 'g': 0.89, 'r': 0.88, 'i': 0.87, 'z': 0.88, 'Y': 0.9}
                self.corrector_eff = {'u': 0.75, 'g': 0.86, 'r': 0.86, 'i': 0.86, 'z': 0.86, 'Y': 0.75}
                self.filter_eff = {'u': 0.95, 'g': 0.9, 'r': 0.9, 'i': 0.9, 'z': 0.9, 'Y': 0.9}
                self.CCD_eff = {'u': 0.25, 'g': 0.59, 'r': 0.75, 'i': 0.85, 'z': 0.85, 'Y': 0.5}

                # geographical location (requires ephem)
                if ephemimport:
                    self.location = ephem.Observer()
                    self.location.lat = "-30:10:10.78"
                    self.location.lon = "-70:48:23.49"
                    self.location.elevation = 2207 # m
                
            elif kwargs["observatory"] == "CFHT-MegaCam":

                # http://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/generalinformation.html
                
                self.area = 8.0216 # effective area to reproduce their reported etendue of 6.0
                self.FoV = 0.94 * 0.94 # deg2
                self.pixscale = 0.187 # arcsec
                self.RON = 5 # e-
                self.gain = 1.67 # e-/ADU
                self.readouttime = 40. # s
                self.filterchange = 90. # s

                self.slewtime = 30. # s # GUESS
                self.vig = 1.0 # vignetting # GUESS
                
                # mirror, lens, filter and CCD efficiencies at filters
                self.primary_eff = {'u': 0.93, 'g': 0.93, 'r': 0.9, 'i': 0.87, 'z': 0.9, 'Y': 0.9}
                self.corrector_eff = {'u': 0.7, 'g': 0.85, 'r': 0.8, 'i': 0.85, 'z': 0.85, 'Y': 0.6}
                self.filter_eff = {'u': 0.89, 'g': 0.94, 'r': 0.97, 'i': 0.96, 'z': 0.96, 'Y': 0.96}
                self.CCD_eff = {'u': 0.6, 'g': 0.85, 'r': 0.78, 'i': 0.6, 'z': 0.2, 'Y': 0.1} 

                # geographical location (requires ephem)
                if ephemimport:
                    self.location = ephem.Observer()
                    self.location.lat = "+19:49:30.720"
                    self.location.lon = "-155:28:7.680"
                    self.location.elevation = 4204 # m

            elif kwargs["observatory"] == "KMTNet":

                self.area = 1.5 # effective area to reproduce their reported etendue of 6.0
                self.FoV = 4.0
                self.pixscale = 0.4
                self.RON = 7 # e-
                self.gain = 4 #?
                self.readouttime = 30. # s

                self.filterchange = 8 # s # GUESS
                self.slewtime = 30. # s # GUESS
                self.vig = 1.0 # vignetting # GUESS
                
                # mirror, lens, filter and CCD efficiencies at filters
                self.primary_eff = {'u': 1., 'g': 1., 'r': 1., 'i': 1., 'z': 1., 'Y': 1.}
                self.corrector_eff = {'u': 1., 'g': 1., 'r': 1., 'i': 1., 'z': 1., 'Y': 1.}
                self.filter_eff = {'u': 1., 'g': 1., 'r': 1., 'i': 1., 'z': 1., 'Y': 1.}
                self.CCD_eff = {'u': 0.6, 'g': 0.85, 'r': 0.86, 'i': 0.86, 'z': 0.85, 'Y': 0.5} # rough guess based on BVRI

                # geographical location (requires ephem)
                if ephemimport:
                    self.location = ephem.Observer()
                    self.location.lat = "-30:10:10.78"
                    self.location.lon = "-70:48:23.49"
                    self.location.elevation = 2207 # m

            elif kwargs["observatory"] == "VST-OmegaCam":

                self.area = 5.5
                self.FoV = 1.0
                self.pixscale = 0.21
                self.RON = 6 # e-
                self.gain = 2.5 # e-/ADU
                self.readouttime = 45. # s

                self.filterchange = 8 # s # GUESS
                self.slewtime = 30. # s # GUESS
                self.vig = 1.0 # vignetting # GUESS

                # mirror, lens, filter and CCD efficiencies at filters
                self.primary_eff = {'u': 1., 'g': 1., 'r': 1., 'i': 1., 'z': 1., 'Y': 1.}
                self.corrector_eff = {'u': 1., 'g': 1., 'r': 1., 'i': 1., 'z': 1., 'Y': 1.}
                self.filter_eff = {'u': 0.8, 'g': 0.95, 'r': 0.95, 'i': 0.95, 'z': 0.95, 'Y': 1. }
                self.CCD_eff = {'u': 0.7, 'g': 0.8, 'r': 0.75, 'i': 0.55, 'z': 0.2, 'Y': 0.1} # rough guess based on figure
                
                # geographical location (requires ephem)
                if ephemimport:
                    self.location = ephem.Observer()
                    self.location.lat = "-24:37:34.79"
                    self.location.lon = "-70:24:14.27"
                    self.location.elevation = 2635 # m
                    
            elif kwargs["observatory"] == "LSST":
                
                self.area = 35.04
                self.FoV = 9.6
                self.pixscale = 0.2
                self.gain = 4 #?
                self.overhead = 5 # s

                
            elif kwargs["observatory"] == "Subaru-HSC":

                self.area = 53.0 # m2
                self.FoV = 1.77 # deg2
                self.pixscale = 0.17 # "
                self.gain = 3.0 # e-/ADU
                self.readouttime = 20 # s
                self.RON = 4.5 # e-
                self.saturation = 150000 # e-
                self.filterchangetime = 1800 # s

            elif kwargs["observatory"] == "ATLAS":
                
                self.area = 0.196
                self.FoV = 29.16
                self.pixscale = 1.86
                self.gain = 4 #?
                
        else:
            
            self.area = kwargs["area"] # m2
            self.FoV = kwargs["FOV"] # deg2
            self.pixscale = kwargs["pixscale"]
            self.gain = kwargs["gain"]

            
if __name__ == "__main__":

    print "Observatory class testing"

    obs = observatory(observatory = "Blanco-DECam")

    print obs.location
