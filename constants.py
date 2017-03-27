import numpy as np

# physical constants

h = 6.6260755e-27       # erg s
G = 6.67259e-8          # cm3 g-1 s-2
k = 1.380658e-16        # erg K-1
sigmaSB = 5.6704e-5     # erg cm-2 s-1 K-4
eV = 1.6021772e-12      # erg
Rsun = 6.96e10          # cm
Msun = 1.99e33          # g
Mchandra = 1.44 # Chandraskhar mass in Msun
sigmaT = 6.652458734e-25 # Thomson cross-section [cm2]
mp = 1.6726219e-24 # proton mass in [g]
yr2sec = 31556926       # seconds
yr2days = 365.242199    # days
days2sec = yr2sec / yr2days # seconds
pc2cm = 3.08567758e18   # cm
cm2AA = 1e8
cspeed = 2.99792458e10  # cm s-1
cspeedAAs = cspeed * cm2AA  # A s-1
Hnot = 71               # km s-1 Mpc-1
Hnotcgs = Hnot * 1e5 / (1e6 * pc2cm)
OmegaM = 0.27
OmegaL = 0.73
ster_sqdeg = (180. / np.pi)**2 # 1 ster in square degrees (~3282.8)

