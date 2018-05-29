import numpy as np

class SFHs(object):

    def __init__(self, **kwargs):
        
        self.label = kwargs["SFH"]

        self.SFRlonglabels = ['Madau & Dickinson 2014', 'Horiuchi+ 2011', 'Cole', 'SNIa-Perrets']
        self.SFRlabels = ['MD14', 'H11', 'Cole', 'Ia-P12']

    def doSFR(self, z):

        if self.label == self.SFRlabels[0]:  #MD14

            # Madau & Dickinson 2014, ARAA, eq 15

            self.SFR = 0.015 * (1. + z)**2.7 / (1. + ((1. + z) / 2.9)**5.6) # Msun/yr/Mpc3

        elif self.label == self.SFRlabels[1]: #H11

            # Horiuchi et al. 2011 + Hopkins & Beacom 2006
            
            a = 3.4
            b =-0.3
            c = -3.5
            rho0 = 0.016
            eta = -10
            B = 2.**(1. - a / b)
            C = 2.**((b - a) / c) * 5.**(1. - b / c)
            
            self.SFR = (rho0 * ((1. + z)**(a * eta) + ((1. + z)*1.0 / B)**(b * eta) + ((1. + z)*1.0 / C)**(c * eta))**(1. / eta))

        elif self.label == self.SFRlabels[2]: #Cole

            # Cole???

            a1 = 0.017
            b1 = 0.13
            c1 = 3.3
            d1 = 5.3
            
            self.SFR = (h * (a1 + b1 * z) / (1. + (z / c1)**d1))

        elif self.label == self.SFRlabels[3]: # Ia Perrets et al. 2012

            alpha = 2.11 # +- 0.28
            r0 = 0.17 * 1e-4 # +- 0.03

            self.SFR = r0 * (1 + z)**alpha # SNIa/yr/Mpc # must use efficiency of 1.0
            
if __name__ == "__main__":

    SFH = SFHs(SFH = "MD14")

    print(SFH.doSFR(0.4))
