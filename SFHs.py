import numpy as np

class SFHs(object):

    def __init__(self, **kwargs):
        
        self.label = kwargs["SFH"]

        self.SFRlonglabels = ['Madau & Dickinson 2014', 'Horiuchi+ 2011', 'Cole']
        self.SFRlabels = ['MD14', 'H11', 'Cole']

    def doSFR(self, z):

        if self.label == self.SFRlabels[0]:  #MD14

            # Madau & Dickinson 2014, ARAA, eq 15

            self.SFR = 0.015 * (1. + z)**2.7 / (1. + ((1. + z) / 2.9)**5.6)

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

if __name__ == "__main__":

    SFH = SFHs(SFH = "MD14")
