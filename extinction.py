import numpy as np

class extinction(object):
    
    def __init__(self, **kwargs):
        
        self.Av = kwargs["Av"]
        self.Rv = kwargs["Rv"]

        if "model" in kwargs.keys():
            self.model = kwargs["model"]
        else:
            self.model = "G03"

        # CCM89+O94 (Cardelli, Clayton, Mathis 1989, O'Donnell 1994), FM90 (Fitzpatrick & Massa 1990), G03 (Gordon et al. 2003)


    def dm(self, **kwargs): 
        
        x = 1. / (1e-4 * kwargs["lambdasAA"])  # from AA to microm

        if self.model == "G03" or self.model == "FM90" or self.model == "CCM89+O94":

            # G03
            # LMC average
            c1 = -0.890
            c2 = 0.998
            c3 = 2.719
            c4 = 0.400
            x0 = 4.579
            gamma = 0.934

            # FM90
            if self.model == "FM90":
                # LMC average 
                c1 = -0.687
                c2 = 0.891
                c3 = 2.550
                c4 = 0.504
                x0 = 4.608
                gamma = 0.994
                
            x2 = x * x
            D = x2 / ((x2 - x0**2)**2 + x2 * gamma)
            F = np.zeros(len(x))
            
            if max(x) >= 5.9:
                mask = x >= 5.9
                F[mask] = 0.5392 * (x[mask] - 5.9)**2 + 0.05644 * (x[mask] - 5.9)**3
                
            AlambdaAv = 1. + (c1 + c2 * x + c3 * D + c4 * F) / self.Rv

        if self.model == "CCM89+O94":
        
            IR = (x >= 0.3) & (x < 1.1)
            NIRopt = (x >= 1.1) & (x < 3.3)
            UV = np.array(x >= 3.3)
            
            a = np.zeros(np.shape(x))
            b = np.zeros(np.shape(x))
            
            x161 = x[IR]**1.61
            a[IR] = 0.574 * x161
            b[IR] = -0.527 * x161
            
            y = x[NIRopt] - 1.82
            y2 = y * y
            y3 = y2 * y
            y4 = y3 * y
            y5 = y4 * y
            y6 = y5 * y
            y7 = y6 * y
            a[NIRopt] = 1. + 0.17699 * y - 0.50447 * y2 - 0.02427 * y3 + 0.72085 * y4 + 0.01979 * y5 - 0.77530 * y6 + 0.32999 * y7
            b[NIRopt] = 1.41338 * y + 2.28305 * y2 + 1.07233 * y3 - 5.38434 * y4 - 0.62251 * y5 + 5.30260 * y6 - 2.09002 * y7
            
            AlambdaAvCCM = a + b / self.Rv

            # extrapolate to shorter wavelengths using G03
            ilim = np.argmin(np.abs(x - 3.3))
            if np.sum(UV) > 0:
                AlambdaAv[UV] = AlambdaAvCCM[ilim] / AlambdaAv[ilim] * AlambdaAv[UV]
            if np.sum(np.invert(UV)) > 0:
                AlambdaAv[np.invert(UV)] = AlambdaAvCCM[np.invert(UV)]
            
                    
        return (x, self.Av * AlambdaAv)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    lambdas = np.linspace(1000, 20000, 100000)
    iV = np.argmin(np.abs(lambdas - 5510))
    iB = np.argmin(np.abs(lambdas - 4450))
    #lambdas = np.array([4450, 5500])
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax2.grid()

    for model in ["FM90", "CCM89+O94"]:

        print model

        for Rv in [2., 4., 6.]:

            #for Av in [0.1, 1., 2., 5.]:

            Av = 1.
            if model == "FM90":
                ls = '-'
            else:
                ls = '--'
                
            EBV = Av / Rv
            ext = extinction(Av = Av, Rv = Rv, model = model)
            x, y = ext.dm(lambdasAA = lambdas)
            ax2.plot(x, lambdas, c = 'gray')
            ax.plot(x, y, label = "%s - Rv: %4.1f" % (model, Rv), lw = Rv, ls = ls)
            
            #ax.plot(x, (y - y[iV]) / EBV)
                
            
    ax.axvline(x[iV])
    ax.axvline(x[iB])
    ax.axhline(1, c = 'gray')
    ax2.axhline(5510)
    ax2.axhline(4450)
    ax.legend(loc = 2, fontsize = 10)
    plt.show()
