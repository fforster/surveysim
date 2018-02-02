# Metric that uses the surveysim tools

from lsst.sims.maf.metrics import BaseMetric
import numpy as np
from obsplan import *
from survey_multimodel import *

class survey_multimodel_metric(BaseMetric):
    """
    Metric that uses surveysim to measure the number of events of a given kind detected
    """
    def __init__(self, TimeCol = 'expMJD', m5Col = 'fiveSigmaDepth', filterCol = 'filter', **kwargs):

        self.TimeCol = TimeCol
        self.m5Col = m5Col
        self.filterCol = filterCol

        self.refsurvey = kwargs.pop("refsurvey")
        self.rvs = kwargs.pop("rvs")
        self.bounds = kwargs.pop("bounds")
        self.pars = kwargs.pop("pars")
        self.nsim = kwargs.pop("nsim")
        self.maxz = kwargs.pop("maxz")
        
        super(survey_multimodel_metric, self).__init__(col = [self.TimeCol, self.m5Col, self.filterCol], **kwargs)

    """
    Run the metric:
    1. Start counting number of exposures
    """
    def run(self, dataSlice, slicePoint = None):

        times = dataSlice[self.TimeCol]
        mlim = dataSlice[self.m5Col]
        filters = dataSlice[self.filterCol]

        # sort times
        idxsort = np.argsort(times)
        times = times[idxsort]
        mlim = mlim[idxsort]
        filters = filters[idxsort]
        
        # create observational plan
        plan = obsplan(obsname = 'LSST', mode = 'maf', MJDs = times, limmag = mlim, bands = filters)

        # increase maf counter
        self.refsurvey.mafcounter += 1
        
        # update times and bands for LCs object in reference survey
        self.refsurvey.LCs.set_observations(mjd = times, filters = filters, \
                                            flux = None, e_flux = None,  \
                                            objname = None, plot = False, \
                                            bandcolors = self.refsurvey.LCs.bandcolors)
        
        # new survey
        minisurvey = survey_multimodel(obsplan = plan, SFH = self.refsurvey.SFH, efficiency = self.refsurvey.efficiency, LCs = self.refsurvey.LCs)
        
        # check attributes and inherit everything from reference survey
        keysms = dir(minisurvey)
        keysref = dir(self.refsurvey)
        
        # inherit non obsplan attributes
        for key in keysref:
            if key not in keysms and key != "obsplan" and key != "efficiency" and key != "SFH":
                setattr(minisurvey, key, getattr(self.refsurvey, key))
        
        # update variables specific to this plan (need to fix do_cosmology)
        minisurvey.set_maxz(self.maxz)
        minisurvey.do_cosmology()

        # update rvs and bounds
        minMJD, maxMJD = min(times) - (20. + self.refsurvey.maxrestframeage) * (1. + max(self.refsurvey.LCs.zs)), max(times)
        self.rvs["texp"] = (lambda nsim: uniform.rvs(loc = minMJD, scale = maxMJD - minMJD, size = nsim))
        self.bounds["texp"] = [minMJD, maxMJD]
        
        # sample nsim events
        minisurvey.sample_events(nsim = self.nsim, doload = False, doplot = False, rvs = self.rvs, bounds = self.bounds, pars = self.pars)

        # check efficiency
        minisurvey.do_efficiency(doplot = False, verbose = False, check1stdetection = False)

        # print counter
        print("\r%i" % (self.refsurvey.mafcounter), end = "")

        #print(minisurvey.x_effs.keys())
        
        #print plan.MJDs
        #print minisurvey.ndetections
        # plot the light curves
        #minisurvey.plot_LCs(save = False)

        # return number of detections
        return minisurvey.y_effs['texp'][-1] * minisurvey.cumtotalSNe[-1] #np.sum(minisurvey.ndetections
        #return minisurvey.cumtotalSNe[-1] #np.sum(minisurvey.ndetections) / nsim * minisurvey.cumtotalSNe[-1]

