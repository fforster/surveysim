# Metric that uses the surveysim tools

from lsst.sims.maf.metrics import BaseMetric
import numpy as np
from obsplan import *
from survey import *

class surveysim_metric(BaseMetric):
    """
    Metric that uses surveysim to measure the number of events of a given kind detected
    """
    def __init__(self, TimeCol = 'expMJD', m5Col = 'fiveSigmaDepth', filterCol = 'filter', **kwargs):

        self.TimeCol = TimeCol
        self.m5Col = m5Col
        self.filterCol = filterCol

        self.refsurvey = kwargs.pop("refsurvey")
        #self.LCz = kwargs.pop("LCz")
        #self.Avs = kwargs.pop("Avs")
        #self.Rv = kwargs.pop("Rv")
        #self.lAv = kwargs.pop("lAv")
        #self.SFH = kwargs.pop("SFH")
        #self.efficiency = kwargs.pop("efficiency")
        #self.filtername = kwargs.pop("filtername")
        #self.nz = kwargs.pop("nz")
        #self.maxrestframeage = kwargs.pop("maxrestframeage")
        
        super(surveysim_metric, self).__init__(col = [self.TimeCol, self.m5Col, self.filterCol], **kwargs)

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
        
        # works only when selecting one band via sql
        plan = obsplan(obsname = 'LSST', mode = 'maf', MJDs = times, limmag = mlim, band = filters[0])

        # new survey
        minisurvey = survey(obsplan = plan, LCz = self.refsurvey.LCz, Avs = self.refsurvey.Avs, Rv = self.refsurvey.Rv, lAv = self.refsurvey.lAv, SFH = self.refsurvey.SFH, efficiency = self.refsurvey.efficiency, filtername = self.refsurvey.filtername, nz = self.refsurvey.nz, maxrestframeage = self.refsurvey.maxrestframeage)
        
        # check attributes and inherit everything from reference survey
        keysms = dir(minisurvey)
        keysref = dir(self.refsurvey)
        
        # inherit non obsplan attributes
        for key in keysref:
            if key not in keysms and key != "obsplan":
                setattr(minisurvey, key, getattr(self.refsurvey, key))
        
        # compute maximum length of simulation by doing one simulation
        minisurvey.ndayssim = minisurvey.LCz_Av.simulate_randomLC(nsim = 1, iz = 2, MJDs = minisurvey.obsplan.MJDs, maxrestframeage = minisurvey.maxrestframeage)[0]
        
        # sample 1 event from the distribution
        nsim = 1000
        minisurvey.sample_events(nsim = nsim)

        #print plan.MJDs
        #print minisurvey.ndetections
        # plot the light curves
        #minisurvey.plot_LCs(save = False)

        # return number of detections
        return np.sum(minisurvey.ndetections) / nsim * minisurvey.cumtotalSNe[-1]

