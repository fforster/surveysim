# Metric that uses the surveysim tools

from lsst.sims.maf.metrics import BaseMetric
import numpy as np
from obsplan import *
from survey_multimodel import *
from pandas import HDFStore

class survey_multimodel_metric(BaseMetric):
    """
    Metric that uses surveysim to measure the number of events of a given kind detected
    """
    # expMJD
    def __init__(self, TimeCol = 'observationStartMJD', m5Col = 'fiveSigmaDepth', filterCol = 'filter', **kwargs):

        self.TimeCol = TimeCol
        self.m5Col = m5Col
        self.filterCol = filterCol

        self.refsurvey = kwargs.pop("refsurvey")
        self.maxz = kwargs.pop("maxz")
        self.nsim = kwargs.pop("nsim")
        self.rvs = kwargs.pop("rvs")

        # arguments for model grid based LCs
        if "bounds" in kwargs.keys():
            self.bounds = kwargs.pop("bounds")
        if "pars" in kwargs.keys():
            self.pars = kwargs.pop("pars")

        # hdf
        if "hdf" in kwargs.keys():
            self.hdf = kwargs.pop("hdf")
        
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
        minisurvey = survey_multimodel(obsplan = plan, SFH = self.refsurvey.SFH, efficiency = self.refsurvey.efficiency, LCs = self.refsurvey.LCs, doplot = True)
        
        # check attributes and inherit everything from reference survey
        keysms = dir(minisurvey)
        keysref = dir(self.refsurvey)
        
        # inherit non obsplan attributes
        for key in keysref:
            if key not in keysms and key != "obsplan" and key != "efficiency" and key != "SFH" and key != "simLCs":
                setattr(minisurvey, key, getattr(self.refsurvey, key))
        
        # update variables specific to this plan
        minisurvey.set_maxz(self.maxz)
        minisurvey.do_cosmology()

        # update rvs and bounds
        minMJD, maxMJD = min(times) - (20. + self.refsurvey.maxrestframeage) * (1. + max(self.refsurvey.LCs.zs)), max(times)
        self.rvs["texp"] = (lambda nsim: uniform.rvs(loc = minMJD, scale = maxMJD - minMJD, size = nsim))

        # sample nsim events when bound and pars are given (model grid LCs)
        if hasattr(self, "bounds") and hasattr(self, "bounds") and hasattr(self, "pars"):
            self.bounds["texp"] = [minMJD, maxMJD]
            minisurvey.sample_events(nsim = self.nsim, doload = False, doplot = False, rvs = self.rvs, bounds = self.bounds, pars = self.pars)
            # check efficiency
            minisurvey.do_efficiency(doplot = False, verbose = False, check1stdetection = False)
        # sample nsim events when bound and pars not given (PCA LCs)
        else:
            foundLCs, simLCs, simpars = minisurvey.sample_events_PCA(nsim = self.nsim, doload = False, doplot = False, rvs = self.rvs, keepLCs = True)
            if foundLCs:
                simLCs["IDpix"] = self.refsurvey.mafcounter
                simpars["IDpix"] = self.refsurvey.mafcounter
                self.hdf.append("simLCs", simLCs, format = "table", data_columns = True)
                self.hdf.append("simpars", simpars, format = "table", data_columns = True)
            # check efficiency
            minisurvey.do_efficiency_PCA(doplot = False, verbose = False, check1stdetection = False)

        # print counter
        print("\r%i" % (self.refsurvey.mafcounter), end = "")

        #print(minisurvey.x_effs.keys())
        
        #print plan.MJDs
        #print minisurvey.ndetections
        # plot the light curves
        #minisurvey.plot_LCs(save = False)

        # return number of detections
        if "texp" in minisurvey.y_effs.keys() and minisurvey.cumtotalSNe[-1] > 0:
            return minisurvey.y_effs['texp'][-1] * minisurvey.cumtotalSNe[-1] #np.sum(minisurvey.ndetections
        else:
            return 0
        #return minisurvey.cumtotalSNe[-1] #np.sum(minisurvey.ndetections) / nsim * minisurvey.cumtotalSNe[-1]

