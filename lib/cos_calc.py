#Module: Cosmological calculator
#Author: Srinivasan Raghunathan
#Date: January, 2011
#Reference: Hogg 2000 (arXiv:astro-ph/9905116)

#Requirements
#python numpy, scipy modules

#call as: fn_cos_calc(h,omega_m,omega_k,omega_lambda,z)

#Input parameters: 
#h: Hubble parameter and H0 = 100*h
#omega_m,omega_k,omega_lambda: Matter, curvature, and cos. constant density of the universe
#z: redshift at which the distances must be calculated

#Module returns the following ouput values:
#For the specified input parameters at the specified redshift
#d_h: Hubble distance - observable universe in Mpc
#d_c: Total line-of-sight comoving distance to redshift z
#d_t: Transverse comoving distance @ z
#d_a: Ang. diameter distance
#d_l: luminosity distance
#dm: Distance modulus (for supernova calculations, etc.)

def fn_cos_calc(h,omega_m,omega_k,omega_lambda,z):

	#Remember to import numpy and scipy modules in the calling program
	#import numpy
	#import scipy

	c = 299797. # VELOCITY OF LIGHT IN Km/s

	H0 = 100.0 * h    # HUBBLE CONSTANT = 100h Km/s/Mpc
	# HUBBLE CONSTANT IN STANDARD UNITS - DIMENSIONS sec-1
	H0_std = (H0/(3.08568025 * 10**19))

	# HUBBLE DISTANCE IN Mpc
	d_h = c/H0

	# INTEGRATION FUNCTION E(z)
        # SCIPY RETURNS THE INTEGRATION RESULT AS A TWO DIMENSIONAL ARRAY. FIRST ELEMENT IS THE RESULT. SECOND ELEMENT IS THE ERROR ASSOCIATED
	def e_z(z):
		return (1.0/math.sqrt(omega_m*((1+z)**3)+ omega_k*((1+z)**2) + omega_lambda))

	e_z_int, e_z_int_err = integrate.quad(e_z,0.,z)
	
	#ALL DISTANCES ARE IN Mpc (SINCE I'VE CALCULATED HUBBLE DISTANCE ABOVE IN Mpc)
	# TOTAL LINE-OF-SIGHT COMOVING DISTANCE	
	d_c = d_h * e_z_int

	# TRANSVERSE COMOVING DISTANCE
	if (omega_k==0.0): #FLAT
		d_t = d_c
	elif (omega_k>0.0): #POSITIVELY CURVED
		d_t = d_h/math.sqrt(omega_k) * math.sinh(math.sqrt(omega_k)*d_c/d_h)
	else: #NEGATIVELY CURVED
		d_t = d_h/math.sqrt(abs(omega_k)) * math.sinh(math.sqrt(abs(omega_k))*d_c/d_h)

	if (omega_lambda==0.0): #UNIVERSE WITHOUT COS. CONSTANT / DARK ENERGY
		d_t = d_h * 2 *(2 - (omega_m *(1-z)) - ((2-omega_m) * (math.sqrt(1+(omega_m*z))))) / (omega_m**2 * (1+z))

	# ANGULAR DIAMETER DISTANCE
	d_a = d_t / (1+z)

	# LUMINOSITY DISTANCE
	d_l = (1+z) * d_t

	# DISTANCE MODULUS
	dm = 5.0 * np.log10(d_l*10**6/10) # 1 Mpc = 10**6 pc
	
	return d_h,d_c,d_t,d_a,d_l,dm
	
#####################################################################################
#
import numpy as np
import scipy as sc
from scipy import integrate
import math
from pylab import *

#h,omega_m,omega_k,omega_lambda=0.71,0.27,0.,0.73
#
##Example 1:
#z=0.1
#results=fn_cos_calc(h,omega_m,omega_k,omega_lambda,z) #results will be array of 1 row, 6 columns
#print results
#
##Example 2: Redshift z as an array and plot luminosity distance
#z=np.arange(0.1,10,1)
#for i in range(len(z)):
#	results=fn_cos_calc(h,omega_m,omega_k,omega_lambda,z[i])
#	print results
#	plot(z[i],results[4],'ro')
#xlabel('Redshift z -->', fontsize=10)
#ylabel('Luminosity distance $d_l$ -->', fontsize=10)
#title('Redshift z vs. Luminosity distance $d_l$', fontsize=12)
#show()
#
##Example 3: Redshift z as an array and faster execution with map(lambda) functions
##results now will be an n-dimesnional array of *** columns = 6; rows = size of redshift array ***
#results=np.asarray(map(lambda x: fn_cos_calc(h,omega_m,omega_k,omega_lambda,x),z[:]))
#print results
#
