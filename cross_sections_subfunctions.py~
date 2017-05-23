# -*- coding: iso-8859-1 -*-
"""
Functions to compute the mean cross-section in each bin.
"""
import numpy as np
import pdb
import matplotlib.pyplot as plt
import scipy.stats
from scipy import interpolate as interp
from matplotlib.pyplot import cm
import cPickle as pickle

micron2cm=1.e-4 #1 micron in cm
"""
***************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************
"""

def compute_band_cross_section(leftedges, rightedges, N_layers, molecule):
	"""
	Objective of this code is to compute the per-molecule cross-section of a given molecule.
	Inputs:
	-left edges of the wavelength bins (nm)
	-right edges of the wavelength bins (nm)
	-Number of atmospheric layers
	-molecule

	Output:
	per-molecule total extinction cross-section in cm^2
	per-molecule absorption cross-section in cm^2
	per-molecule scattering cross-section in cm^2

	2D datastructures are returned, the second dimension of which corresponds to atmospheric layers. Identical cross-sections are returned for each layer. This seemingly superflous step is taken in order to permit the introduction of temperature dependence, which requires the ability to specify cross-sections unique to each layer
	"""
	import numpy as np
	import scipy.integrate
	from scipy import interpolate as interp

	n_bins=len(leftedges)
	#import data
	data=np.genfromtxt('./XCs/composite_xc_extended_'+molecule, skip_header=1, skip_footer=0)
	wav=data[:,0] #wavelengths in nm
	tot_xc=data[:,1] #total xc in cm2, rayleigh+abs
	abs_xc=data[:,2] #absorption xc in cm2
	ray_xc=data[:,3] #rayleigh scattering xc in cm2

	#form functions of cross-sections
	tot_xc_func=interp.interp1d(wav, tot_xc, kind='linear')
	#abs_xc_func=interp.interp1d(wav, abs_xc, kind='linear')
	ray_xc_func=interp.interp1d(wav, ray_xc, kind='linear')
	
	#initialize variables to hold the bandpass-integrated cross-sections\
	tot_xc_band_layer=np.zeros([n_bins, N_layers])
	#abs_xc_band_layer=np.zeros([n_bins, N_layers])
	ray_xc_band_layer=np.zeros([n_bins, N_layers])
	
	for ind in range(0,n_bins):
		#find average cross-sections by integrating across band and dividing by size of bandpass...
		tot_xc_band_layer[ind,0]=scipy.integrate.quad(tot_xc_func, leftedges[ind], rightedges[ind], epsabs=0, epsrel=1.e-2, limit=1000)[0]/(rightedges[ind]-leftedges[ind])
		#abs_xc_band_layer[ind,0]=scipy.integrate.quad(abs_xc_func, leftedges[ind], rightedges[ind], epsabs=0, epsrel=1.e-1, limit=1000)[0]/(rightedges[ind]-leftedges[ind])
		ray_xc_band_layer[ind,0]=scipy.integrate.quad(ray_xc_func, leftedges[ind], rightedges[ind], epsabs=0, epsrel=1.e-2, limit=1000)[0]/(rightedges[ind]-leftedges[ind])
	
	for ind in range(1, N_layers):
		tot_xc_band_layer[:, ind]=tot_xc_band_layer[:, 0]
		#abs_xc_band_layer[:, ind]=abs_xc_band_layer[:, 0]
		ray_xc_band_layer[:, ind]=ray_xc_band_layer[:, 0]
		
	#abs_xc_band_layer=tot_xc_band_layer-ray_xc_band_layer #self-consistent, faster way of doing calculation...
	
	return (tot_xc_band_layer, ray_xc_band_layer)
#Test to make sure it is precisely repeated...

def compute_band_cross_section_td(leftedges, rightedges, temps,molecule):
	"""
	Objective of this code is to compute the per-molecule cross-section of a given molecule, on a temperature-dependent basis. 
	Inputs:
	-left edges of the wavelength bins (nm)
	-right edges of the wavelength bins (nm)
	-temperature in each atmospheric layer (K)
	-molecule (currently implemented for: CO2)

	Output:
	per-molecule total extinction cross-section in cm^2
	per-molecule absorption cross-section in cm^2
	per-molecule scattering cross-section in cm^2
	"""
	import numpy as np
	import scipy.integrate
	from scipy import interpolate as interp

	n_bins=len(leftedges) #number of wavelength bins
	n_layers=len(temps) #number of layers with temperatures

	#initialize variables to hold the bandpass-integrated cross-sections\
	tot_xc_band_layer=np.zeros([n_bins,n_layers])
	#abs_xc_band_layer=np.zeros([n_bins,n_layers])
	ray_xc_band_layer=np.zeros([n_bins,n_layers])
	
	#Load molecule-specific information
	if molecule=='co2':
		RT=300. #temperature of the room temperature dataset (K)
		LT=195. #temperature of the low temperature dataset (K)
		
		#import data
		co2_wav_195, co2_tot_xc_195, co2_abs_xc_195, co2_ray_xc_195=np.genfromtxt('./XCs/composite_xc_extended_co2-195', skip_header=1, skip_footer=0,usecols=(0,1,2,3), unpack=True) #low-temperature (195K dataset)
		co2_wav_300, co2_tot_xc_300, co2_abs_xc_300, co2_ray_xc_300=np.genfromtxt('./XCs/composite_xc_extended_co2', skip_header=1, skip_footer=0,usecols=(0,1,2,3), unpack=True) #room temperature (nominally 300K) dataset
		
		#Interpolate low-temperature data to the room-temperature data's scale
		lt_tot_xc_func=interp.interp1d(co2_wav_195, co2_tot_xc_195, kind='linear')
		#lt_abs_xc_func=interp.interp1d(co2_wav_195, co2_abs_xc_195, kind='linear')
		lt_ray_xc_func=interp.interp1d(co2_wav_195, co2_ray_xc_195, kind='linear')

		rt_wav=co2_wav_300
		rt_tot_xc=co2_tot_xc_300
		#rt_abs_xc=co2_abs_xc_300
		rt_ray_xc=co2_ray_xc_300
		
		lt_tot_xc_interp=lt_tot_xc_func(rt_wav)
		#lt_abs_xc_interp=lt_abs_xc_func(rt_wav)
		lt_ray_xc_interp=lt_ray_xc_func(rt_wav)
	elif molecule=='so2':
		RT=293. #temperature of the room temperature dataset (K)
		LT=200. #temperature of the low temperature dataset (K)
		
		#import data
		so2_wav_200, so2_tot_xc_200, so2_abs_xc_200, so2_ray_xc_200=np.genfromtxt('./XCs/composite_xc_extended_so2-200', skip_header=1, skip_footer=0,usecols=(0,1,2,3), unpack=True) #low-temperature (200K dataset)
		so2_wav_293, so2_tot_xc_293, so2_abs_xc_293, so2_ray_xc_293=np.genfromtxt('./XCs/composite_xc_extended_so2', skip_header=1, skip_footer=0,usecols=(0,1,2,3), unpack=True) #room temperature (nominally 293K) dataset
		
		#Interpolate low-temperature data to the room-temperature data's scale
		lt_tot_xc_func=interp.interp1d(so2_wav_200, so2_tot_xc_200, kind='linear')
		#lt_abs_xc_func=interp.interp1d(so2_wav_200, so2_abs_xc_200, kind='linear')
		lt_ray_xc_func=interp.interp1d(so2_wav_200, so2_ray_xc_200, kind='linear')

		rt_wav=so2_wav_293
		rt_tot_xc=so2_tot_xc_293
		#rt_abs_xc=so2_abs_xc_293
		rt_ray_xc=so2_ray_xc_293
		
		lt_tot_xc_interp=lt_tot_xc_func(rt_wav)
		#lt_abs_xc_interp=lt_abs_xc_func(rt_wav)
		lt_ray_xc_interp=lt_ray_xc_func(rt_wav)

	else:
		print 'Error: invalid value for molecule'
	
	#With the molecule-specific info loaded, calculate cross-sections
	#For each atmospheric layer:
	for l_ind in range(0, n_layers):
		print l_ind
		T=temps[l_ind] #temperature of layer
		
		#Form interpolated temperature functions, for integration
		if (T<=LT): #if the temperature goes below the range we have coverage for, just use the low temperature cross-sections
			tot_xc_func=lt_tot_xc_func
			#abs_xc_func=lt_abs_xc_func
			ray_xc_func=lt_ray_xc_func
		elif (T>=RT): #if the temperature goes above the range we have coverage for, just use the high temperature cross-sections
			tot_xc_func=interp.interp1d(rt_wav, rt_tot_xc, kind='linear')
			#abs_xc_func=interp.interp1d(rt_wav, rt_abs_xc, kind='linear')
			ray_xc_func=interp.interp1d(rt_wav, rt_ray_xc, kind='linear')
		else: #otherwise, the temperature is in the intermediate range we have coverage for, and it's linear interpolation time!
			lt_weight=(RT-T)/(RT-LT)
			rt_weight=(T-LT)/(RT-LT)

			td_tot_xc=lt_tot_xc_interp*lt_weight+rt_tot_xc*rt_weight
			#td_abs_xc=lt_abs_xc_interp*lt_weight+rt_abs_xc*rt_weight
			td_ray_xc=lt_ray_xc_interp*lt_weight+rt_ray_xc*rt_weight

			tot_xc_func=interp.interp1d(rt_wav, td_tot_xc, kind='linear')
			#abs_xc_func=interp.interp1d(rt_wav, td_abs_xc, kind='linear')
			ray_xc_func=interp.interp1d(rt_wav, td_ray_xc, kind='linear')
		#Step over each bin and calculate the cross-section
		for b_ind in range(0,n_bins):
			#find average cross-sections by integrating across band and dividing by size of bandpass...
			tot_xc_band_layer[b_ind,l_ind]=scipy.integrate.quad(tot_xc_func, leftedges[b_ind], rightedges[b_ind], epsabs=0, epsrel=1.e-2, limit=1000)[0]/(rightedges[b_ind]-leftedges[b_ind])
			#abs_xc_band_layer[b_ind,l_ind]=scipy.integrate.quad(abs_xc_func, leftedges[b_ind], rightedges[b_ind], epsabs=0, epsrel=1.e-1, limit=1000)[0]/(rightedges[b_ind]-leftedges[b_ind])
			ray_xc_band_layer[b_ind,l_ind]=scipy.integrate.quad(ray_xc_func, leftedges[b_ind], rightedges[b_ind], epsabs=0, epsrel=1.e-2, limit=1000)[0]/(rightedges[b_ind]-leftedges[b_ind])
	#abs_xc_band_layer=tot_xc_band_layer-ray_xc_band_layer
	return (tot_xc_band_layer, ray_xc_band_layer)

##########Test compute_band_cross_section_td for the CO2 case
####Get data to plot
##Set inputs
#leftedges=np.arange(100., 500., step=1.)
#rightedges=np.arange(101., 501., step=1.)
#centers=0.5*(leftedges+rightedges)
#temps=np.linspace(150, 350., num=10)

##calculate TD cross-sections
#tot_xc, ray_xc=compute_band_cross_section_td(leftedges, rightedges, temps, 'co2')

##load raw data files as further check
#co2_wav_195, co2_tot_xc_195, co2_abs_xc_195, co2_ray_xc_195=np.genfromtxt('./XCs/composite_xc_extended_co2-195', skip_header=1, skip_footer=0,usecols=(0,1,2,3), unpack=True) #low-temperature (195K dataset)
#co2_wav_300, co2_tot_xc_300, co2_abs_xc_300, co2_ray_xc_300=np.genfromtxt('./XCs/composite_xc_extended_co2', skip_header=1, skip_footer=0,usecols=(0,1,2,3), unpack=True)


####Plot
#fig1, (ax1, ax2)=plt.subplots(2, figsize=(8,11), sharex=True, sharey=False)
#colorseq1=iter(cm.rainbow(np.linspace(0,1,len(temps))))
#colorseq2=iter(cm.rainbow(np.linspace(0,1,len(temps))))

#ax1.set_title('Total Extinction')
#ax1.set_ylabel('Cross-Section (cm^2/molecule)')
#ax1.plot(co2_wav_195, co2_tot_xc_195, color='blue', linewidth=3., label='LT Data')
#ax1.plot(co2_wav_300, co2_tot_xc_300, color='red', linewidth=3., label='RT Data')
#ax1.set_yscale('log')

#ax2.set_title('Relative Extinction')
#ax2.set_ylabel('Extinction/LT Extinction')
#ax2.set_xlabel('Wavelength (nm)')
#ax2.set_yscale('linear')
#ax2.set_xlim([100., 250.])

#for ind in range(0, len(temps)):
	#ax1.plot(centers, tot_xc[:,ind], marker='s', linestyle='--', color=next(colorseq1), label=str(np.round(temps[ind],1)))
	#ax2.plot(centers, tot_xc[:,ind]/tot_xc[:,0], marker='s', linestyle='--', color=next(colorseq2), label=str(temps[ind]))


#plt.tight_layout(rect=(0,0,1,0.85))
#ax1.legend(bbox_to_anchor=[0, 1.13, 1., .152], loc=3, ncol=3, mode='expand', borderaxespad=0., fontsize=14)
#plt.savefig('./Plots/tdco2.pdf', orientation='portrait',papertype='letter', format='pdf')
#plt.show()

def compute_cloud_params(leftedges, rightedges, N_layers, picklefile):
	"""
	This code calculates the cloud optical parameters (sigma, w_0, g) in each layer on a per-molecule basis. It is essentially a selector function for the 
	"""
	n_bins=len(leftedges)
	
	f=open(picklefile, 'r')
	wav, sigma, w_0, g, qsca=pickle.load(f) #units: nm, microns**2, dimless, dimless, dimless NEED TO CHECK ALL UNITS
	####sigma_cgs=(sigma*(micron2cm)**2)/w_0 #convert XC from microns**2 to cm**2 #Also, convert from SCATTERING XC to TOTAL XC. Temp kludge until whole code properly fixed. 
	sigma_cgs=sigma*(micron2cm)**2 #convert XC from microns**2 to cm**2
		
	#sigma_2=np.pi*10.**2.*qsca
	#print np.max(np.abs(sigma_2-sigma)/sigma)
	#print np.min(np.abs(sigma_2-sigma)/sigma)
	##Sigma_2 is 4/3 the value of sigma, what gives??? Is the distribution just slightly biased towards the smaller cross-sections?
	
	#form functions of cross-sections
	sigma_func=interp.interp1d(wav, sigma_cgs, kind='linear')
	w_0_func=interp.interp1d(wav, w_0, kind='linear')
	g_func=interp.interp1d(wav, g, kind='linear')
	
	#initialize variables to hold the bandpass-integrated cross-sections\
	sigma_band_layer=np.zeros([n_bins, N_layers])
	w_0_band_layer=np.zeros([n_bins, N_layers])
	g_band_layer=np.zeros([n_bins, N_layers])
	
	for ind in range(0,n_bins):
		#find average cross-sections by integrating across band and dividing by size of bandpass...
		sigma_band_layer[ind,0]=scipy.integrate.quad(sigma_func, leftedges[ind], rightedges[ind], epsabs=0, epsrel=1.e-5, limit=200)[0]/(rightedges[ind]-leftedges[ind])
		w_0_band_layer[ind,0]=scipy.integrate.quad(w_0_func, leftedges[ind], rightedges[ind], epsabs=0, epsrel=1.e-5, limit=200)[0]/(rightedges[ind]-leftedges[ind])
		g_band_layer[ind,0]=scipy.integrate.quad(g_func, leftedges[ind], rightedges[ind], epsabs=0, epsrel=1.e-5, limit=200)[0]/(rightedges[ind]-leftedges[ind])
	
	for ind in range(1, N_layers):
		sigma_band_layer[:, ind]=sigma_band_layer[:, 0]
		w_0_band_layer[:, ind]=w_0_band_layer[:, 0]
		g_band_layer[:, ind]=g_band_layer[:, 0]
			
	return (sigma_band_layer, w_0_band_layer, g_band_layer)

#compute_cloud_params(np.arange(100., 500., step=1.), np.arange(101., 501., step=1.), 10, './cloud_h2o_reff10_vareff0p1_lognormal.pickle')

def get_rugheimer_xc(leftedges, rightedges, N_layers,molecule, mr_n2, mr_co2):
	"""
	Objective of this code is to load the Rugheimer molecular cross-sections. They are taken from photos.pdat. Note that this assumes preset wavelength bins.
	-left and right edges of wavelength bins in nm.
	-molecule
	-mixing ratio of n2
	-mixing ratio of co2

	Output:
	per-molecule absorption cross-section in cm^2
	per-molecule scattering cross-section in cm^2

	"""
	import numpy as np
	import scipy.integrate
	from scipy import interpolate as interp

	n_bins=len(leftedges)
	
	abs_xc_band=np.zeros(n_bins)
	ray_xc_band=np.zeros(n_bins)
	
	faruvdata=np.genfromtxt('./Raw_Data/Rugheimer_Metadata/faruvs_mod.pdat', skip_header=3, skip_footer=1)

	nearuv0=np.genfromtxt('./Raw_Data/Rugheimer_Metadata/photos.pdat',skip_header=2, skip_footer=400)
	nearuv1=np.genfromtxt('./Raw_Data/Rugheimer_Metadata/photos.pdat',skip_header=159, skip_footer=319)
	nearuv2=np.genfromtxt('./Raw_Data/Rugheimer_Metadata/photos.pdat',skip_header=397, skip_footer=70)

	if molecule=='o2':
		abs_xc_band[0:9]=faruvdata[::-1, 1]
		abs_xc_band[9:44]=nearuv1[:,1]
	if molecule=='co2':
		abs_xc_band[0:9]=faruvdata[::-1, 2]		
		abs_xc_band[9:44]=nearuv1[:,3]
	if molecule=='h2o':
		abs_xc_band[0:9]=faruvdata[::-1, 3]	
		abs_xc_band[9:44]=nearuv1[:,2]
	if molecule=='so2':
		abs_xc_band[0:9]=faruvdata[::-1, 4]	
		abs_xc_band[9:(9+68)]=nearuv2[:,2]+nearuv2[:,3]+nearuv2[:,4]	
	if molecule=='o3':
		abs_xc_band[9:(9+108)]=nearuv0[:,3]#+nearuv0[:,4]
	if molecule=='h2s':
		abs_xc_band[9:(9+68)]=nearuv2[:,5]
	if molecule=='n2':
		#no absorption from n2, only Rayleigh scattering. Take all Rayleigh to come from N2
		#Compute Rayleigh scattering according to method of SIGRAY in ltning.f and the modification in photo.f
		wavcen=0.5*(leftedges+rightedges)*1.e-3 #convert to microns
		ray_xc_band=(4.006e-28*(1.+0.0113/wavcen**2.+0.00013/wavcen**4.)/wavcen**4.)*(1.+1.5*mr_co2)/mr_n2 #scale by the mixing ratio of N2 to account for correction.
	
	tot_xc_band=abs_xc_band+ray_xc_band

	#initialize variables to hold the bandpass-integrated cross-sections PER LAYER
	tot_xc_band_layer=np.zeros([n_bins, N_layers])
	ray_xc_band_layer=np.zeros([n_bins, N_layers])
	
	for ind in range(0, N_layers):
		tot_xc_band_layer[:, ind]=tot_xc_band
		ray_xc_band_layer[:, ind]=ray_xc_band
	
	
	
	return (tot_xc_band_layer, ray_xc_band_layer)