# -*- coding: iso-8859-1 -*-
"""
This file contains function definitions and calls that create the atmospheric profile files (T, P, and gas molar concentrations as a function of altitude), that form inputs into our radiative transfer codes. There are two broad families of such files:

I) VALIDATION CASES: these include form_profiles_primitive_earth_rugheimer and form_profiles_wuttke. These functions, once called, generate atmospheric profiles files as well as files containing the TOA solar input that we can use to reproduce the calculations of Rugheimer et al (2015) and the measurements of Wuttke et al (2006), as validation cases.

II) RESEARCH CASES: these are the calls that create the feedstock files used in our study. They include form_spectral_feedstock_youngmars, which is used to define a uniform TOA solar flux file for the young Mars. They also include calls to generate_profiles_cold_dry_mars and generate_profiles_volcanic_mars, which are defined in the file mars_atmosphere_models.py, to give the atmospheric profiles for atmospheres with user specified P_0(CO2), T_0, and specified SO2 and H2S loading levels (latter file only)

All file generation calls are at the end of the respective section.
"""
import numpy as np
import pdb
import matplotlib.pyplot as plt
import scipy.stats
from scipy import interpolate as interp
import prebioticearth_atmosphere_models as prebioticearth
import cookbook

bar2Ba=1.0e6 #1 bar in Ba
k=1.3806488e-16 #Boltzmann Constant in erg/K


############################################
###I. RUGHEIMER METADATA
############################################
def form_profiles_rugheimer_M8A_Ep0():
	"""
	Purpose of this code is to form spectra, mixing ratio files, and T/P profiles for the revised Epoch 0 (3.9 Ga Earth) models for M8A star (Rugheimer et al 2015).
	"""
	
	#####Zeroth: set value of constants, specify filenames
	filename='./Raw_Data/Rugheimer_Metadata/RugheimerModels/outchem_M8A_Ep0.dat'
	
	
	#####First, form the spectra for comparison.
	importeddata=np.genfromtxt(filename, skip_header=196, skip_footer=1277)
	#Remove the first wavelength bin which corresponds to Lyman Alpha and which does not have a bin width that fits with its neighbors.
	rugheimer_wav_centers=importeddata[1:,1]/10. #Convert wavelengths from Angstroms to nm
	rugheimer_toa_energies=importeddata[1:,2]*10. #TOA flux in erg/cm2/s/A, converted to erg/cm2/s/nm
	rugheimer_s=importeddata[1:,4] #ratio of 4piJ(surf)/I_0
	#rugheimer_s[19]=3.20481e-114 #one element of rugheimer_s is so small its negative exponent is 3 digits. Python has trouble with this and imports as a NaN. Here, we manually set its value.
	#print rugheimer_wav_centers
	#pdb.set_trace()

	###Form wavelength bins from Rugheimer wavelength centers
	rugheimer_wav_bin_leftedges=np.zeros(len(rugheimer_wav_centers))
	rugheimer_wav_bin_rightedges=np.zeros(len(rugheimer_wav_centers))

	#First ten FUV fluxes are 5 nm (50 A) bins (email from srugheimer@gmail.com, 3/12/2015) 
	rugheimer_wav_bin_leftedges[0:9]=rugheimer_wav_centers[0:9]-2.5
	rugheimer_wav_bin_rightedges[0:9]=rugheimer_wav_centers[0:9]+2.5

	#Remainder of FUV fluxes are taken from a file that sarah sent me (srugheimer@gmail.com, 3/12/2015)
	importeddata=np.genfromtxt('./Raw_Data/Rugheimer_Metadata/Active_M9_Teff2300_photo.pdat', skip_header=1, skip_footer=0)
	rugheimer_wav_bin_leftedges[9:]=importeddata[:,2]*0.1 #convert A to nm
	rugheimer_wav_bin_rightedges[9:]=importeddata[:,3]*0.1 #convert A to nm

	####Check that bins are correct:
	###print np.sum(rugheimer_wav_centers-0.5*(rugheimer_wav_bin_leftedges+rugheimer_wav_bin_rightedges)) #0 to within 1e-12 rounding error.

	###Compute bottom-of-atmosphere actinic flux, which is what is reported in Rugheimer+2015.
	rugheimer_ground_energies=rugheimer_toa_energies*rugheimer_s
	
	#Let's print out the results
	spectable=np.zeros([len(rugheimer_wav_bin_leftedges),4])
	
	spectable[:,0]=rugheimer_wav_bin_leftedges
	spectable[:,1]=rugheimer_wav_bin_rightedges
	spectable[:,2]=rugheimer_wav_centers
	spectable[:,3]=rugheimer_toa_energies
	
	header='Left Bin Edge (nm)	Right Bin Edge (nm)	Bin Center (nm)		Solar Flux at Earth (erg/s/nm/cm2)\n'

	f=open('./StellarInput/rugheimer2015_M8A_ep0_stellar_input.dat', 'w')
	f.write(header)
	np.savetxt(f, spectable, delimiter='		', fmt='%1.7e', newline='\n')
	f.close()

	######Let's print out the results: Literature Spectra
	#####spectable=np.zeros([len(rugheimer_wav_bin_leftedges),5])
	#####spectable[:,0]=rugheimer_wav_bin_leftedges
	#####spectable[:,1]=rugheimer_wav_bin_rightedges
	#####spectable[:,2]=rugheimer_wav_centers
	#####spectable[:,3]=rugheimer_toa_energies
	#####spectable[:,4]=rugheimer_ground_energies
	
	#####header='Left Bin Edge (nm)	Right Bin Edge (nm)	Bin Center (nm)		Solar Flux at Earth (erg/s/nm/cm2)		3.9 Ga BOA Intensity (erg/s/nm/cm2)\n'

	#####f=open('./LiteratureSpectra/rugheimer2015_M8A_ep0.dat', 'w')
	#####f.write(header)
	#####np.savetxt(f, spectable, delimiter='		', fmt='%1.7e', newline='\n')
	#####f.close()
	
	############################################################################################
	############################################################################################
	############################################################################################
	######Second, form the gas mixing ratio files
	#importeddata1=np.genfromtxt(filename, skip_header=685, skip_footer=873) #O2, O3, H2O
	##print importeddata1[:,0]
	##pdb.set_trace()
	#importeddata2=np.genfromtxt(filename, skip_header=743, skip_footer=817) #CH4, SO2
	##print importeddata2[:,0]
	##pdb.set_trace()
	#importeddata4=np.genfromtxt(filename, skip_header=864, skip_footer=704) #N2, CO2
	##print importeddata4[:,0]
	##pdb.set_trace()

	##Let's print out the results. We have established that the z values are the same, so can use a common block
	#printtable=np.zeros([np.shape(importeddata1)[0],9])
	#printtable[:,0]=importeddata1[:,0] #altitude in cm
	##N2 and CO2: We use the values from this block rather than block 1 because rugheimer et al force it to these values in their code, regardless of what the photochemistry code wants to do.
	#printtable[:,1]=importeddata4[:,2] #N2. 
	#printtable[:,2]=importeddata4[:,1] #CO2
	##The rest are normal
	#printtable[:,3]=importeddata1[:,3] #H2O
	#printtable[:,4]=importeddata2[:,2] #CH4
	#printtable[:,5]=importeddata2[:,9] #SO2
	#printtable[:,6]=importeddata1[:,2] #O2
	#printtable[:,7]=importeddata1[:,8] #O3
	##printtable[:,8]# H2S; left as zeros since not included in Rugheimer model
	
	##print np.sum(printtable[:,1:],1)
	
	#header0='Extracted from Rugheimer outchem_Earth_Ep0.dat\n'
	#header1='Z (cm)		N2	CO2	H2O	CH4	SO2	O2	O3	H2S \n'

	#f=open('./MolarConcentrations/rugheimer2015_M8A_ep0_molarconcentrations.dat', 'w')
	#f.write(header0)
	#f.write(header1)
	#np.savetxt(f, printtable, delimiter='	', fmt='%1.7e', newline='\n')
	#f.close()

	################################################################################################
	################################################################################################
	################################################################################################
	##########Third, form the aerosol mixing ratio files
	#####importeddata5=np.genfromtxt(filename, skip_header=804, skip_footer=760) #SO4 (sulfate) aerosol
	######print importeddata5[:,0]
	######pdb.set_trace()
	
	######Let's print out the results. We have established that the z values are the same, so can use a common block
	#####printtable=np.zeros([np.shape(importeddata5)[0],2])
	#####printtable[:,0]=importeddata5[:,0] #altitude in cm
	#####printtable[:,1]=importeddata5[:,1] #MixingRatio of sulfate aerosol
	#####printtable[:,1]=np.nan_to_num(printtable[:,1]) #there are a bunch of <e-99 values which are printed wrong in the metadata so numpy can't read them. These show as NaNs. np.nan_to_num converts these nans to 0s, which is right. WARNING: don't use this blindly when applying this function to other files, there might be "real" NaNs as well as these artefacts!
	######pdb.set_trace()
	
	#####header0='Extracted from Rugheimer outchem_Earth_Ep0.dat\n'
	#####header1='Z (cm)		SO4 aerosol \n'

	#####f=open('./MixingRatios/rugheimer2015_M8A_ep0_particulate_mixingratios.dat', 'w')
	#####f.write(header0)
	#####f.write(header1)
	#####np.savetxt(f, printtable, delimiter='	', fmt='%1.7e', newline='\n')
	#####f.close()

	############################################################################################
	############################################################################################
	############################################################################################
	######Third, form the T/P profiles
	
	##Extract temperature and pressure profile from climate model output
	##these are dry pressures
	#importeddata=np.genfromtxt(filename, skip_header=1474, skip_footer=105)
	#model_z=importeddata[:,0] #altitude in cm
	#model_t=importeddata[:,1] #temperature in K
	#model_n=importeddata[:,3] #number density in cm**-3.
	#model_p=importeddata[:,4] #pressure, in bar (based on text in draft manuscript sent to me by Sarah Rugheimer)

	##print model_z
	##pdb.set_trace()
	##Let's print out the results
	#printtable=np.zeros([len(model_z)+1,4])
	#printtable[1:,0]=model_z
	#printtable[1:,1]=model_t
	#printtable[1:,2]=model_n
	#printtable[1:,3]=model_p
	
	##Rugheimer data file does not explicitly include t, P, n at z=0 (Surface). Our code requires z=0 data. To reconcile, we include these data manually as follows:
	#printtable[0,0]=0. #z=0 case
	#printtable[0,3]=1. #In the paper, p=1.0 bar at surface is specified
	#printtable[0,1]=printtable[1,1]+0.5*(printtable[1,1]-printtable[2,1]) #From linear extrapolation from z=0.5 km and z=1.5 km points
	#printtable[0,2]= printtable[0,3]*bar2Ba/(k*printtable[0,1])#Compute number density self-consistently from temperature, pressure via Ideal Gas Law as is done elsewhere (n [cm**-3] = p [Barye]/(k*T [K])
	##pdb.set_trace()

	#header0='Extracted from Rugheimer outchem_Earth_Ep0.dat\n'
	#header1='Z (cm)	T (K)	DEN (cm**-3)	P (bar) \n'

	#f=open('./TPProfiles/rugheimer2015_M8A_ep0_tpprofile.dat', 'w')
	#f.write(header0)
	#f.write(header1)
	#np.savetxt(f, printtable, delimiter='		', fmt='%1.7e', newline='\n')
	#f.close()

############################################
###GENERAL YOUNG SUN FILE
#############################################
def form_spectral_feedstock_youngearth():
	"""
	Purpose of this code is to form the spectral feedstock file (TOA solar flux) to explore formally the dependence of UV surface radiance on various factors.
	"""
	#Define spectral bins.
	bin_left_edges=np.arange(100.,500.,1.)
	bin_right_edges=np.arange(101.,501.,1.)
	bin_centers=0.5*(bin_left_edges+bin_right_edges)
	
	#load solar spectrum at 3.9 Ga from Claire et al (2012) models, normalized to 1 AU. These are really TOA intensities. Multiply by mu_0 to get TOA fluxes. 
	importeddata=np.genfromtxt('./Raw_Data/Claire_Model/claire_youngsun_highres.dat', skip_header=1, skip_footer=0)
	claire_wav=importeddata[:,0] #nm, 0.01 nm resolution, 100-900 nm.
	claire_fluxes=importeddata[:,1] #units of erg/s/cm2/nm
	
	#rebin claire spectrum
	claire_fluxes_rebinned=cookbook.rebin_uneven(np.arange(99.995,900.005,0.01), np.arange(100.005, 900.015,0.01),claire_fluxes,bin_left_edges, bin_right_edges)   
	
	#Plot to make sure rebinning worked correctly
	fig, ax1=plt.subplots(1, figsize=(6,4))
	ax1.plot(claire_wav, claire_fluxes, marker='s', color='black', label='Claire Fluxes')
	ax1.plot(bin_centers, claire_fluxes_rebinned, marker='s', color='blue', label='Binned Claire Fluxes')	
	ax1.set_yscale('log')
	ax1.set_ylim([1.e-2, 1.e4])
	ax1.set_xlim([100.,500.])
	ax1.set_xlabel('nm')
	ax1.set_ylabel('erg/s/cm2/nm')
	ax1.legend(loc=0)
	plt.show()	
	
	#Let's print out the results
	spectable=np.zeros([len(bin_left_edges),4])
	spectable[:,0]=bin_left_edges
	spectable[:,1]=bin_right_edges
	spectable[:,2]=bin_centers
	spectable[:,3]=claire_fluxes_rebinned
	
	header='Left Bin Edge (nm)	Right Bin Edge (nm)	Bin Center (nm)		Top of Atm Flux (erg/s/nm/cm2)\n'

	f=open('./StellarInput/general_youngsun_earth_highres_widecoverage_spectral_input.dat', 'w')
	f.write(header)
	np.savetxt(f, spectable, delimiter='		', fmt='%1.7e', newline='\n')
	f.close()

############################################
###RUN
############################################
###Generate Rugheimer spectral feedstock for M8 star
form_profiles_rugheimer_M8A_Ep0()

#Generate spectral feedstock for young Sun
form_spectral_feedstock_youngearth()

#Generate T/P profiles and mixing ratios for generic prebiotic atmospheres
prebioticearth.generate_profiles_prebioticearth_exponential(1., 288., 'generalprebioticatm_exponential')
prebioticearth.generate_profiles_prebioticearth_exponential(1.e-6, 288., 'generalprebioticatm_exponential_thinatm')
prebioticearth.generate_profiles_prebioticearth_n2_co2_h2o_unsaturated(1., 288., 0.1,1., 'generalprebioticatm_dryadiabat_relH=1')

#form_spectral_feedstock_youngearth()