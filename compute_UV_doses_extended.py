# -*- coding: iso-8859-1 -*-
"""
This code is used to weigh the UV radiances we compute by biological action spectra.

When run, it iterates over all the action spectra in ./TwoStreamOutput/ and computes the relevate dose rates and puts them in ./DoseRates/. Each file contains only one line of data. While inefficient from an I/O perspective, this makes it easy to see at a glance the mapping between values and filenames. It should also make it easier for us to port code from Plot_Results to Plot_Discussion.
"""
########################
###Import useful libraries
########################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pdb
from matplotlib.pyplot import cm
from scipy import interpolate as interp
import scipy.integrate
import os
import os.path


########################
###Set physical constants
########################
hc=1.98645e-9 #value of h*c in erg*nm

def cm2inch(cm): #function to convert cm to inches; useful for complying with Astrobiology size guidelines
	return cm/2.54

########################
###Decide which bits of the calculation will be run
########################
plotactionspec=True #if true, plots the action spectra we are using.
plotactionspec_talk=False #if true, plots the action spectra we are using...but, optimized for a talk instead of a paper

calculate_all_doserates=True #if true, calculates the UV doserates associated with every file in ./TwoStreamOutput/

########################
###Helper functions: I/O
########################
def get_UV(filename):
	"""
	Input: filename (including path)
	Output: (wave_leftedges, wav_rightedges, surface radiance) in units of (nm, nm, photons/cm2/sec/nm)
	"""
	wav_leftedges, wav_rightedges, wav, toa_intensity, surface_flux,  SS,surface_intensity, surface_intensity_diffuse, surface_intensity_direct=np.genfromtxt(filename, skip_header=1, skip_footer=0, usecols=(0, 1, 2,3,4,5,6,7,8), unpack=True)
	
	surface_intensity_photons=surface_intensity*(wav/(hc))
	
	return wav_leftedges, wav_rightedges, surface_intensity_photons

########################
###Helper functions: UV Dosimeters
########################

def integrated_radiance(wav_left, wav_right, surf_int, leftlim, rightlim):
	"""
	Computes the surface radiance integrated from leftlim to rightlim. Does this by doing a trapezoid sum. NOTE: The method I have chosen works only so long as the limits line up with the bin edges!
	
	wav_left: left edge of wavelength bin, in nm
	wav_right: right edge of wavelength bin, in nm
	surf_int: total surface intensity (radiance, hemispherically-integrated) in photons/cm2/s/nm, in bin defined by wav_left and wav_right
	produceplots: if True, shows plots of what it is computing
	returnxy: if True, returns x,y for action spectrum. 
	"""
	
	allowed_inds=np.where((wav_left>=leftlim) & (wav_right<=rightlim))
	delta_wav=wav_right[allowed_inds]-wav_left[allowed_inds]
	
	surf_int_integrated=np.sum(surf_int[allowed_inds]*delta_wav) #integration converts from photons/cm2/s/nm to photons/cm2/s
	
	return surf_int_integrated
	

def tricyano_aqe_prodrate(wav_left, wav_right, surf_int, lambda0, produceplots, returnxy):
	"""
	Weights the input surface intensities by the action spectrum for the photoproduction of aquated electrons from Ritson+2012 and Patel+2015, i.e. irradiation of tricyano cuprate. The action spectrum is composed of the absorption spectrum multiplied by an assumed quantum yield function. We assume the QY function to be a step function, stepping from 0 at wavelengths longer than lambda0 to 0.06 at wavelengths shorter than lambda0. We choose  0.06 for the step function to match the estimate found by Horvath+1984; we note this value may be pH sensitive. Empirically, we know that lambda0>254 nm, but that's about it. 
	
	This process is an eustressor for abiogenesis.
	
	wav_left: left edge of wavelength bin, in nm
	wav_right: right edge of wavelength bin, in nm
	surf_int: total surface intensity (radiance, hemispherically-integrated) in photons/cm2/s/nm, in bin defined by wav_left and wav_right
	lambda0: value assume for lambda0.
	produceplots: if True, shows plots of what it is computing
	returnxy: if True, returns x,y for action spectrum. 
	"""
	####Step 1: reduce input spectrum to match bounds of available dataset.
	int_min=190.0 #This lower limit of integration is set by the limits of the cucn3 absorption dataset (left edge of bin)
	int_max=351.0 #This upper limit of integration is set by the limits of the cucn3 absorption dataset (right edge of bin)
	allowed_inds=np.where((wav_left>=int_min) & (wav_right<=int_max)) #indices that correspond to included data
	
	wav_left_cut=wav_left[allowed_inds]
	wav_right_cut=wav_right[allowed_inds]
	surf_int_cut=surf_int[allowed_inds]
	
	delta_wav=wav_right-wav_left #size of wavelength bins in nm
	
	####Step 2: form the action spectrum from the absorption spectrum and QY curve.
	#Import the tricyanocuprate absorption spectrum
	importeddata=np.genfromtxt('./Raw_Data/Magnani_Data/CuCN3_XC.dat', skip_header=2)
	cucn3_wav=importeddata[:,0] #wav in nm
	cucn3_molabs=importeddata[:,1] #molar absorptivities in  L/(mol*cm), decadic
	cucn3_molabs_func=interp.interp1d(cucn3_wav, cucn3_molabs, kind='linear') #functionalized form of cucn3 molar absorption
	#does not matter if you use decadic or natural logarithmic as constant factors normalize out anyway
	
	#Formulate the step-function quantum yield curve
	def qy_stepfunc(wav, lambda0): #step function, for the photoionization model
		"""Returns 1 for wav<=lambda0 and 0 for wav>lambda0"""
		qy=np.zeros(np.size(wav))# initialize all to zero
		inds=np.where(wav<=lambda0) #indices where the wavelength is below the threshold
		qy[inds]=qy[inds]+0.06 #increase the QE to 1 at the indices where the wavelength is below the threshold
		return qy
	
	#Integrate these quantities to match the input spectral resolution
	qy_dist=np.zeros(np.shape(wav_left_cut))#initialize variable to hold the QY integrated over the surface intensity wavelength bins
	cucn3_molabs_dist=np.zeros(np.shape(wav_left_cut))#initialize variable to hold the QY integrated over the surface intensity wavelength bins
	for ind in range(0, len(wav_left_cut)):
		leftedge=wav_left_cut[ind]
		rightedge=wav_right_cut[ind]
		
		cucn3_molabs_dist[ind]=scipy.integrate.quad(cucn3_molabs_func, leftedge, rightedge, epsabs=0, epsrel=1e-3,limit=500)[0]/(rightedge-leftedge)
		qy_dist[ind]=scipy.integrate.quad(qy_stepfunc, leftedge, rightedge, args=(lambda0), epsabs=0, epsrel=1e-3,limit=500)[0]/(rightedge-leftedge)
	
	action_spectrum=cucn3_molabs_dist*qy_dist
	
	#Normalize action spectrum to 1 at 195 (arbitrary)
	action_spectrum=action_spectrum*(1./(np.interp(190., 0.5*(wav_left_cut+wav_right_cut), action_spectrum)))
	
	#Extend action spectrum for <190, >351 nm
	action_spectrum_extended=np.zeros(np.shape(wav_left))
	action_spectrum_extended[allowed_inds]=action_spectrum
	action_spectrum_extended[0:np.min(allowed_inds)]+=action_spectrum[0]
	action_spectrum_extended[np.max(allowed_inds):-1]+=action_spectrum[-1]
	
	####Step 3: Compute action-spectrum weighted total intensity
	weighted_surface_intensity=surf_int*action_spectrum_extended
	
	total_weighted_radiance=np.sum(weighted_surface_intensity*delta_wav) #units: photons/cm2/s
	
	####Step 4 (Optional): Plot various components of action spectrum to show the multiplication
	if produceplots:
		legendfontsize=12
		axisfontsize=12
		
		
		##Plot ribonucleotide absorption and interpolation
		fig1, axarr=plt.subplots(3,2,sharex=True, figsize=(8., 10.5)) #specify figure size (width, height) in inches

		axarr[0,0].bar(wav_left, surf_int,width=delta_wav, color='black', alpha=0.5, log=True)
		axarr[0,0].set_ylim([1e10,1e16])
		axarr[0,0].legend(loc=2, prop={'size':legendfontsize})
		axarr[0,0].yaxis.grid(True)
		axarr[0,0].xaxis.grid(True)
		axarr[0,0].set_ylabel('Surface Radiance \n(photons cm$^{-2}$s$^{-1}$nm$^{-1}$)', fontsize=axisfontsize)
		#axarr[0,0].title.set_position([0.5, 1.11])
		#axarr[0,0].text(0.5, 1.1, r'a(i)', transform=axarr[0].transAxes, va='top')
		
		axarr[1,0].bar(wav_left, cucn3_molabs_dist,width=delta_wav, color='black', alpha=0.5, log=True)
		#axarr[1,0].set_ylim([-0.1, 1.1])
		axarr[1,0].legend(loc=6, prop={'size':legendfontsize})
		axarr[1,0].yaxis.grid(True)
		axarr[1,0].xaxis.grid(True)
		axarr[1,0].set_ylabel('CuCN3 Molar Absorptivity\n(M$^{-1}$cm$^{-1}$)', fontsize=axisfontsize)
		#axarr[1,0].text(0.5, 1.10, r'b(i)', fontsize=12, transform=axarr[1].transAxes, va='top')

		axarr[2,0].bar(wav_left, qy_dist,width=delta_wav, color='black', alpha=0.5)
		axarr[2,0].set_ylim([-0.01, 0.06])
		axarr[2,0].legend(loc=6, prop={'size':legendfontsize})
		axarr[2,0].yaxis.grid(True)
		axarr[2,0].xaxis.grid(True)
		axarr[2,0].set_ylabel('Quantum Efficiency \n(reductions absorption$^{-1}$)', fontsize=axisfontsize)
		#axarr[2,0].text(0.5, 1.10, r'c(i)', fontsize=12,transform=axarr[2].transAxes, va='top')

		axarr[0,1].bar(wav_left, action_spectrum,width=delta_wav, color='black', alpha=0.5)
		#axarr[0,1].set_ylim([-0.1, 1.1])
		axarr[0,1].legend(loc=6, prop={'size':legendfontsize})
		axarr[0,1].yaxis.grid(True)
		axarr[0,1].xaxis.grid(True)
		axarr[0,1].set_ylabel('Action Spectrum', fontsize=axisfontsize)
		#axarr[0,1].text(0.5, 1.10, r'b(i)', fontsize=12, transform=axarr[1].transAxes, va='top')
		
		axarr[1,1].bar(wav_left, weighted_surface_intensity,width=delta_wav, color='black', alpha=0.5)
		#axarr[1,1].set_ylim([-0.1, 1.1])
		axarr[1,1].legend(loc=6, prop={'size':legendfontsize})
		axarr[1,1].yaxis.grid(True)
		axarr[1,1].xaxis.grid(True)
		axarr[1,1].set_ylabel('Weighted Surface Radiance', fontsize=axisfontsize)
		#axarr[1,1].text(0.5, 1.10, r'b(i)', fontsize=12, transform=axarr[1].transAxes, va='top')
		#plt.savefig('/home/sranjan/Python/UV/Plots/ritson_assumed_qe_v3.pdf', orientation='portrait',papertype='letter', format='pdf')
		plt.show()	
	if returnxy:
		return 0.5*(wav_left+wav_right), action_spectrum_extended
	else:
		return total_weighted_radiance
	
def ump_glycosidic_photol(wav_left, wav_right, surf_int, lambda0, produceplots, returnxy):
	"""
	Weights the input surface intensities by the action spectrum for cleavage of the glycosidic bond in UMP (the U-RNA monomer), aka base release. We form this spectrum by convolving the pH=7.6 absorption spectrum for Uridine-3'-(2')-phosporic acid (i.e. uridylic acid, UMP) from Voet et al (1963) with an assumed QY curve. The QY curve is based on the work of Gurzadyan and Gorner (1994); they measure (wavelength, QY) for N-glycosidic bond cleavage in UMP in anoxic aqueous solution (Ar-suffused) to be (193 nm, 4.3e-3) and (254 nm, (2-3)e-5). Specifically, we assume that QY=4.3e-3 for lambda<=lambda_0 and QY=2.5e-5 for lambda>lambda_0. natural choices of lambda_0 are 194, 254, and 230 (first two: empirical limits. Last: end of pi-pi* absorption bad, Sinsheimer+1949 suggest it is onset of irreversible photolytic damage).
	
	This process is a stressor for abiogenesis.
	
	wav_left: left edge of wavelength bin, in nm
	wav_right: right edge of wavelength bin, in nm
	surf_int: total surface intensity (radiance, hemispherically-integrated) in photons/cm2/s/nm, in bin defined by wav_left and wav_right
	lambda0: value assume for lambda0.
	produceplots: if True, shows plots of what it is computing
	returnxy: if True, returns x,y for action spectrum. 
	"""
	####Step 1: reduce input spectrum to match bounds of available dataset (absorption).
	int_min=184.0 #This lower limit of integration is set by the limits of the cucn3 absorption dataset (left edge of bin)
	int_max=299.0 #This upper limit of integration is set by the limits of the cucn3 absorption dataset (right edge of bin)
	allowed_inds=np.where((wav_left>=int_min) & (wav_right<=int_max)) #indices that correspond to included data
	
	wav_left_cut=wav_left[allowed_inds]
	wav_right_cut=wav_right[allowed_inds]
	surf_int_cut=surf_int[allowed_inds]
	
	delta_wav=wav_right-wav_left #size of wavelength bins in nm
	
	####Step 2: form the action spectrum from the absorption spectrum and QY curve.
	#Import the UMP absorption spectrum from Voet et al 1963
	importeddata=np.genfromtxt('./Raw_Data/Voet_Data/ribouridine_pH_7.3_v2.txt', skip_header=0, delimiter=',')
	ump_wav=importeddata[:,0] #wav in nm
	ump_molabs=importeddata[:,1] #molar absorptivities\times 10^{3}, i.e. in units of 10^{-3} L/(mol*cm), decadic (I think -- unit scheme unclear in paper. Not important since normalized out)
	ump_molabs_func=interp.interp1d(ump_wav, ump_molabs, kind='linear') #functionalized form of molar absorption
	#does not matter if you use decadic or natural logarithmic as constant factors normalize out anyway
	
	#Formulate the step-function quantum yield curve
	def qy_stepfunc(wav, lambda0): #step function, for the photoionization model
		"""QY based on work of  Gurzadyan and Gorner 1994"""
		qy=np.zeros(np.size(wav))# initialize all to zero
		inds1=np.where(wav<=lambda0) #indices where the wavelength is below the threshold
		inds2=np.where(wav>lambda0) #indices where the wavelength is below the threshold
		qy[inds1]=qy[inds1]+4.3e-3 #High QY for lambda<=lambda0
		qy[inds2]=qy[inds2]+2.5e-5 #Low QY for lambda>lambda0
		return qy
	
	#Integrate these quantities to match the input spectral resolution
	qy_dist=np.zeros(np.shape(wav_left_cut))#initialize variable to hold the QY integrated over the surface intensity wavelength bins
	ump_molabs_dist=np.zeros(np.shape(wav_left_cut))#initialize variable to hold the UMP absorption integrated over the surface intensity wavelength bins
	for ind in range(0, len(wav_left_cut)):
		leftedge=wav_left_cut[ind]
		rightedge=wav_right_cut[ind]
		
		ump_molabs_dist[ind]=scipy.integrate.quad(ump_molabs_func, leftedge, rightedge, epsabs=0, epsrel=1e-3,limit=500)[0]/(rightedge-leftedge)
		qy_dist[ind]=scipy.integrate.quad(qy_stepfunc, leftedge, rightedge, args=(lambda0),epsabs=0, epsrel=1e-3,limit=500)[0]/(rightedge-leftedge)

	
	action_spectrum=ump_molabs_dist*qy_dist
	
	#Normalize action spectrum to 1 at 195 (arbitrary)
	action_spectrum=action_spectrum*(1./(np.interp(190., 0.5*(wav_left_cut+wav_right_cut), action_spectrum)))
	#Extend action spectrum for <184, >299 nm
	action_spectrum_extended=np.zeros(np.shape(wav_left))
	action_spectrum_extended[allowed_inds]=action_spectrum
	#pdb.set_trace()
	action_spectrum_extended[0:np.min(allowed_inds)]+=action_spectrum[0]
	action_spectrum_extended[np.max(allowed_inds):-1]+=action_spectrum[-1]
	
	####Step 3: Compute action-spectrum weighted total intensity
	weighted_surface_intensity=surf_int*action_spectrum_extended
	
	total_weighted_radiance=np.sum(weighted_surface_intensity*delta_wav) #units: photons/cm2/s
	
	####Step 4 (Optional): Plot various components of action spectrum to show the multiplication
	if produceplots:
		legendfontsize=12
		axisfontsize=12
		
		
		##Plot ribonucleotide absorption and interpolation
		fig1, axarr=plt.subplots(3,2,sharex=True, figsize=(8., 10.5)) #specify figure size (width, height) in inches

		axarr[0,0].bar(wav_left, surf_int,width=delta_wav, color='black', alpha=0.5, log=True)
		axarr[0,0].set_ylim([1e10,1e16])
		axarr[0,0].legend(loc=2, prop={'size':legendfontsize})
		axarr[0,0].yaxis.grid(True)
		axarr[0,0].xaxis.grid(True)
		axarr[0,0].set_ylabel('Surface Radiance \n(photons cm$^{-2}$s$^{-1}$nm$^{-1}$)', fontsize=axisfontsize)
		#axarr[0,0].title.set_position([0.5, 1.11])
		#axarr[0,0].text(0.5, 1.1, r'a(i)', transform=axarr[0].transAxes, va='top')
		
		axarr[1,0].bar(wav_left, ump_molabs_dist,width=delta_wav, color='black', alpha=0.5, log=False)
		#axarr[1,0].set_ylim([-0.1, 1.1])
		axarr[1,0].legend(loc=6, prop={'size':legendfontsize})
		axarr[1,0].yaxis.grid(True)
		axarr[1,0].xaxis.grid(True)
		axarr[1,0].set_ylabel('UMP Molar Absorptivity\n(M$^{-1}$cm$^{-1}$)', fontsize=axisfontsize)
		#axarr[1,0].text(0.5, 1.10, r'b(i)', fontsize=12, transform=axarr[1].transAxes, va='top')

		axarr[2,0].bar(wav_left, qy_dist,width=delta_wav, color='black', alpha=0.5, log=True)
		axarr[2,0].set_ylim([1e-5, 1e-2])
		axarr[2,0].legend(loc=6, prop={'size':legendfontsize})
		axarr[2,0].yaxis.grid(True)
		axarr[2,0].xaxis.grid(True)
		axarr[2,0].set_ylabel('Quantum Efficiency \n(reductions absorption$^{-1}$)', fontsize=axisfontsize)
		#axarr[2,0].text(0.5, 1.10, r'c(i)', fontsize=12,transform=axarr[2].transAxes, va='top')

		axarr[0,1].bar(wav_left, action_spectrum,width=delta_wav, color='black', alpha=0.5)
		#axarr[0,1].set_ylim([-0.1, 1.1])
		axarr[0,1].legend(loc=6, prop={'size':legendfontsize})
		axarr[0,1].yaxis.grid(True)
		axarr[0,1].xaxis.grid(True)
		axarr[0,1].set_ylabel('Action Spectrum', fontsize=axisfontsize)
		#axarr[0,1].text(0.5, 1.10, r'b(i)', fontsize=12, transform=axarr[1].transAxes, va='top')
		
		axarr[1,1].bar(wav_left, weighted_surface_intensity,width=delta_wav, color='black', alpha=0.5)
		#axarr[1,1].set_ylim([-0.1, 1.1])
		axarr[1,1].legend(loc=6, prop={'size':legendfontsize})
		axarr[1,1].yaxis.grid(True)
		axarr[1,1].xaxis.grid(True)
		axarr[1,1].set_ylabel('Weighted Surface Radiance', fontsize=axisfontsize)
		#axarr[1,1].text(0.5, 1.10, r'b(i)', fontsize=12, transform=axarr[1].transAxes, va='top')
		#plt.savefig('/home/sranjan/Python/UV/Plots/ritson_assumed_qe_v3.pdf', orientation='portrait',papertype='letter', format='pdf')
		plt.show()	
	if returnxy:
		return 0.5*(wav_left+wav_right), action_spectrum_extended
	else:
		return total_weighted_radiance


########################
###Plot UV Dosimeters
########################
if plotactionspec:
	#Set up wavelength scale
	wave_left=np.arange(100., 500.)
	wave_right=np.arange(101., 501.)
	wave_centers=0.5*(wave_left+wave_right)
	surf_int=np.ones(np.shape(wave_centers)) #for our purposes here, this is a thunk.

	#Extract action spectra
	wav_gly_193, actspec_gly_193=ump_glycosidic_photol(wave_left, wave_right, surf_int, 193., False, True)
	wav_gly_230, actspec_gly_230=ump_glycosidic_photol(wave_left, wave_right, surf_int, 230., False, True)
	wav_gly_254, actspec_gly_254=ump_glycosidic_photol(wave_left, wave_right, surf_int, 254., False, True)
	wav_aqe_254, actspec_aqe_254=tricyano_aqe_prodrate(wave_left, wave_right, surf_int, 254., False, True)
	wav_aqe_300, actspec_aqe_300=tricyano_aqe_prodrate(wave_left, wave_right, surf_int, 300., False, True)

	#####Plot action spectra
	#Initialize Figure
	fig, (ax1)=plt.subplots(1, figsize=(cm2inch(16.5),6), sharex=True)
	colorseq=iter(cm.rainbow(np.linspace(0,1,5)))
	#Plot Data
	ax1.plot(wav_gly_193,actspec_gly_193, linestyle='-',linewidth=2, color=next(colorseq), label=r'UMP Gly Bond Cleavage ($\lambda_0=193$)')
	ax1.plot(wav_gly_230,actspec_gly_230, linestyle='-',linewidth=2, color=next(colorseq), label=r'UMP Gly Bond Cleavage ($\lambda_0=230$)')
	ax1.plot(wav_gly_254,actspec_gly_254, linestyle='-',linewidth=2, color=next(colorseq), label=r'UMP Gly Bond Cleavage ($\lambda_0=254$)')
	ax1.plot(wav_aqe_254,actspec_aqe_254, linestyle='-',linewidth=2, color=next(colorseq), label=r'CuCN$_{3}$$^{2-}$ Photoionization ($\lambda_0=254$)')
	ax1.plot(wav_aqe_300,actspec_aqe_300, linestyle='--',linewidth=2, color=next(colorseq), label=r'CuCN$_{3}$$^{2-}$ Photoionization ($\lambda_0=300$)')

	#####Finalize and save figure
	ax1.set_title(r'Action Spectra')
	ax1.set_xlim([100.,400.])
	ax1.set_xlabel('nm')
	ax1.set_ylabel(r'Relative Sensitivity')
	ax1.set_yscale('log')
	ax1.set_ylim([1e-6, 1e2])	
	#ax1.legend(bbox_to_anchor=[0, 1.1, 1,1], loc=3, ncol=2, mode='expand', borderaxespad=0., fontsize=10)
	ax1.legend(loc='upper right', ncol=1, fontsize=10)
	plt.tight_layout(rect=(0,0,1,1))
	plt.savefig('./Plots/actionspectra.pdf', orientation='portrait',papertype='letter', format='pdf')


if plotactionspec_talk:
	#Set up wavelength scale
	wave_left=np.arange(100., 500.)
	wave_right=np.arange(101., 501.)
	wave_centers=0.5*(wave_left+wave_right)
	surf_int=np.ones(np.shape(wave_centers)) #for our purposes here, this is a thunk.

	#Extract action spectra
	wav_gly_193, actspec_gly_193=ump_glycosidic_photol(wave_left, wave_right, surf_int, 193., False, True)
	wav_gly_230, actspec_gly_230=ump_glycosidic_photol(wave_left, wave_right, surf_int, 230., False, True)
	wav_gly_254, actspec_gly_254=ump_glycosidic_photol(wave_left, wave_right, surf_int, 254., False, True)
	wav_aqe_254, actspec_aqe_254=tricyano_aqe_prodrate(wave_left, wave_right, surf_int, 254., False, True)
	wav_aqe_300, actspec_aqe_300=tricyano_aqe_prodrate(wave_left, wave_right, surf_int, 300., False, True)

	#####Plot action spectra
	#Initialize Figure
	fig, (ax1)=plt.subplots(1, figsize=(10,9), sharex=True)
	colorseq=iter(cm.rainbow(np.linspace(0,1,5)))
	#Plot Data
	ax1.plot(wav_gly_193,actspec_gly_193, linestyle='-',linewidth=3, color=next(colorseq), label=r'UMP-193')
	ax1.plot(wav_gly_230,actspec_gly_230, linestyle='-',linewidth=3, color=next(colorseq), label=r'UMP-230')
	ax1.plot(wav_gly_254,actspec_gly_254, linestyle='-',linewidth=3, color=next(colorseq), label=r'UMP-254')
	ax1.plot(wav_aqe_254,actspec_aqe_254, linestyle='-',linewidth=3, color=next(colorseq), label=r'CuCN3-254')
	ax1.plot(wav_aqe_300,actspec_aqe_300, linestyle='--',linewidth=3, color=next(colorseq), label=r'CuCN3-300')

	#####Finalize and save figure
	ax1.set_title(r'Action Spectra', fontsize=24)
	ax1.set_xlim([180.,360.])
	ax1.set_xlabel('nm',fontsize=24)
	ax1.set_ylabel(r'Relative Sensitivity', fontsize=24)
	ax1.set_yscale('log')
	ax1.set_ylim([1e-6, 1e2])	
	ax1.legend(bbox_to_anchor=[0, 1.1, 1,0.5], loc=3, ncol=2, mode='expand', borderaxespad=0., fontsize=24)
	#ax1.legend(loc='upper right', ncol=1, fontsize=16)
	ax1.xaxis.set_tick_params(labelsize=24)
	ax1.yaxis.set_tick_params(labelsize=24)
	plt.tight_layout(rect=(0,0,1,0.75))
	plt.savefig('./TalkFigs/actionspectra.pdf', orientation='portrait',papertype='letter', format='pdf')

########################
###Set "base" values to normalize the dosimeters by
########################

##Use the TOA flux in order to get a good, physically understandable denominator.
#wav_leftedges, wav_rightedges, wav, toa_intensity=np.genfromtxt('./Solar_Input/general_youngsun_mars_spectral_input.dat', skip_header=1, skip_footer=0, usecols=(0, 1,2, 3), unpack=True)
#toa_intensity_photons=toa_intensity*(wav/(hc))
#wav_leftedges, wav_rightedges, wav, toa_intensity=np.genfromtxt('./TwoStreamOutput/general_youngsun_mars_spectral_input.dat', skip_header=1, skip_footer=0, usecols=(0, 1,2, 3), unpack=True)
#toa_intensity_photons=toa_intensity*(wav/(hc))

##Compute base doses
#intrad100_165_base=integrated_radiance(wav_leftedges, wav_rightedges, toa_intensity_photons, 100, 165.) #This measures the flux vulnerable to activity
#intrad200_300_base=integrated_radiance(wav_leftedges, wav_rightedges, toa_intensity_photons, 200., 300.) #This is just an empirical gauge.
#umpgly_193_base=ump_glycosidic_photol(wav_leftedges, wav_rightedges, toa_intensity_photons, 193., False, False)		
#umpgly_230_base=ump_glycosidic_photol(wav_leftedges, wav_rightedges, toa_intensity_photons,230.,  False, False)
#umpgly_254_base=ump_glycosidic_photol(wav_leftedges, wav_rightedges, toa_intensity_photons, 254., False, False)
#tricyano254_base=tricyano_aqe_prodrate(wav_leftedges, wav_rightedges, toa_intensity_photons, 254., False, False)
#tricyano300_base=tricyano_aqe_prodrate(wav_leftedges, wav_rightedges, toa_intensity_photons, 300., False, False)


##Use the Rugheimer 3.9 Ga Earth case to normalize by
#wav_leftedges, wav_rightedges, rugheimer_surface_int=get_UV('./TwoStreamOutput/rugheimer2015_earth_ep0_reproduce.dat')

#Use the young Earth case to normalize by (should be same as Rugheimer 3.9 Ga case tbh). Can check by running quick comparison.
wav_leftedges, wav_rightedges, rugheimer_surface_int=get_UV('./TwoStreamOutput/generalprebioticatm_exponential_general_youngsun_earth_highres_widecoverage_spectral.dat')

#Compute base doses
intrad120_200_base=integrated_radiance(wav_leftedges, wav_rightedges, rugheimer_surface_int, 120., 200.) #This measures the flux vulnerable to activity
intrad200_300_base=integrated_radiance(wav_leftedges, wav_rightedges, rugheimer_surface_int, 200., 300.) #This is just an empirical gauge.
umpgly_193_base=ump_glycosidic_photol(wav_leftedges, wav_rightedges, rugheimer_surface_int, 193., False, False)		
umpgly_230_base=ump_glycosidic_photol(wav_leftedges, wav_rightedges, rugheimer_surface_int,230.,  False, False)
umpgly_254_base=ump_glycosidic_photol(wav_leftedges, wav_rightedges, rugheimer_surface_int, 254., False, False)
tricyano254_base=tricyano_aqe_prodrate(wav_leftedges, wav_rightedges, rugheimer_surface_int, 254., False, False)
tricyano300_base=tricyano_aqe_prodrate(wav_leftedges, wav_rightedges, rugheimer_surface_int, 300., False, False)

########################
###Run over all 
########################
if calculate_all_doserates:
	#First, get all files in directory
	file_list = [f for f in os.listdir('./TwoStreamOutput/') if os.path.isfile(os.path.join('./TwoStreamOutput/', f))] #gets list of all filenames from directory ./TwoStreamOutput/
	###file_list.remove('wuttke2006_reproduce_wuttke2006.dat') #Remove the Wuttke validation file...it does not have the wavelength coverage to be considered here.
	#Loop over all files in directory

	for filename in file_list:
		inputfile='./TwoStreamOutput/'+filename
		left, right, surface_int=get_UV(inputfile)
		
		intrad120_200=integrated_radiance(left, right, surface_int, 120., 200.) #This measures the flux vulnerable to activity
		intrad200_300=integrated_radiance(left, right, surface_int, 200., 300.) #This is just an empirical gauge.
		umpgly_193=ump_glycosidic_photol(left, right, surface_int, 193., False, False)		
		umpgly_230=ump_glycosidic_photol(left, right, surface_int,230.,  False, False)
		umpgly_254=ump_glycosidic_photol(left, right, surface_int, 254., False, False)
		tricyano254=tricyano_aqe_prodrate(left, right, surface_int, 254., False, False)
		tricyano300=tricyano_aqe_prodrate(left, right, surface_int, 300., False, False)
		
		line=np.array([intrad120_200/intrad120_200_base,intrad200_300/intrad200_300_base, umpgly_193/umpgly_193_base, umpgly_230/umpgly_230_base, umpgly_254/umpgly_254_base, tricyano254/tricyano254_base, tricyano300/tricyano300_base])

		#Save output
		outputfile='./DoseRates/dose_rates_'+filename
		f=open(outputfile,'w')
		f.write('All Dosimeters Normalized to Earth Case\n')
		np.savetxt(f, line[np.newaxis], delimiter='	', fmt='%s', newline='\n', header='Radiance (120-200 nm) & Radiance (200-300 nm) & UMP Gly Cleavage (lambda0=193nm) & UMP Gly Cleavage (lambda0=230nm) & UMP Gly Cleavage (lambda0=254nm) &  CuCN3 Photoionization (lambda0=254 nm) & CuCN3 Photoionization (lambda0=300 nm)\n')
		f.close()



########################
###Wrap Up
########################
plt.show()