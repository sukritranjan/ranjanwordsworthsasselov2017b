# -*- coding: iso-8859-1 -*-
"""
This script holds the subfunctions used to define the surface albedo for the radiativetransfer.py code.
"""
import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy.integrate
from scipy import interpolate as interp

def get_surface_albedo(wav_left, wav_right, solarzenithangle,uniformalbedoflag, uniformalbedo, nonuniformalbedo, mode, toreturn):
	"""
	This function returns the albedo of the surface in different wavelength bins as a function of the composition and the solar zenith angle.
	Inputs:
	-wav_left, wav_right: left and right edges of wavelength bins (ascending) in nm
	-solarzenithangle: solar zenith angle in RADIANS
	-uniformalbedoflag: If adopting a uniform albedo, set this flag to 'uniformalbedo'
	-uniformalbedo: If adopting a uniform albedo, what will its value be?
	-nonuniformalbedo: If adopting a nonuniform albedo, what are the fractional ground covers? Variable is a vector, with ordering:  ocean, tundra, desert, old snow, new snow. Sum should be 1. 
	-mode: if set to 'plot' will plot the resulting albedo distribution. Useful to check the albedo function is behaving as desired.
	-toreturn: if set to 'diffuse', return diffuse albedo. if 'direct', return direct albedo.
	
	Outputs:
	-albedo_dif: diffuse albedo.
	OR
	-albedo_dir: direct albedo.
	
	Calling example:
	albedo_dif=get_surface_albedo(wav_left, wav_right, solarzenithangle,uniformalbedo,uniformalbedoflag, frac_ocean,frac_old_snow, frac_new_snow, frac_desert, frac_tundra, mode, 'diffuse')
	"""
	
	numbins=np.size(wav_left)
	albedo_dif=np.zeros(np.shape(wav_left))
	albedo_dir=np.zeros(np.shape(wav_left))
	
	if uniformalbedoflag=='uniformalbedo':
		albedo_dif=albedo_dif+uniformalbedo
		albedo_dir=albedo_dir+uniformalbedo
		
	else:
		wavscale=np.arange(wav_left[0], wav_right[-1]+0.1, step=0.1)
		
		#Force a normalization of the frac_X, so that the fractions add up to 1
		
		
		total=np.sum(nonuniformalbedo)
		frac_ocean=nonuniformalbedo[0]/total
		frac_tundra=nonuniformalbedo[1]/total
		frac_desert=nonuniformalbedo[2]/total
		frac_old_snow=nonuniformalbedo[3]/total
		frac_new_snow=nonuniformalbedo[4]/total
		
		composite_albedo_dir=frac_ocean*albedo_ocean(wavscale, solarzenithangle, 'direct')+frac_new_snow*albedo_snow(wavscale, solarzenithangle, 'new', 'direct')+frac_old_snow*albedo_snow(wavscale, solarzenithangle, 'old', 'direct')+frac_desert*albedo_desert(wavscale, solarzenithangle, 'direct')+frac_tundra*albedo_tundra(wavscale, solarzenithangle, 'direct')
		
		composite_albedo_dif=frac_ocean*albedo_ocean(wavscale, solarzenithangle, 'diffuse')+frac_new_snow*albedo_snow(wavscale, solarzenithangle, 'new', 'diffuse')+frac_old_snow*albedo_snow(wavscale, solarzenithangle, 'old', 'diffuse')+frac_desert*albedo_desert(wavscale, solarzenithangle, 'diffuse')+frac_tundra*albedo_tundra(wavscale, solarzenithangle, 'diffuse')
		
		composite_albedo_dir_func=interp.interp1d(wavscale, composite_albedo_dir, kind='linear')
		composite_albedo_dif_func=interp.interp1d(wavscale, composite_albedo_dif, kind='linear')
		
		for ind in range(0, numbins):
			albedo_dif[ind]=scipy.integrate.quad(composite_albedo_dif_func, wav_left[ind], wav_right[ind])[0]/(wav_right[ind]-wav_left[ind])
			albedo_dir[ind]=scipy.integrate.quad(composite_albedo_dir_func, wav_left[ind], wav_right[ind])[0]/(wav_right[ind]-wav_left[ind])
	
	if mode=='plot':
		figx, ax=plt.subplots(1, figsize=(6, 4))
		ax.plot(0.5*(wav_left+wav_right),albedo_dif, color='black', linestyle='-', marker='s', label=toreturn)
		ax.legend(loc=0)
		ax.set_yscale('linear')
		ax.set_ylabel('Albedo')
		ax.set_xlabel('Wavelength (nm)')	
		ax.set_ylim([0.,1.])
		#plt.show()
	
	
	if toreturn=='diffuse':
		return albedo_dif
	elif toreturn=='direct':
		return albedo_dir
	else:
		print 'Error: invalid value for toreturn'
		return np.zeros(np.shape(wav_left))


def albedo_ocean(wav, solarzenithangle, albtype):
	"""
	Returs surface albedo of pure (ice free, land free) ocean. Based on the formalism outlined in Briegleib et al (1986), and interpreted with Coakley (1986). Briegleb et al in turn take it from Payne (1972). Note that Briegleib et al explicity include mu-dependence.
	Input: 
	-wavelengths in nm.  Cannot be >4000 nm (4 um) or less than 1 nm. (It won't produce a bug, but the value returned will not be justified by the data.)
	-solar zenith angle in RADIANS
	-albtype: has value 'diffuse' or 'direct'. If 'diffuse', returns the diffuse albedo. If 'direct', returns the direct albedo. 
	"""
	mu=np.cos(solarzenithangle)

	alpha_dif=np.zeros(np.shape(wav))+0.06 #diffuse albedo
	alpha_dir=np.zeros(np.shape(wav))+0.026/(mu**1.7+0.065)+0.15*(mu-0.1)*(mu-0.5)*(mu-1.0) #direct albedo
	
	#The above is for albedo averaged from 280-2800 nm. We include spectral corrections here. 
	alpha_cor=np.zeros(np.shape(wav))
	inds1=np.where(wav<500.) #200-500 nm; use for everything shortward of 500 nm
	inds2=np.where((wav>=500.) & (wav<=700.)) #500-700 nm
	inds3=np.where(wav>700.) #700 nm - 4 um value; use for everything longward of 700 nm
	
	alpha_cor[inds1]=0.02
	alpha_cor[inds2]=-0.003
	alpha_cor[inds3]=-0.007
	
	alpha_dir=alpha_dir+alpha_cor
	alpha_dif=alpha_dif+alpha_cor
	
	alpha_dir[alpha_dir<0.]=0.
	alpha_dif[alpha_dif<0.]=0.
	alpha_dir[alpha_dir>1.]=1.
	alpha_dif[alpha_dif>1.]=1.
	
	if albtype=='direct':
		return alpha_dir
	elif albtype=='diffuse':
		return alpha_dif
	else:
		return np.zeros(np.shape(wav))
		print 'Error: invalid argument for albtype'

def albedo_snow(wav, solarzenithangle, isnew, albtype):
	"""
	This function returns the surface albedo of new-fallen snow. It is based on the formalism outlined in Briegleb and Ramanathan (1982), with interprative help from Coakley (2003).
	Inputs:
	-wavelengths in nm.  Cannot be >4000 nm (4 um) or less than 1 nm. (It won't produce a bug, but the value returned will not be justified by the data.)
	-Solar zenith angle, in RADIANS
	-isnew: controls the age of the snow. If 'new', new snow. If 'old', old snow
	-albtype: has value 'diffuse' or 'direct'. If 'diffuse', returns the diffuse albedo. If 'direct', returns the direct albedo. 
	"""
	mu=np.cos(solarzenithangle)
	
	alpha_1=np.zeros(np.shape(wav)) #This is an intermediate term used in their formalism to confer mu-dependence, from Dickinson et al (1981)
	
	#Table 1 of Briegleb and Ramanathan
	inds1=np.where(wav<500.) #200-500 nm; use for everything shortward of 500 nm
	inds2=np.where((wav>=500.) & (wav<=700.)) #500-700 nm
	inds3=np.where(wav>700.) #700 nm - 4 um value; use for everything longward of 700 nm
	
	if isnew=='new': #if new snow, use new snow values
		alpha_1[inds1]=alpha_1[inds1]+.95
		alpha_1[inds2]=alpha_1[inds2]+.95
		alpha_1[inds3]=alpha_1[inds3]+.65
	elif isnew=='old': #else use old snow values
		alpha_1[inds1]=alpha_1[inds1]+.76
		alpha_1[inds2]=alpha_1[inds2]+.76
		alpha_1[inds3]=alpha_1[inds3]+.325
	else:
		print 'Error: invalid value for isnew'
	
	#Add in mu-dependence using Dickinson et al (1981) methodology
	if mu>0.5:
		alpha_dir=alpha_1
	else:
		alpha_dir=alpha_1+(1.-alpha_1)*0.5*(3./(1.+4.*mu)-1.)
	
	alpha_dif=alpha_1 #diffuse flux as usual taken to correspond to direct flux at 60 degrees.

	alpha_dir[alpha_dir<0.]=0.
	alpha_dif[alpha_dif<0.]=0.
	alpha_dir[alpha_dir>1.]=1.
	alpha_dif[alpha_dif>1.]=1.

	if albtype=='direct':
		return alpha_dir
	elif albtype=='diffuse':
		return alpha_dif
	else:
		return np.zeros(np.shape(wav))
		print 'Error: invalid argument for albtype'


def albedo_desert(wav, solarzenithangle, albtype):
	"""
	Returns surface albedo of desert, following methodology of Coakley (2003), which in turn draws on Briegleb (1986)
	Input: 
	-wavelengths in nm. Cannot be >4000 nm (4 um) or less than 1 nm. (It won't produce a bug, but the value returned will not be justified by the data.)
	-solar zenith angle in RADIANS
	-albtype: has value 'diffuse' or 'direct'. If 'diffuse', returns the diffuse albedo. If 'direct', returns the direct albedo. 
	"""
	mu=np.cos(solarzenithangle)
	
	alpha_dif=np.zeros(np.shape(wav)) #Surface albedo at 60 degrees, taken to be the diffuse albedo
	
	inds1=np.where(wav<500.) #200-500 nm; use for everything shortward of 500 nm
	inds2=np.where((wav>=500.) & (wav<700.)) #500-700 nm
	inds3=np.where((wav>=700.) & (wav<=850.))#700-850 nm
	inds4=np.where(wav>850.) #840 nm - 4 um value; use for everything longward of 850 nm

	alpha_dif[inds1]=alpha_dif[inds1]+0.5*0.28+0.5*0.15
	alpha_dif[inds2]=alpha_dif[inds2]+0.5*0.42+0.5*0.25
	alpha_dif[inds3]=alpha_dif[inds3]+0.5*0.50+0.5*0.35
	alpha_dif[inds4]=alpha_dif[inds4]+0.5*0.50+0.5*0.40

	d=0.4 #semi-empirical fit
	alpha_dir=alpha_dif*(1.+d)/(1.+2.*d*mu)

	alpha_dir[alpha_dir<0.]=0.
	alpha_dif[alpha_dif<0.]=0.
	alpha_dir[alpha_dir>1.]=1.
	alpha_dif[alpha_dif>1.]=1.

	if albtype=='direct':
		return alpha_dir
	elif albtype=='diffuse':
		return alpha_dif
	else:
		return np.zeros(np.shape(wav))
		print 'Error: invalid argument for albtype'	

def albedo_tundra(wav, solarzenithangle, albtype):
	"""
	Returns surface albedo of tundra, following methodology of Coakley (2003), which in turn draws on Briegleb (1986)
	Input: 
	-wavelengths in nm. Cannot be >4000 nm (4 um) or less than 1 nm. (It won't produce a bug, but the value returned will not be justified by the data.)
	-solar zenith angle in RADIANS
	-albtype: has value 'diffuse' or 'direct'. If 'diffuse', returns the diffuse albedo. If 'direct', returns the direct albedo. 
	"""
	mu=np.cos(solarzenithangle)
	
	alpha_dif=np.zeros(np.shape(wav)) #Surface albedo at 60 degrees, taken to be the diffuse albedo
	
	inds1=np.where(wav<500.) #200-500 nm; use for everything shortward of 500 nm
	inds2=np.where((wav>=500.) & (wav<700.)) #500-700 nm
	inds3=np.where((wav>=700.) & (wav<=850.))#700-850 nm
	inds4=np.where(wav>850.) #840 nm - 4 um value; use for everything longward of 850 nm

	alpha_dif[inds1]=alpha_dif[inds1]+0.5*0.04+0.5*0.07
	alpha_dif[inds2]=alpha_dif[inds2]+0.5*0.10+0.5*0.13
	alpha_dif[inds3]=alpha_dif[inds3]+0.5*0.25+0.5*0.19
	alpha_dif[inds4]=alpha_dif[inds4]+0.5*0.25+0.5*0.28
	
	
	d=0.1 #semi-empirical fit
	alpha_dir=alpha_dif*(1.+d)/(1.+2.*d*mu)

	alpha_dir[alpha_dir<0.]=0.
	alpha_dif[alpha_dif<0.]=0.
	alpha_dir[alpha_dir>1.]=1.
	alpha_dif[alpha_dif>1.]=1.

	if albtype=='direct':
		return alpha_dir
	elif albtype=='diffuse':
		return alpha_dif
	else:
		return np.zeros(np.shape(wav))
		print 'Error: invalid argument for albtype'	
	