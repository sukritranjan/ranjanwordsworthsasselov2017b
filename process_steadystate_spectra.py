# -*- coding: iso-8859-1 -*-
"""
The purpose of this code is to import & plot literature UV spectra of M-dwarfs, and process them into a form usable by our code.
"""


########################
###Import useful libraries
########################
import numpy as np
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pdb
from scipy import interpolate as interp
from astropy.io import fits

########################
###Define useful constants, all in CGS (via http://www.astro.wisc.edu/~dolan/constants.html)
########################

#Unit conversions
amu2g=1.66054e-24 #1 amu in g
bar2atm=0.9869 #1 bar in atm
Pa2bar=1.e-5 #1 Pascal in bar
bar2Pa=1.e5 #1 bar in Pascal
bar2barye=1.e6 #1 Bar in Barye (the cgs unit of pressure)
barye2bar=1.e-6 #1 Barye in Bar


#Fundamental constants
c=2.997924e10 #speed of light, cm/s
h=6.6260755e-27 #planck constant, erg/s
k=1.380658e-16 #boltzmann constant, erg/K

#Astronomical quantities
AU=1.496e13 #1 AU in cm
pc=3.086e18 #1 pc in cm
Lsun=3.9e33 #solar luminosity in erg/s

#masses
m_n2=28.01*amu2g #molecular mass of n2, converted to g
m_co2=44.01*amu2g #molecular mass of co2, converted to g


########################################################################
########################################################################
########################################################################
###READ IN DATA
########################################################################
########################################################################
########################################################################

########################
###Initialize dicts to hold all data
#########################
wav={} #in nm
flux={} #in erg/s/cm2/nm
starTeffs={} #in K
starplanetdists={} #in AU

########################
###Import quiescent Sun data
########################

#3.9 Ga Sun quiescent for comparison (from models of Claire+2012)
youngsun_wav, youngsun_flux=np.genfromtxt('./StellarInput/general_youngsun_earth_highres_widecoverage_spectral_input.dat', skip_header=1, skip_footer=0,usecols=(2,3), unpack=True) #nm, erg/s/cm2/nm
wav['youngsun']=youngsun_wav
flux['youngsun']=youngsun_flux

starTeffs['youngsun']=5300. #From 0.7*(5772K)^4=Teff,youngsun^4, assumes R_sun constant
starplanetdists['youngsun']=1. 

########################
###Import VPL data
########################
#Proxima Centauri [Meadows et al 2016]
starplanetdists['proxcen-vpl']=0.042 #Actual prox cen b distance is 0.485 AU, but here we use the Earth-equivalent distance we use elsewhere (Segura+2005) for consistency.
starTeffs['proxcen-vpl']=3042. #From file metadata

proxcen_wav_um, proxcen_flux_dist=np.genfromtxt('./Raw_Data/Mdwarf_Spectra/Steady-State/VPL/proxima_cen_1AU.txt', skip_header=37, skip_footer=0,usecols=(0,1), unpack=True) #A, Watt/m2/um; fluxes are at 1AU distance
wav['proxcen-vpl']=proxcen_wav_um*1.e3 #Convert um to nm
flux['proxcen-vpl']=proxcen_flux_dist*((1./starplanetdists['proxcen-vpl'])**2.)*1#Inverse Square law scaling from 1AU equivalent to actual semimajor axis of Prox Cen b, plus conversion from Watt/m2/um to erg/s/cm2/nm (it's a wash).

#AD Leo (quiescence) [Segura et al 2005] NOTE: May lack LyA peak (see comments in file)
starplanetdists['adleo']=0.16 #Segura et al 2005
starTeffs['adleo']=3400. #Segura et al 2005

adleo_wav_um, adleo_flux_dist=np.genfromtxt('./Raw_Data/Mdwarf_Spectra/Steady-State/VPL/adleo_dat.txt', skip_header=175, skip_footer=1,usecols=(0,1), unpack=True) #um, Watt/cm2/um; fluxes are at Earth-star distance
wav['adleo']=adleo_wav_um*1.e3 #convert um to nm
flux['adleo']=adleo_flux_dist*((4.9*pc)**2./(starplanetdists['adleo']*AU)**2.)*1.e4 #Inverse-square law scaling from star-Earth distance to HZ distance (Segura et al. 2010, 2005), plus conversion from Watt/cm2/um to erg/s/nm/cm2



#GJ644 [Segura et al 2005] NOTE: These data are specifically highlighted as unreliable by France et al (2016) due to geocoronal contamination, instrument background. Hence, they are of dubious value and should not be trusted. I am importing them for completeness and just to see, but BEWARE!!!
starplanetdists['gj644']=0.23 #Segura et al 2005
starTeffs['gj644']=3250. #Segura et al 2005

gj644_wav_um, gj644_flux_dist=np.genfromtxt('./Raw_Data/Mdwarf_Spectra/Steady-State/VPL/gj644_dat.txt', skip_header=98, skip_footer=0,usecols=(0,1), unpack=True) #um, Watt/cm2/um; fluxes are at Earth-star distance. To scale to Earth case, divide by 0.65 (since receives 65% of Earth's insolation, see Meadows et al. 2016)
wav['gj644']=gj644_wav_um*1.e3 #Convert um to nm
flux['gj644']=gj644_flux_dist*((6.2*pc)**2./(0.07*AU)**2.)*1.e4 #Inverse-square law scaling from star-Earth distance to HZ distance (Segura 2005), plus conversion from Watt/cm2/um to erg/s/nm/cm2 QUESTION: Did the ID error (see VPL descriptor page) affect the distance determination?


#HD22049 (K2V) [Segura et al 2003]. 
starplanetdists['hd22049']=0.53 #Segura et al 2003
starTeffs['hd22049']=5180. #Segura et al 2003 (???) It is from their model spectrum Teff...

hd22049_wav_a, hd22049_flux_dist=np.genfromtxt('./Raw_Data/Mdwarf_Spectra/Steady-State/VPL/hd22049um.txt', skip_header=1, skip_footer=0,usecols=(0,1), unpack=True) #A, erg/s/cm2/A; fluxes are at Earth-star distance.
wav['hd22049']=hd22049_wav_a*1.e-1 #Convert A to nm
flux['hd22049']=hd22049_flux_dist*((3.2*pc)**2./(starplanetdists['hd22049']*AU)**2.)*1.e1 #Inverse-square law scaling from star-Earth distance to HZ distance (Segura 2003), plus conversion from erg/s/cm2/A to erg/s/cm2/nm QUESTION: Did the ID error (see VPL descriptor page) affect the distance determination?
#ALSO EQUALS EPSILON ERIDANI!!! Overlap with MUSCLES!

########################
###Import MUSCLES Data
########################
muscles_starnames=np.array(['gj1214','gj876','gj436','gj581','gj667c','gj176','gj832','hd85512','hd40307','hd97658','v-eps-eri', 'gj551']) #NOTE: Prox Cen=GJ 551. The HD stars and the epsilon eridani are K-dwarfs (not M-dwarfs)
muscles_stardists=np.array([14.6, 4.7, 10.1, 6.2, 6.8, 9.3, 5.0, 11.2, 13.0, 21.1, 3.2, 1.3]) #distances to MUSCLES stars, from Loyd+2016, in pc. Prox Cen distance is 1.3 pc from  Earth
muscles_starTeffs=np.array([2935., 3062., 3281., 3295., 3327., 3416., 3816., 4305., 4783., 5156., 5162., 3042.]) #From Loyd et al 2016 Table 2. Prox Cen T_eff from VPL file metadata
muscles_albcorr=np.array([0.9, 0.9, 0.9, 0.9, 0.9,0.9, 0.9, 1., 1., 1., 1., 0.9]) #Following Segura et al. 2003, a factor to correct for different albedo of planet due to redshifted SED of M-dwarf. We take it to be 0.9 for all M-dwarf stars and 1 for the K-dwarfs, absent detailed modelling right now.

muscles_earthequivdists=np.zeros(len(muscles_starnames))


for ind in range(0, len(muscles_starnames)):
	muscles_starname=muscles_starnames[ind]
	muscles_stardist=muscles_stardists[ind]
	
	albedo_correction_factor=muscles_albcorr[ind]
	
	filename='./Raw_Data/Mdwarf_Spectra/Steady-State/MUSCLES/hlsp_muscles_multi_multi_'+muscles_starname+'_broadband_v20_adapt-const-res-sed.fits'
	
	spec= fits.getdata(filename,1)
	spec_wav=spec['WAVELENGTH'] #wavelength scale in A
	spec_flux=spec['FLUX'] #fluxes in erg/s/cm2/A

	#header=fits.getheader(filename,0)
	#boloflux=header['BOLOFLUX'] #units of ergs/s/cm2 ### This is not working b/c missing header keyword. For the time being, kludge as follows:
	boloflux_list=spec['FLUX']/spec['BOLOFLUX'] 
	boloflux=np.median(boloflux_list)
	
	luminosity=boloflux*4.*np.pi*(muscles_stardist*pc)**2. #total luminosity of star
	
	earthequivdist=np.sqrt((luminosity/Lsun)/albedo_correction_factor) #distance in AU, INCLUDING correction for M-dwarf redshift of radiation (Segura et al. 2003)
	
	#if muscles_starname=='gj551':
		#print luminosity/Lsun #0..00158, matches 0.00155 L_sun (Meadows et al. 2016) to 2 sig figs.
	
	muscles_earthequivdists[ind]=earthequivdist
	
	starTeffs[muscles_starname]=muscles_starTeffs[ind]
	starplanetdists[muscles_starname]=earthequivdist
	
	wav[muscles_starname]=spec_wav*1.e-1 #Convert from A to nm
	flux[muscles_starname]=spec_flux*((muscles_stardist*pc)**2./(earthequivdist*AU)**2.)*1.e1 #Convert from erg/s/cm2/A to erg/s/cm2/nm
	
	

#pdb.set_trace()

########################
###Import Kapetyn's Star Data
########################
kapteyn_wav_1, kapteyn_flux_1=np.genfromtxt('./Raw_Data/Mdwarf_Spectra/Steady-State/Guinan_2016/Kapteyn_G130M_1AU.txt', skip_header=0, skip_footer=0,usecols=(0,1), unpack=True) #A, erg/s/cm2/A; fluxes are at 1 AU.
kapteyn_wav_2, kapteyn_flux_2=np.genfromtxt('./Raw_Data/Mdwarf_Spectra/Steady-State/Guinan_2016/Kapteyn_G140L_1AU.txt', skip_header=0, skip_footer=0,usecols=(0,1), unpack=True) #A, erg/s/cm2/A; fluxes are at 1 AU.
kapteyn_wav_3, kapteyn_flux_3=np.genfromtxt('./Raw_Data/Mdwarf_Spectra/Steady-State/Guinan_2016/Kapteyn_COS_IUE_merged_1AU.txt', skip_header=0, skip_footer=0,usecols=(0,1), unpack=True) #A, erg/s/cm2/A; fluxes are at 1 AU.

#Assemble relevant metadata
starTeffs['kapteyn']=3570. #Table 1 of Guinan et al 2016, ultimate reference Anglada-Escude et al 2014
starplanetdists['kapteyn']=np.sqrt((0.012)/0.9) #distance in AU, INCLUDING correction for M-dwarf redshift of radiation (Segura et al. 2003). L/Lsun from Table 1 of Guinan et al 2016, ultimate reference Anglada-Escude et al 2014. NOTE: The Earth-equiv distance this gives is WITHIN the inner edge of the HZ ?Z (0.115 AU) listed in Anglada-Escude+2014...what gives?!




#Concatenate files and convert units

wav['kapteyn']=np.concatenate((kapteyn_wav_1,kapteyn_wav_2,kapteyn_wav_3))*0.1 #convert A to nm
flux['kapteyn']=np.concatenate((kapteyn_flux_1,kapteyn_flux_2,kapteyn_flux_3))*(1./starplanetdists['kapteyn'])**2.*10. #scale from 1 AU to earth-equiv distance and convert from erg/s/cm2/A to erg/s/cm2/nm

########################################################################
########################################################################
########################################################################
###Plot: Comparison
########################################################################
########################################################################
########################################################################

dataset_list=np.array(['proxcen-vpl', 'adleo', 'gj644', 'gj1214','gj876','gj436','gj581','gj667c','gj176','gj832', 'gj551', 'kapteyn'])

########################
###Plot
########################
fig, ax=plt.subplots(1, figsize=(8, 7), sharex=True, sharey=False)
markersizeval=5.
numdata=len(dataset_list)
colors=cm.rainbow(np.linspace(0,1,numdata))

ax.plot(wav['youngsun'], flux['youngsun'],linewidth=1, linestyle='--', color='black', label='Young Sun')
for ind in range(0, numdata):
	dataset=dataset_list[ind]
	ax.plot(wav[dataset], flux[dataset],linewidth=1, linestyle='-', color=colors[ind], label=dataset)



ax.set_yscale('log')
ax.set_ylim([1.e-5, 1.e3])
ax.set_xscale('linear')
ax.set_xlim([150., 300.])

ax.set_ylabel('Flux (erg/s/cm2/nm)')
ax.set_xlabel('Wavelength (nm)')

ax.legend(bbox_to_anchor=[0, 1.03, 1., .152],loc=3, ncol=3, mode='expand', borderaxespad=0., fontsize=10)
plt.tight_layout(rect=(0,0,1,0.85))


plt.savefig('./Plots/quiescent_plot.pdf', orientation='portrait',papertype='letter', format='pdf')


#########################################################################
#########################################################################
#########################################################################
####Plot & Print: Data
#########################################################################
#########################################################################
#########################################################################

########################
###Define subfunction to integrate data to coarser bin
########################

def integrate_data(abscissa, data, leftedges, rightedges):
	"""
	Takes: abscissa and corresponding data to be integrated, left edges of new bins, right edges of new bins (abscissa, leftedges and rightedges must have same units)
	
	Returns: data integrated to new bins
	
	Method: functionalizes data using stepwise linear interpolation, then integrates using gaussian quadrature technique
	"""
	data_func=interp.interp1d(abscissa, data, kind='linear')
	
	num_bin=len(leftedges)
	
	data_integrated=np.zeros(num_bin)
	
	for ind in range(0, num_bin):
		data_integrated[ind]=scipy.integrate.quad(data_func, leftedges[ind], rightedges[ind], epsabs=0., epsrel=1.e-3, limit=1000)[0]/(rightedges[ind]-leftedges[ind])
	
	return data_integrated

#########################
####Rebin to common scale
#########################

dataset_list=np.array(['adleo','proxcen-vpl',  'gj644', 'gj1214','gj876','gj436','gj581','gj667c','gj176','gj832', 'gj551'])
outfile_list=np.array(['vpl_adleo', 'vpl_proxcen', 'vpl_gj644', 'muscles_gj1214','muscles_gj876','muscles_gj436','muscles_gj581','muscles_gj667c','muscles_gj176','muscles_gj832', 'muscles_proxcen'])
numdata=len(dataset_list)

for ind in range(0, numdata):
	####################
	###Step 1: identify data
	####################
	dataset=dataset_list[ind]
	outfile_name=outfile_list[ind]
	
	starplanetdist=starplanetdists[dataset]
	wavelengths=wav[dataset]
	fluxes=flux[dataset]
	
	####################
	###Step 2: set rebinned wavelength scale
	####################
	
	wav_left=np.arange(120., 400., step=4.)
	wav_right=np.arange(121., 401., step=4.)
	wav_centers=0.5*(wav_left+wav_right)
	
	rebinned_fluxes=integrate_data(wavelengths, fluxes, wav_left, wav_right)
	
	####################
	###Plot rebinned data, so we can make sure everything is shipshape
	####################
	fig, ax1=plt.subplots(1, figsize=(8,5))
	ax1.plot(wavelengths, fluxes, marker='s', color='black', label='Original')
	ax1.plot(wav_centers, rebinned_fluxes, marker='s', color='blue', label='Rebinned')	
	ax1.set_yscale('log')
	ax1.set_ylim([1.e-4, 1.e4])
	ax1.set_xlim([120.,400.])
	ax1.set_xlabel('nm')
	ax1.set_ylabel('erg/s/cm2/nm')
	ax1.set_title(dataset)
	ax1.legend(loc=0)
	plt.show()		
	####################
	###Print rebinned data to file
	####################
	spectable=np.zeros([len(wav_left), 4])
	spectable[:,0]=wav_left
	spectable[:,1]=wav_right
	spectable[:,2]=wav_centers
	spectable[:,3]=rebinned_fluxes
	
	header='Left Bin Edge (nm)	Right Bin Edge (nm)	Bin Center (nm)		Stellar Flux (erg/s/nm/cm2) at '+str(starplanetdist)+' (AU)\n'
	
	f=open('./StellarInput/'+outfile_name+'_stellar_input.dat', 'w')
	f.write(header)
	np.savetxt(f, spectable, delimiter='		', fmt='%1.7e', newline='\n')
	f.close()
plt.show()


