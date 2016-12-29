# -*- coding: iso-8859-1 -*-
"""
The purpose of this code is to import & plot literature UV spectra of flares from  M-dwarfs, and process them into a form usable by our code.

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

AU=1.496e13 #1 AU in cm
pc=3.086e18 #1 pc in cm

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

#3.9 Ga Sun quiescent for comparison (from models of Claire+2012)
youngsun_wav, youngsun_flux=np.genfromtxt('./StellarInput/general_youngsun_earth_highres_widecoverage_spectral_input.dat', skip_header=1, skip_footer=0,usecols=(2,3), unpack=True) #nm, erg/s/cm2/nm

########################
###Import M-dwarf flare data
########################

#AD Leo (quiescence) [Segura et al 2005] NOTE: May lack LyA peak (see comments in file)
adleo_wav_um, adleo_flux_dist=np.genfromtxt('./Raw_Data/Mdwarf_Spectra/Steady-State/VPL/adleo_dat.txt', skip_header=175, skip_footer=1,usecols=(0,1), unpack=True) #um, Watt/cm2/um; fluxes are at Earth-star distance
adleo_quiesc_wav=adleo_wav_um*1.e3 #convert um to nm
adleo_quiesc_flux=adleo_flux_dist*((4.9*pc)**2./(0.16*AU)**2.)*1.e4 #Inverse-square law scaling from star-Earth distance to HZ distance (Segura et al. 2010, 2005), plus conversion from Watt/cm2/um to erg/s/nm/cm2

#From Segura+2010: Peak of flare is at 912 s
adleo_flare_912s_wav_a, adleo_flare_912s_flux_distant=np.genfromtxt('./Raw_Data/Mdwarf_Spectra/Flares/ADLeo_912s.dat', skip_header=2, skip_footer=0,usecols=(0,1), unpack=True) #A, erg/s/cm2/A; fluxes are at Earth-star distance
adleo_flare_912s_wav=adleo_flare_912s_wav_a*0.1 #convert A to nm
adleo_flare_912s_flux=adleo_flare_912s_flux_distant*10.*((4.9*pc)**2./(0.16*AU)**2.)#Convert erg/s/cm2/A to erg/s/cm2/nm, AND scale flux (inverse square law) from stellar distance of 4.9 pc (Rojas-Ayala et al 2012 via Venot et al 2016) to HZ distance of 0.16 AU (Segura et al 2010 via Segura et al 2005).



######
######
######

#AD Leo (quiescence) from Hawley et al 2003, compiled for us by Christopher M. Johns-Krull on 12/14
hawley2003_adleo_quiesc_wav_a, hawley2003_adleo_quiesc_flux_dist=np.genfromtxt('./Raw_Data/Mdwarf_Spectra/Flares/adleo_hawley2003_cmjk_quiet_ascii.txt', skip_header=0, skip_footer=0,usecols=(0,1), unpack=True) #um, Watt/cm2/um; fluxes are at Earth-star distance
hawley2003_adleo_quiesc_wav=hawley2003_adleo_quiesc_wav_a*0.1 #convert um to nm
hawley2003_adleo_quiesc_flux=hawley2003_adleo_quiesc_flux_dist*((4.9*pc)**2./(0.16*AU)**2.)*10. #Convert erg/s/cm2/A to erg/s/cm2/nm, AND scale flux (inverse square law) from stellar distance of 4.9 pc (Rojas-Ayala et al 2012 via Venot et al 2016) to HZ distance of 0.16 AU (Segura et al 2010 via Segura et al 2005).



########################################################################
########################################################################
########################################################################
###INTEGRATE DATA TO COARSER WAVELENGTH BIN
########################################################################
########################################################################
########################################################################

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
########################
###Integrate data to coarser wavelength bins, specifically 1.0 nm resolution
########################
wav_left=np.arange(120., 400., step=4.)
wav_right=np.arange(121., 401., step=4.)
wav=0.5*(wav_left+wav_right)

adleo_quiesc_flux_int=integrate_data(adleo_quiesc_wav, adleo_quiesc_flux, wav_left, wav_right)
adleo_flare_912s_flux_int=integrate_data(adleo_flare_912s_wav, adleo_flare_912s_flux, wav_left, wav_right)



hawley_wav_left=np.arange(120., 172., step=4.)
hawley_wav_right=np.arange(121., 173., step=4.)
hawley_wav=0.5*(hawley_wav_left+hawley_wav_right)
hawley2003_adleo_quiesc_flux_int=integrate_data(hawley2003_adleo_quiesc_wav, hawley2003_adleo_quiesc_flux, hawley_wav_left, hawley_wav_right)


########################################################################
########################################################################
########################################################################
###plot
########################################################################
########################################################################
########################################################################



##Compute attenuation. Only includes absorption effects (scattering extinction negligible at these levels). 
#attenuation_wav_pco2=np.zeros([len(wav_left), len(pco2)])

#attenuation_wav_pco2[=np.exp(-N_tot_pco2

#pdb.set_trace()
########################
###Plot
########################
fig, ax=plt.subplots(2, figsize=(8, 10), sharex=True, sharey=False)

markersizeval=5.
#colors=cm.rainbow(np.linspace(0,1,len(pco2)))

ax[0].plot(wav, adleo_quiesc_flux_int, linewidth=1, linestyle='-', color='red', label='Quiesc. (rebinned)')
ax[0].plot(wav, adleo_flare_912s_flux_int, linewidth=1, linestyle='-', color='blue', label='Flare (rebinned)')
ax[0].plot(hawley_wav, hawley2003_adleo_quiesc_flux_int, linewidth=1, linestyle='-', color='green', label='Quiesc. (CMJK, rebinned)')

#ax[0].plot(adleo_quiesc_wav, adleo_quiesc_flux, linewidth=1, linestyle='--', color='red', label='Quiesc.')
#ax[0].plot(adleo_flare_912s_wav, adleo_flare_912s_flux, linewidth=1, linestyle='--', color='black', label='Flare')
#ax[0].plot(hawley2003_adleo_quiesc_wav, hawley2003_adleo_quiesc_flux, linewidth=1, linestyle='--', color='green', label='Quiesc. (Hawley)')
ax[0].plot(youngsun_wav, youngsun_flux, linewidth=1, linestyle='--', color='black', label='3.9 Ga Sun')

flareratio=adleo_flare_912s_flux_int/adleo_quiesc_flux_int
ax[1].plot(wav, flareratio,linewidth=1, linestyle='-', color='black', label='Flare/Quiesc.')


ax[0].legend(loc=0, ncol=1, borderaxespad=0., fontsize=10)
ax[1].legend(loc='lower right', ncol=1, borderaxespad=0., fontsize=10)

ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[0].set_ylim([1.e-6, 1.e4])
ax[1].set_ylim([1.e-3, 1.e4])
ax[1].set_xscale('linear')
ax[1].set_xlim([120., 400.])

ax[0].set_ylabel('Flux (erg/s/cm2/nm)')
ax[1].set_ylabel('Flare/Quiesc')
ax[1].set_xlabel('Wavelength (nm)')

plt.savefig('./Plots/flares_plot_v2.pdf', orientation='portrait',papertype='letter', format='pdf')


####################
###Print rebinned data to file
####################
spectable=np.zeros([len(wav_left), 4])
spectable[:,0]=wav_left
spectable[:,1]=wav_right
spectable[:,2]=wav
spectable[:,3]=adleo_flare_912s_flux_int

header='Left Bin Edge (nm)	Right Bin Edge (nm)	Bin Center (nm)		Stellar Flux (erg/s/nm/cm2) at 0.16 AU\n'

f=open('./StellarInput/vpl_adleo_greatflare_stellar_input.dat', 'w')
f.write(header)
np.savetxt(f, spectable, delimiter='		', fmt='%1.7e', newline='\n')
f.close()

plt.show()


