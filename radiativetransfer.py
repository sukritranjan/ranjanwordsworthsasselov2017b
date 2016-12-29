# -*- coding: iso-8859-1 -*-
"""
Main radiative transfer code. This version includes 8 gases (N2, CO2, H2O, CH4, SO2, O2, O3, H2S) and 3 particulates (h2o ice, co2 ice, Mars dust)

This version is functionalized to facilitate easier replication of runs.

Note: if replicating Wuttke measurements, be sure to uncomment the different print command at the end of the file. Ditto WOUDC replication.
Note: if want to see figure with computed surface radiance, uncomment plt.show() at the end of the file.
"""

########################
###Import useful libraries & define constants
########################

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import pdb
import cross_sections_subfunctions as css
import twostream_toon_func as twostr
import radiativetransfer_albedo_subfunctions as ras
import radiativetransfer_subfunctions as rts

########################
###Constants
########################

bar2Ba=1.0e6 #1 bar in Ba
amu2g=1.66054e-24 #1 amu in g
nm2micron=1.e-3 #1 nm in micron

def uv_radtrans(z_upper_limit=64.e5, z_step=1.e5, inputatmofilelabel='volcanicmars_0.02bar_250K_10ppmso2_0h2s', outputfilelabel='_z=0_A=0_noTD_DS_co2cloudod=1000_z=20.5_reff=10_trimSO2', inputspectrafile='general_youngsun_mars_spectral_input.dat', 	h2oiceparamsfile='cloud_h2o_reff10_vareff0p1_lognormal.pickle', co2iceparamsfile='cloud_co2_reff10_vareff0p1_lognormal.pickle', dustparamsfile='dust_wolff_reff1p5_vareff0p5_lognormal.pickle', TDXC=False, DeltaScaling=True, SZA_deg=0., albedoflag='uniformalbedo',uniformalbedo=0., nonuniformalbedo=[0., 1., 0., 0., 0.], includedust=False, tau_d=0.,includeco2cloud=False, co2cloudlayerinds=np.array([43]), co2cloudlayerods=np.array([0.]),includeh2ocloud=False, h2ocloudlayerinds=np.array([60]), h2ocloudlayerods=np.array([0.])):

	"""
	Required inputs:
	#RT model layers
	---z_upper_limit: Upper edge of the atmospheric layers, in cm. The bottom edge of the layers is assumed to be at 0.
	---z_step: Thickness of the atmospheric layers, in cm.
	
	#I/0 file parameters
	---inputatmofilelabel: label that identifies the TP profile files and molar concentration files. Those files are assumed to have names inputatmofilelabel_tpprofile.dat or inputatmofilelabel_molarconcentrations.dat
	---outputfilelabel: label used to create output file name, which is equal to inputfilelabel+outputfilelabel
	---inputspectrafile: TOA Stellar Input
	---h2oiceparamsfile: pickle file giving the optical properties of the H2O ice particles. Typically integrated over size distribution, specified in filename
	---co2iceparamsfile: pickle file giving the optical properties of the CO2 ice particles. Typically integrated over size distribution, specified in filename
	---dustparamsfile:pickle file giving the optical properties of the dust ice particles. Typically integrated over size distribution, specified in filename
	
	#Particulate optical depth specification
	---tau_d: total dust optical depth (distributed according to exponential profile)
	
	#Flags
	---TDXC: Are temperature-dependent cross-sections of CO2, SO2 included? If True, yes.
	---DeltaScaling: Are we using delta scaling, to better approximate the forward-peaked scattering function of many atmospherically relevant particles? If True, yes.
	---includedust: should dust be included?
	---includeh2ocloud: should h2o clouds be included?
	---includeco2cloud: should co2clouds be included?
	
	#Albedo and SZA
	---SZA_deg: solar zenith angle, in degrees
	---albedoflag='uniformalbedo' #value of 'nonuniformalbedo' or 'uniformalbedo'. If desire uniform (spectrally flat) albedo: set flag to 'uniformalbedo'. If desire nonuniform albedo: set flag to 'nonuniformalbedo'.
	---uniformalbedo=0. #if adopting uniform albedo: what value?
	---nonuniformalbedo=[0., 1., 0., 0., 0.] #if adopting nonuniform albedo, fraction of ground that is: ocean, tundra, desert, old snow, new snow.
	
	Optional inputs (Default values):
	----
	----
	----
	----
	"""
	########################
	###User-set parameters
	########################
	N_gas_species=8 #how many gas species are in our model?
	N_particle_species=3 #number of species of particles we are treating, here h2o ice clouds and co2 ice clouds
	mu_1=1./np.sqrt(3.) #The Gaussian quadrature angle for the diffuse flux direction.

	########################
	###Code preliminaries
	########################
	#Get layers of the atmosphere
	z_lower, z_center, z_upper, N_layers=rts.get_z_layers(z_upper_limit, z_step) #define layers of atmosphere.
	z_edges=np.append(z_upper, z_lower[-1])
	z_widths=z_upper-z_lower
	
	#-----------------
	###Solar Zenith Angle
	solarzenithangle=SZA_deg*np.pi/180. #
	mu_0=np.cos(solarzenithangle)#Cosine of solar zenith angle.

	#-----------------
	###Clouds/particulates

	#To get co2 cloud optical depths, specify the optical depth at 500 nm.
	co2_cloud_optical_depths=np.zeros(np.shape(z_center))
	if includeco2cloud:
		co2iceparamsfile='./ParticulateOpticalParameters/'+co2iceparamsfile
		co2_cloud_optical_depths[co2cloudlayerinds]=co2cloudlayerods
		print 'CO2 cloud deck at (km):', z_center[co2cloudlayerinds]/1.e5
	print 'co2 cloud optical depth is (unscaled):', np.sum(co2_cloud_optical_depths) #check to make sure all shipshape

	#To get h2o cloud optical depths, specify the optical depth at 500 nm.
	h2o_cloud_optical_depths=np.zeros(np.shape(z_center))
	if includeh2ocloud:
		h2oiceparamsfile='./ParticulateOpticalParameters/'+h2oiceparamsfile
		h2o_cloud_optical_depths[h2ocloudlayerinds]=h2ocloudlayerods
		print 'H2O cloud deck at (km):', z_center[h2ocloudlayerinds]/1.e5		
	print 'h2o cloud optical depth is (unscaled):', np.sum(h2o_cloud_optical_depths) #check to make sure all shipshape

	##To get dust optical depths: assume exponential profile
	dust_optical_depths=np.zeros(np.shape(z_center))
	if includedust:
		dustparamsfile='./ParticulateOpticalParameters/'+dustparamsfile
		H_d=11.e5 #dust scale height, in cm
		prefactor=tau_d*(np.exp(z_center[-1]/H_d))*(1.-np.exp(-z_step/H_d))
		dust_optical_depths=prefactor*np.exp(-z_center/H_d)
	print 'dust optical depth is (unscaled):', np.sum(dust_optical_depths), '\n' #check to make sure all shipshape

	#-----------------
	#File I/O

	#T/P Profile.
	atmoprofilefile='./TPProfiles/'+inputatmofilelabel+'_tpprofile.dat' #T/P Profile File. File boundaries should match or exceed z_lower and z_upper
	#Molar concentrations
	gas_profilefile='./MolarConcentrations/'+inputatmofilelabel+'_molarconcentrations.dat' #Atmospheric composition file. File boundaries should match or exceed z_lower and z_upper
	
	filename=inputatmofilelabel+'_'+outputfilelabel #name of file to write output, plot to	
	writefilename='./TwoStreamOutput/'+filename+'.dat' #name of file to write output to. 




	########################
	###Load in TOA stellar input
	########################
	importeddata=np.genfromtxt('./StellarInput/'+inputspectrafile, skip_header=1, skip_footer=0)
	wav_leftedges=importeddata[:,0] #left edges of wavelength bins, nm
	wav_rightedges=importeddata[:,1] #right edges of wavelength bins, nm
	wav_centers=importeddata[:,2] #centers of wavelength bins, nm
	intensity_toa=importeddata[:,3] #top-of-atmosphere total intensity, erg/s/cm2/nm. Multiply by mu_0 to get TOA flux.

	N_wavelengths=len(wav_centers)
	wav_edges=np.append(wav_leftedges, wav_rightedges[-1])
	########################
	###Load albedos
	########################
	####Albedos
	albedo_dif_wav=ras.get_surface_albedo(wav_leftedges, wav_rightedges,solarzenithangle, albedoflag, uniformalbedo, nonuniformalbedo, 'noplot', 'diffuse')
	albedo_dir_wav=ras.get_surface_albedo(wav_leftedges, wav_rightedges,solarzenithangle, albedoflag, uniformalbedo, nonuniformalbedo, 'noplot', 'direct')


	########################
	###Get atmospheric layer column densities, temperatures 
	########################
	###extract integrated column densities for each layer in this atmosphere
	n_z, t_z, p_z, columndensity_z, t_c=rts.get_atmospheric_profile(z_lower, z_upper, atmoprofilefile)

	########################
	###Get gas composition of atmosphere
	########################
	###Molar concentrations
	#Set the molar concentrations of the gases. If set to a number, the gas is assumed to be well-mixed, i.e. have this concentration everywhere in the column. If set to a filename, the concentration profile is taken from that file.
	mr_n2=rts.get_molar_concentrations(z_center, gas_profilefile, 'n2')
	mr_co2=rts.get_molar_concentrations(z_center, gas_profilefile, 'co2')
	mr_h2o=rts.get_molar_concentrations(z_center, gas_profilefile, 'h2o')
	mr_ch4=rts.get_molar_concentrations(z_center, gas_profilefile, 'ch4')
	mr_so2=rts.get_molar_concentrations(z_center, gas_profilefile, 'so2')
	mr_o2=rts.get_molar_concentrations(z_center, gas_profilefile, 'o2')
	mr_o3=rts.get_molar_concentrations(z_center, gas_profilefile, 'o3')
	mr_h2s=rts.get_molar_concentrations(z_center, gas_profilefile, 'h2s')


	########################
	###Compute column densities of each gas species
	########################
	#Compute column densities
	colden_gas_species_z=np.zeros([N_gas_species,N_layers])
	colden_gas_species_z[0,:]=columndensity_z*mr_n2
	colden_gas_species_z[1,:]=columndensity_z*mr_co2
	colden_gas_species_z[2,:]=columndensity_z*mr_h2o
	colden_gas_species_z[3,:]=columndensity_z*mr_ch4
	colden_gas_species_z[4,:]=columndensity_z*mr_so2
	colden_gas_species_z[5,:]=columndensity_z*mr_o2
	colden_gas_species_z[6,:]=columndensity_z*mr_o3
	colden_gas_species_z[7,:]=columndensity_z*mr_h2s



	########################
	###Load gas absorption and scattering cross-sections.
	########################
	#Initialize variables to hold gas cross-sections
	gas_xc_tot_species_wav_z=np.zeros([N_gas_species,N_wavelengths, N_layers])
	gas_xc_scat_species_wav_z=np.zeros([N_gas_species,N_wavelengths, N_layers])

	#Load in band-averaged cross-sections.  
	#Function called returns xc in cm2/molecule in each band (total extinction, absorption, and rayleigh scattering). tot=abs+ray.
	#Only load cross-sections (expensive calculation) if some of the species exists
	if np.sum(mr_n2)>0:
		(gas_xc_tot_species_wav_z[0,:,:], gas_xc_scat_species_wav_z[0,:,:])=css.compute_band_cross_section(wav_leftedges, wav_rightedges,N_layers, 'n2')
	if np.sum(mr_h2o)>0:
		(gas_xc_tot_species_wav_z[2,:,:], gas_xc_scat_species_wav_z[2,:,:])=css.compute_band_cross_section(wav_leftedges, wav_rightedges,N_layers, 'h2o')
	if np.sum(mr_ch4)>0:
		(gas_xc_tot_species_wav_z[3,:,:], gas_xc_scat_species_wav_z[3,:,:])=css.compute_band_cross_section(wav_leftedges, wav_rightedges,N_layers, 'ch4')
	if np.sum(mr_o2)>0:
		(gas_xc_tot_species_wav_z[5,:,:], gas_xc_scat_species_wav_z[5,:,:])=css.compute_band_cross_section(wav_leftedges, wav_rightedges,N_layers, 'o2')
	if np.sum(mr_o3)>0:
		(gas_xc_tot_species_wav_z[6,:,:], gas_xc_scat_species_wav_z[6,:,:])=css.compute_band_cross_section(wav_leftedges, wav_rightedges,N_layers, 'o3')
	if np.sum(mr_h2s)>0:
		(gas_xc_tot_species_wav_z[7,:,:],  gas_xc_scat_species_wav_z[7,:,:])=css.compute_band_cross_section(wav_leftedges, wav_rightedges,N_layers, 'h2s')

	if np.sum(mr_co2)>0:
		if TDXC: #If we require temperature-dependence in our cross-sections (currently implemented for: CO2)
			(gas_xc_tot_species_wav_z[1,:,:], gas_xc_scat_species_wav_z[1,:,:])=css.compute_band_cross_section_td(wav_leftedges, wav_rightedges, t_z, 'co2')
		else:
			(gas_xc_tot_species_wav_z[1,:,:], gas_xc_scat_species_wav_z[1,:,:])=css.compute_band_cross_section(wav_leftedges, wav_rightedges,N_layers, 'co2')


	if np.sum(mr_so2)>0:
		if TDXC: #If we require temperature-dependence in our cross-sections (currently implemented for: CO2)
			(gas_xc_tot_species_wav_z[4,:,:], gas_xc_scat_species_wav_z[4,:,:])=css.compute_band_cross_section_td(wav_leftedges, wav_rightedges, t_z, 'so2')
		else:
			(gas_xc_tot_species_wav_z[4,:,:], gas_xc_scat_species_wav_z[4,:,:])=css.compute_band_cross_section(wav_leftedges, wav_rightedges,N_layers, 'so2')

	#Note that in some cases, the Rayleigh scattering cross-section exceeds the extinction cross-section measured in laboratory studies. This is fixed by setting the extinction cross-section to the Rayleigh value in this case.


	########Try Rugheimer cross-sections and scattering instead
	######(gas_xc_tot_species_wav_z[0,:], gas_xc_tot_species_wav_z[0,:])=css.get_rugheimer_xc(wav_leftedges, wav_rightedges, N_layers,'n2',0.78, 0.)
	#####(gas_xc_tot_species_wav_z[0,:,:], gas_xc_scat_species_wav_z[0,:,:])=css.get_rugheimer_xc(wav_leftedges, wav_rightedges, N_layers,'n2',0.89, 0.1)
	#####(gas_xc_tot_species_wav_z[1,:,:], gas_xc_scat_species_wav_z[1,:,:])=css.get_rugheimer_xc(wav_leftedges, wav_rightedges, N_layers,'co2',0,0)
	#####(gas_xc_tot_species_wav_z[2,:,:], gas_xc_scat_species_wav_z[2,:,:])=css.get_rugheimer_xc(wav_leftedges, wav_rightedges, N_layers,'h2o',0,0)
	#####(gas_xc_tot_species_wav_z[4,:,:], gas_xc_scat_species_wav_z[4,:,:])=css.get_rugheimer_xc(wav_leftedges, wav_rightedges, N_layers,'so2',0,0)
	#####(gas_xc_tot_species_wav_z[5,:,:], gas_xc_scat_species_wav_z[5,:,:])=css.get_rugheimer_xc(wav_leftedges, wav_rightedges, N_layers,'o2',0,0)
	#####(gas_xc_tot_species_wav_z[6,:,:], gas_xc_scat_species_wav_z[6,:,:])=css.get_rugheimer_xc(wav_leftedges, wav_rightedges, N_layers,'o3',0,0)
	######Rugheimer has no CH4 or H2S so we keep our own, but get rid of the scattering formalism 
	#####gas_xc_scat_species_wav_z[3,:,:]=gas_xc_scat_species_wav_z[3,:]*0.0
	#####gas_xc_scat_species_wav_z[7,:,:]=gas_xc_scat_species_wav_z[7,:]*0.0

	########################
	###Get inventories of particles per-layer. Only run if we are including effect of that layer.
	########################
	colden_particles_species_z=np.zeros([N_particle_species, N_layers])
	if includeh2ocloud:
		colden_particles_species_z[0,:]=rts.get_particulate_columndensities(h2o_cloud_optical_depths, h2oiceparamsfile) #water clouds, cm**-2
	if includeco2cloud:
		colden_particles_species_z[1,:]=rts.get_particulate_columndensities(co2_cloud_optical_depths, co2iceparamsfile) #co2 clouds, cm**-2
	if includedust:
		colden_particles_species_z[2,:]=rts.get_particulate_columndensities(dust_optical_depths, dustparamsfile) #dust, cm**-2

	########################
	###Load particulate optical parameters
	########################
	partoptparam_species_wav_z_params=np.zeros([N_particle_species,N_wavelengths,N_layers, 3]) #parameters: sigma, w_0, g

	if np.sum(colden_particles_species_z[0,:])>0: #only run H2O optical parameter code if have H2O ice crystals:
		partoptparam_species_wav_z_params[0,:,:,0], partoptparam_species_wav_z_params[0,:,:,1], partoptparam_species_wav_z_params[0,:,:,2]=css.compute_cloud_params(wav_leftedges,wav_rightedges, N_layers, h2oiceparamsfile)

	if np.sum(colden_particles_species_z[1,:])>0: #only run CO2 optical parameter code if have CO2 ice crystals:
		partoptparam_species_wav_z_params[1,:,:,0], partoptparam_species_wav_z_params[1,:,:,1], partoptparam_species_wav_z_params[1,:,:,2]=css.compute_cloud_params(wav_leftedges,wav_rightedges, N_layers, co2iceparamsfile)

	if np.sum(colden_particles_species_z[2,:])>0: #only run dust optical parameter code if have dust:
		partoptparam_species_wav_z_params[2,:,:,0], partoptparam_species_wav_z_params[2,:,:,1], partoptparam_species_wav_z_params[2,:,:,2]=css.compute_cloud_params(wav_leftedges,wav_rightedges, N_layers, dustparamsfile)

	#1st param is cross-section (cm**2/particle), 2nd param is w_0, 3rd param is g.

	########################
	###Compute atmospheric optical parameters required for two-stream code: tau_n, tau_c, w0, and g in each layer as a function of wavelength.
	########################

	#Call subfunction to extract composite values
	tau_n_tot_z_wav, w_0_z_wav, g_z_wav,tau_n_gas_tot_z_wav, tau_n_part_tot_z_wav=rts.compute_optical_parameters(colden_gas_species_z,gas_xc_tot_species_wav_z, gas_xc_scat_species_wav_z, colden_particles_species_z, partoptparam_species_wav_z_params, DeltaScaling)
	#Reminder: Toon et al define tau=0 at the TOA, and it is computed along the zenith direction (so solar zenith angle is accounted for separately in the code).


	########################
	###compute the flux via the two-stream approximation
	########################
	F_plus_tau0=np.zeros(np.shape(tau_n_tot_z_wav)) #F_plus evaluated at tau=0 for every layer n
	F_plus_taumax=np.zeros(np.shape(tau_n_tot_z_wav))#F_plus evaluated at tau=tau_n[n] for every layer n
	F_minus_tau0=np.zeros(np.shape(tau_n_tot_z_wav))#F_minus evaluated at tau=0 for every layer n
	F_minus_taumax=np.zeros(np.shape(tau_n_tot_z_wav))#F_minus evaluated at tau=tau_n[n] for every layer n

	F_net=np.zeros(np.shape(tau_n_tot_z_wav))#Net flux at the BASE of layer n. 
	AMEAN=np.zeros(np.shape(tau_n_tot_z_wav))#AMEAN, 4*pi*mean intensity at the base of layer n. 
	SS=np.zeros(np.shape(intensity_toa)) #This quantity is the SS quantity from twostr.f.
	surface_intensity=np.zeros(np.shape(intensity_toa)) #an  estimate of the total amount of intensity received by a point at the surface of the planet. It is equal to the direct intensity plus F_[surface]/mu_1, i.e. the downward diffuse intensity at the surface

	#Core loop over wavelength:
	for ind in range(0,N_wavelengths):
		wavelength=wav_centers[ind] #width doesn't matter as wav primarily matters for BB which varies in smooth way.
		solar_input=intensity_toa[ind]/np.pi #this converts the TOA flux to the F_s in Toon et al. Recall pi*F_s=solar flux (really solar intensity) in that formalism.
		w_0=w_0_z_wav[:,ind]
		
		g=g_z_wav[:,ind]
		tau_n=tau_n_tot_z_wav[:,ind]
		albedo_dif=albedo_dif_wav[ind]
		albedo_dir=albedo_dir_wav[ind]
		F_plus_tau0[:,ind], F_plus_taumax[:,ind], F_minus_tau0[:, ind], F_minus_taumax[:,ind], F_net[:,ind], AMEAN[:,ind], surface_intensity[ind]=twostr.twostr_func(wavelength, solar_input, solarzenithangle, albedo_dif, albedo_dir, w_0, g, tau_n)
		
		AMEAN[np.abs(AMEAN)<1.e-100]=0 #if AMEAN is a very small number, it is really 0 (eliminates errors in the square root for SS)
		SS[ind]=np.sqrt(AMEAN[-1,ind]*AMEAN[-2,ind])


	########################
	###Compute the direct fluxes throughout the atmosphere, and the surface flux.
	########################
	tau_c_tot_z_wav=np.zeros([N_layers+1, N_wavelengths])
	for ind in range(0, N_layers):
		tau_c_tot_z_wav[ind+1,:]=tau_c_tot_z_wav[ind,:]+tau_n_tot_z_wav[ind, :]

	direct_flux_z_wav=mu_0*intensity_toa*np.exp(-tau_c_tot_z_wav/mu_0) #Direct flux at the boundary of each layer. First layer=TOA, last layer=surface. See: Toon et al eqn 50, and recall: flux_toa=F_s*np.pi

	surface_direct_flux=direct_flux_z_wav[-1,:] #Get the direct flux at the surface
	surface_diffuse_flux=F_minus_taumax[-1,:] #Get downwelling diffuse flux at the bottom layer, i.e. the surface

	surface_flux=surface_diffuse_flux+surface_direct_flux

	########################
	###Compute the surface intensity at the base of the atmosphere, and compare to what is reported by the code.
	########################
	direct_intensity_z_wav=intensity_toa*np.exp(-tau_c_tot_z_wav/mu_0) #Direct intensity at the boundary of each layer, with first layer=TOA and last layer=BOA. 

	surface_direct_intensity=direct_intensity_z_wav[-1,:] #direct intensity at base of atmosphere

	surface_diffuse_intensity=F_minus_taumax[-1,:]/mu_1 #diffuse intensity at base of atmosphere.

	surface_intensity_2=surface_direct_intensity+surface_diffuse_intensity

	###Check for consistency:
	surf_int_diff=(surface_intensity-surface_intensity_2)/surface_intensity
	print 'Check: fractional agreement of two ways of calculating surface radiance', np.nanmax(np.abs(surf_int_diff)), '\n' #need nanmax to elimate divide by 0 errors

	########################
	###Check energy conservation
	########################
	incoming_flux_tot=np.sum(mu_0*intensity_toa)

	outgoing_flux_tot=np.sum(F_plus_tau0[0,:])

	if outgoing_flux_tot <= incoming_flux_tot:
		print 'Outgoing Flux<= Incoming Flux: Consistent with Energy Conservation', '\n'
	if outgoing_flux_tot > incoming_flux_tot:
		print 'Outgoing Flux > Incoming Flux: Energy Conservation Violated DANGER DANGER DANGER', '\n','DANGER DANGER DANGER'

	########################
	###Print some more diagnostics (useful check)
	########################


	print 'Total gas column density is (cm-2):', np.sum(columndensity_z)
	print 'N2 column density is (cm-2):', np.sum(colden_gas_species_z[0,:])
	print 'CO2 column density is (cm-2):', np.sum(colden_gas_species_z[1,:])
	print 'H2O column density is (cm-2):', np.sum(colden_gas_species_z[2,:])
	print 'CH4 column density is (cm-2):', np.sum(colden_gas_species_z[3,:])
	print 'SO2 column density is (cm-2):', np.sum(colden_gas_species_z[4,:])
	print 'O2 column density is (cm-2):', np.sum(colden_gas_species_z[5,:])
	print 'O3 column density is (cm-2):', np.sum(colden_gas_species_z[6,:])
	print 'H2S column density is (cm-2):', np.sum(colden_gas_species_z[7,:])
	print '\n'
	print 'Total particulate column density is (cm-2):', np.sum(colden_particles_species_z)
	print 'H2O ice cloud column density is (cm-2):', np.sum(colden_particles_species_z[0,:])
	print 'CO2 ice cloud column density is (cm-2):', np.sum(colden_particles_species_z[1,:])
	print 'Dust column density is (cm-2):', np.sum(colden_particles_species_z[2,:])
	print '\n'

	#inds=np.where(surface_intensity/intensity_toa<0.01)
	#indmax=np.max(inds) #here I assume monotonically decreasing
	##print 'index at which intensity first suppressed to 0.01 x incident is:', indmax
	#print 'wavelength at which intensity first suppressed 0.01 x incident is (nm):', wav_leftedges[indmax]
	#print '\n'


	#############################
	########Plot results
	#############################

	fig, (ax1)=plt.subplots(1, figsize=(8,6), sharex=True)
	ax1.plot(wav_centers, intensity_toa, marker='s', color='black', label='TOA Intensity')
	######ax1.plot(wav_centers, intensity_toa*np.exp(-np.sum(colden_gas_species_z[4,:])*gas_xc_tot_species_wav_z[4,:,-1]/mu_0), marker='s', color='green', label='SO2 Beers Law')
	ax1.plot(wav_centers, surface_intensity , marker='s', color='orange', label='Surface Intensity (This Model)')
	ax1.set_yscale('log')
	ax1.set_ylim([1.e-2, 1.e4])
	ax1.set_xlabel('nm')
	ax1.set_ylabel('erg/s/cm2/nm')
	ax1.legend(loc=0)


	#-----------------------------------------------
	tau_tot_wav=tau_c_tot_z_wav[-1,:]
	w_0_mean_wav=np.zeros(np.shape(tau_tot_wav))
	g_mean_wav=np.zeros(np.shape(tau_tot_wav))

	tau_n_gas_tot_wav=np.sum(tau_n_gas_tot_z_wav,axis=0)
	tau_n_part_tot_wav=np.sum(tau_n_part_tot_z_wav,axis=0)

	for ind in range(0, N_wavelengths):
		w_0_mean_wav[ind]=np.sum(w_0_z_wav[:,ind]*tau_n_tot_z_wav[:,ind])/np.sum(tau_n_tot_z_wav[:,ind])
		g_mean_wav[ind]=np.sum(g_z_wav[:,ind]*tau_n_tot_z_wav[:,ind])/np.sum(tau_n_tot_z_wav[:,ind])

	fig2, (ax1, ax2, ax3)=plt.subplots(3, figsize=(8,10), sharex=True)

	ax1.plot(wav_centers, tau_tot_wav, color='black', linestyle='-', label='total (gas+particle)')
	ax1.plot(wav_centers, tau_n_gas_tot_wav, color='red', linestyle='--', label='gas')
	ax1.plot(wav_centers, tau_n_part_tot_wav, color='blue', linestyle='--', label='particle')
	ax1.legend(loc=0)
	ax1.set_yscale('log')
	ax1.set_ylabel(r'$\tau$ (total)')

	ax2.plot(wav_centers, w_0_mean_wav)
	ax2.set_yscale('linear')
	ax2.set_ylim([0.,1.])
	ax2.set_ylabel(r'$w_0$ (mean)')

	ax3.plot(wav_centers, g_mean_wav)
	ax3.set_xlabel('nm')
	ax3.set_yscale('linear')
	ax3.set_ylim([0.,1.])
	ax3.set_ylabel(r'$g$ (mean)')



	###############################
	#######Print spectra
	############################
	toprint=np.zeros([np.size(wav_centers), 9])
	toprint[:,0]=wav_leftedges #left edge of wavelength bin (nm)
	toprint[:,1]=wav_rightedges #right edge of wavelength bin (nm)
	toprint[:,2]=wav_centers #center of wavelength bin (nm)
	toprint[:,3]=intensity_toa #intensity incident at top of atmosphere (erg/s/cm2/nm)
	toprint[:,4]=surface_flux #flux incident at bottom of atmosphere (erg/s/cm2/nm)
	toprint[:,5]=SS #total intensity (4\pi J) in middle of bottom layer of atmosphere, same as what CLIMA code reports (erg/s/cm2/nm)
	toprint[:,6]=surface_intensity #total intensity incident on surface. It is equal to sum of direct intensity and diffuse downward intensity. (erg/s/cm2/nm)
	toprint[:,7]=surface_diffuse_intensity #total downward diffuse intensity at surface, i.e. 2\pi*I_minus[N]. (erg/s/cm2/nm)
	toprint[:,8]=surface_direct_intensity #total direct intensity incident at surface, i.e. I_0*exp(-tau_0/mu_0). (erg/s/cm2/nm)


	header='Left Bin Edge (nm)	Right Bin Edge (nm)	Bin Center (nm)		Top of Atm Intensity (erg/s/nm/cm2)		Total Surface Flux (erg/s/nm/cm2)		Total Intensity at BOA (erg/s/nm/cm2)		Total Surface Intensity (erg/s/nm/cm2)		Total Surface Diffuse Intensity (erg/s/nm/cm2)		Total Surface Direct Intensity (erg/s/nm/cm2)\n'
	f=open(writefilename, 'w')
	f.write(header)
	np.savetxt(f, toprint, delimiter='		', fmt='%1.7e', newline='\n')
	f.close()
	print 'END\n'
	#plt.show()

	"""
	********************************************************************************
	SCRATCH CODE
	********************************************************************************
	"""

	##############################
	#########Print spectra for Wuttke measurement:
	##############################
	##toprint=np.zeros([np.size(wav_centers), 6])
	##toprint[:,0]=wav_leftedges #left edge of wavelength bin (nm)
	##toprint[:,1]=wav_rightedges #right edge of wavelength bin (nm)
	##toprint[:,2]=wav_centers #center of wavelength bin (nm)
	##toprint[:,3]=intensity_toa #intensity incident at top of atmosphere (erg/s/cm2/nm)
	##toprint[:,4]=surface_diffuse_intensity #diffuse intensity at bottom of atmosphere (erg/s/cm2/nm)
	##toprint[:,5]=surface_intensity_basecase #reference measurements
	##header='Left Bin Edge (nm)	Right Bin Edge (nm)	Bin Center (nm)		Top of Atm Intensity (erg/s/nm/cm2)		Total Surface Flux (erg/s/nm/cm2)		Total Intensity at BOA (erg/s/nm/cm2)		Total Surface Intensity (erg/s/nm/cm2)		Total Surface Diffuse Intensity (erg/s/nm/cm2)		Total Surface Direct Intensity (erg/s/nm/cm2)\n'
	##f=open(writefilename, 'w')
	##f.write(header)
	##np.savetxt(f, toprint, delimiter='		', fmt='%1.7e', newline='\n')
	##f.close()

