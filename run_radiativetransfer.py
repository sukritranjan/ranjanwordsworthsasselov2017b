# -*- coding: iso-8859-1 -*-
"""
Purpose of this file is to run the uv_radtrans function from radiativetransfer.py to generate the surface radiance calculations used to derive the results in our paper.
"""

import radiativetransfer as rt
import numpy as np
import pdb

##################################
###First research case: Replicate R+2015 3.9 Ga calculations
##################################

#elt_list=np.array(['M8A_ep0']) #list of files to import and consider


#for ind in range(0, len(elt_list)):
	#elt=elt_list[ind]
	#inputspectrafile='rugheimer2015_'+elt+'_stellar_input.dat'
	#inputatmofilelabel='rugheimer2015_'+elt
	
	#rt.uv_radtrans(z_upper_limit=64.e5, z_step=1.e5, inputatmofilelabel=inputatmofilelabel, outputfilelabel=outputfilelabel, inputspectrafile=inputspectrafile,TDXC=False, DeltaScaling=False, SZA_deg=48.2, albedoflag='uniformalbedo',uniformalbedo=0.2, includedust=False, includeco2cloud=False,includeh2ocloud=False)


#################################
##First: Run on 1-bar exponential atmosphere 
#################################

elt_list=np.array(['general_youngsun_earth_highres_widecoverage_spectral','rugheimer2015_M8A_ep0_stellar', 'muscles_gj1214_stellar',
'muscles_gj176_stellar','muscles_gj436_stellar','muscles_gj581_stellar','muscles_gj667c_stellar','muscles_gj832_stellar','muscles_gj876_stellar','muscles_proxcen_stellar', 'vpl_adleo_greatflare_stellar', 'vpl_adleo_stellar', 'vpl_gj644_stellar', 'vpl_proxcen_stellar', 'rugheimer2015_M8A_ep0_stellar']) #list of files to import and consider


for ind in range(0, len(elt_list)):
	elt=elt_list[ind]
	inputspectrafile=elt+'_input.dat'
	inputatmofilelabel='generalprebioticatm_exponential'
	outputfilelabel=elt

	
	rt.uv_radtrans(z_upper_limit=64.e5, z_step=1.e5, inputatmofilelabel=inputatmofilelabel, outputfilelabel=outputfilelabel, inputspectrafile=inputspectrafile,TDXC=False, DeltaScaling=False, SZA_deg=48.2, albedoflag='uniformalbedo',uniformalbedo=0.2, includedust=False, includeco2cloud=False,includeh2ocloud=False)

#################################
##Second: Run on 1-microbar exponential atmosphere (thin atmosphere case)
#################################

elt_list=np.array(['general_youngsun_earth_highres_widecoverage_spectral','rugheimer2015_M8A_ep0_stellar', 'muscles_gj1214_stellar',
'muscles_gj176_stellar','muscles_gj436_stellar','muscles_gj581_stellar','muscles_gj667c_stellar','muscles_gj832_stellar','muscles_gj876_stellar','muscles_proxcen_stellar', 'vpl_adleo_greatflare_stellar', 'vpl_adleo_stellar', 'vpl_gj644_stellar', 'vpl_proxcen_stellar']) #list of files to import and consider


for ind in range(0, len(elt_list)):
	elt=elt_list[ind]
	inputspectrafile=elt+'_input.dat'
	inputatmofilelabel='generalprebioticatm_exponential_thinatm'
	outputfilelabel=elt

	
	rt.uv_radtrans(z_upper_limit=64.e5, z_step=1.e5, inputatmofilelabel=inputatmofilelabel, outputfilelabel=outputfilelabel, inputspectrafile=inputspectrafile,TDXC=False, DeltaScaling=False, SZA_deg=48.2, albedoflag='uniformalbedo',uniformalbedo=0.2, includedust=False, includeco2cloud=False,includeh2ocloud=False)

#################################
##Third: Run on 1-bar atmosphere, dry adiabat evolution, but full humidity (ignore thermodynamic effects of water vapor, permissible if H2O levels low which they are for T_0=288 K)
#################################

elt_list=np.array(['general_youngsun_earth_highres_widecoverage_spectral','rugheimer2015_M8A_ep0_stellar', 'muscles_gj1214_stellar',
'muscles_gj176_stellar','muscles_gj436_stellar','muscles_gj581_stellar','muscles_gj667c_stellar','muscles_gj832_stellar','muscles_gj876_stellar','muscles_proxcen_stellar', 'vpl_adleo_greatflare_stellar', 'vpl_adleo_stellar', 'vpl_gj644_stellar', 'vpl_proxcen_stellar', 'rugheimer2015_M8A_ep0_stellar']) #list of files to import and consider


for ind in range(0, len(elt_list)):
	elt=elt_list[ind]
	inputspectrafile=elt+'_input.dat'
	inputatmofilelabel='generalprebioticatm_dryadiabat_relH=1'
	outputfilelabel=elt

	
	rt.uv_radtrans(z_upper_limit=64.e5, z_step=1.e5, inputatmofilelabel=inputatmofilelabel, outputfilelabel=outputfilelabel, inputspectrafile=inputspectrafile,TDXC=False, DeltaScaling=False, SZA_deg=48.2, albedoflag='uniformalbedo',uniformalbedo=0.2, includedust=False, includeco2cloud=False,includeh2ocloud=False)