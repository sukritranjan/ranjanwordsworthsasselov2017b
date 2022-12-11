# -*- coding: iso-8859-1 -*-
"""
Plot the Results section of the paper
"""

import numpy as np
import pdb
import matplotlib.pyplot as plt
import scipy.stats
from scipy import interpolate as interp
import pdb
from matplotlib.pyplot import cm

hc=1.98645e-9 #value of h*c in erg*nm

def cm2inch(cm): #function to convert cm to inches; useful for complying with Astrobiology size guidelines
    return cm/2.54

def erg2phot(energy, wavelength):
    """
    Convert energy (radiances, fluxes) to photon (radiances, fluxes)
    
    Takes: 
    ---energy, in ergs (etc...)
    ---corresponding wavelength(s) in nm
    Returns:
    ---corresponding number of photons in photons (etc...)
    """
    
    numphot=energy*wavelength/hc
    return numphot

plot_steadystate_toa_surf=True #Plot the TOA fluxes and the surface radiances for each stellar type
plot_steadystate_doserates=False #Plot the dose rates as a function of stellar type
plot_steadystate_doserates_relative=False #Plot the bad/good dose rates as a function of stellar type

plot_flare_toa_surf=False #Plot the TOA fluxes and the surface radiances for each stellar type
plot_flare_doserates=False #Plot the dose rates as a function of stellar type
plot_flare_doserates_relative=False #Plot the bad/good dose rates as a function of stellar type


############################################
###I. Plot all TOA, BOA radiances for non-flaring stars
############################################

if plot_steadystate_toa_surf:
    #Define file list
    #elt_list=np.array(['general_youngsun_earth_highres_widecoverage_spectral','muscles_gj832_stellar', 'muscles_gj176_stellar','vpl_adleo_stellar','muscles_gj667c_stellar', 'muscles_gj581_stellar','muscles_gj436_stellar', 'vpl_gj644_stellar', 'muscles_gj876_stellar', 'muscles_proxcen_stellar', 'vpl_proxcen_stellar', 'muscles_gj1214_stellar','rugheimer2015_M8A_ep0_stellar']) #list of files to import and consider
    #labels=np.array(['Young \nSun','GJ 832','GJ 176','AD Leo','GJ 667C','GJ 581','GJ 436','GJ644','GJ876','Prox Cen \n(MUSCLES)','Prox Cen \n(VPL)','GJ1214','R+2015 M8V'])

    elt_list=np.array(['general_youngsun_earth_highres_widecoverage_spectral','muscles_gj832_stellar' ,'vpl_adleo_stellar','muscles_gj667c_stellar', 'muscles_gj581_stellar', 'muscles_gj876_stellar', 'muscles_proxcen_stellar', 'hip23309_stellar']) #list of files to import and consider
    labels=np.array(['Young \nSun','GJ 832','AD Leo','GJ 667C','GJ 581','GJ 876','Prox Cen', 'HIP 23309'])
    
    numelt=len(elt_list)
    
    #Read in models
    wav_dict={}
    toa_dict={}    
    surf_dict={}
    
    for elt in elt_list:
        wav_dict[elt], toa_dict[elt], surf_dict[elt]=np.genfromtxt('./TwoStreamOutput/generalprebioticatm_exponential_'+elt+'.dat', skip_header=1, skip_footer=0,usecols=(2,3,6), unpack=True)# | 2:centers of wavelength bins, nm | 3:TOA flux, erg/cm2/s/nm | 6: surface radiance, erg/cm2/s/nm
    
    #Plot
    fig, ax1=plt.subplots(1, figsize=(cm2inch(16.5)*1.1,6.), sharex=True, sharey=True)
    markersizeval=8.
    colors=cm.rainbow(np.linspace(0,1,numelt-1))
    markerlist=['s', 'h', 'X','d','o','^','p','*']
    ax1.set_title('Surface Radiance', fontsize=14)
    for ind in range(0, numelt):
        elt=elt_list[ind]
        
        if elt=='general_youngsun_earth_highres_widecoverage_spectral':
            colorval='black'
            linewidthval=3
        elif elt=='hip23309_stellar':
            linewidthval=3
            colorval=colors[ind-1]
        else:
            colorval=colors[ind-1]
            linewidthval=1

        
        ax1.plot(wav_dict[elt], erg2phot(surf_dict[elt], wav_dict[elt]), marker=markerlist[ind], markeredgewidth=0., markersize=markersizeval, linestyle='-', linewidth=linewidthval, color=colorval, label=labels[ind])
        
        
        inds_to_average=np.where((wav_dict[elt]>=200.0) & (wav_dict[elt] <=282.0))
        print(elt, np.average(erg2phot(surf_dict[elt], wav_dict[elt])[inds_to_average])/1.0E10)

        

    ax1.set_yscale('log')
    ax1.set_ylim([1.e7, 1.e14])
    ax1.set_ylabel(r'Surface Radiance (photons s$^{-1}$cm$^{-2}$nm$^{-1}$)', fontsize=12)
    ax1.set_xscale('linear')
    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_xlim([200., 280.])
    
    xvals=np.linspace(200., 280., num=100)
    ax1.fill_between(xvals, np.ones(np.shape(xvals))*3.2E10, np.ones(np.shape(xvals))*1.04E11, facecolor='hotpink')
    

    ax1.legend(bbox_to_anchor=[-0.15, 1.08, 1.2, .152], loc=3, ncol=4, mode='expand', borderaxespad=0., fontsize=12)
    plt.tight_layout(rect=(0,0,1,0.87))
    plt.subplots_adjust(wspace=0., hspace=0.2)
    plt.savefig('./Plots/results_radiances_steadystate_toa_boa_spectype_FORREV.pdf', orientation='portrait',papertype='letter', format='pdf')

if plot_steadystate_doserates:
    #Define file list
    elt_list=np.array(['general_youngsun_earth_highres_widecoverage_spectral','muscles_gj832_stellar', 'muscles_gj176_stellar','vpl_adleo_stellar','muscles_gj667c_stellar', 'muscles_gj581_stellar','muscles_gj436_stellar', 'muscles_gj876_stellar', 'muscles_proxcen_stellar', 'muscles_gj1214_stellar','rugheimer2015_M8A_ep0_stellar']) #list of files to import and consider
    labels=np.array(['', 'Young \nSun','GJ 832','GJ 176','AD Leo','GJ 667C','GJ 581','GJ 436','GJ876','Prox Cen \n(MUSCLES)','GJ1214','R+2015 M8V', ''])
    
    numelt=len(elt_list)
    
    #Load data
    dose_120_200=np.zeros(np.shape(elt_list)) #surface radiance integrated 120-200 nm
    dose_200_300=np.zeros(np.shape(elt_list)) #surface radiance integrated 200-300 nm
    dose_ump_193=np.zeros(np.shape(elt_list)) #dose rate for UMP glycosidic bond cleavage, assuming lambda0=193
    dose_ump_230=np.zeros(np.shape(elt_list)) #dose rate for UMP glycosidic bond cleavage, assuming lambda0=230
    dose_ump_254=np.zeros(np.shape(elt_list)) #dose rate for UMP glycosidic bond cleavage, assuming lambda0=254
    dose_cucn3_254=np.zeros(np.shape(elt_list)) #dose rate for solvated electron production from tricyanocuprate, assuming lambda0=254
    dose_cucn3_300=np.zeros(np.shape(elt_list)) #dose rate for solvated electron production from tricyanocuprate, assuming lambda0=300
    
    num_elt=len(elt_list)
    for ind in range(0, num_elt):
        elt=elt_list[ind]
        dose_120_200[ind],dose_200_300[ind],dose_ump_193[ind],dose_ump_230[ind],dose_ump_254[ind],dose_cucn3_254[ind],dose_cucn3_300[ind]=np.genfromtxt('./DoseRates/dose_rates_generalprebioticatm_exponential_'+elt+'.dat', skip_header=1, skip_footer=0,usecols=(0,1,2,3,4,5,6), unpack=True)
    
    #Plot results
    fig, ax=plt.subplots(1, figsize=(8.5,7.))
    markersizeval=7.
    colors=cm.rainbow(np.linspace(0,1,7))
    
    abscissa=range(0, numelt+2)
    
    ax.plot(abscissa[1:-1], dose_200_300, marker='s',  markersize=markersizeval, linestyle='', color=colors[1], label=r'Radiance 200-300 nm')    
    ax.plot(abscissa[1:-1], dose_ump_193, marker='s', markersize=markersizeval, linestyle='', color=colors[2], label=r'UMP Bond Cleavage ($\lambda_0=193$)')
    ax.plot(abscissa[1:-1], dose_ump_230, marker='s', markersize=markersizeval, linestyle='', color=colors[3], label=r'UMP Bond Cleavage ($\lambda_0=230$)')
    ax.plot(abscissa[1:-1], dose_ump_254, marker='s',  markersize=markersizeval, linestyle='', color=colors[4], label=r'UMP Bond Cleavage ($\lambda_0=254$)')
    ax.plot(abscissa[1:-1], dose_cucn3_254, marker='s', markersize=markersizeval, linestyle='', color=colors[5], label=r'CuCN$_3$ Photoionization ($\lambda_0=254$)')
    ax.plot(abscissa[1:-1], dose_cucn3_300, marker='s',  markersize=markersizeval, linestyle='', color=colors[6], label=r'CuCN$_3$ Photoionization ($\lambda_0=300$)')
    
    
    ax.set_xticks(abscissa)
    ax.set_xticklabels(labels, rotation=90, fontsize=16)
    
    ax.set_yscale('log')
    ax.set_ylim([1.e-4, 2.e0])
    ax.set_ylabel(r'Relative Dose Rate $\bar{D}_i$', fontsize=16)

    ax.legend(bbox_to_anchor=[0, 1.08, 1., .152], loc=3, ncol=2, mode='expand', borderaxespad=0., fontsize=14)
    plt.tight_layout(rect=(0,0,1,0.8))
    plt.subplots_adjust(wspace=0., hspace=0.2)
    plt.savefig('./Plots/results_steadystate_doserates.pdf', orientation='portrait',papertype='letter', format='pdf')

if plot_steadystate_doserates_relative:
    #Define file list
    elt_list=np.array(['general_youngsun_earth_highres_widecoverage_spectral','muscles_gj832_stellar', 'muscles_gj176_stellar','vpl_adleo_stellar','muscles_gj667c_stellar', 'muscles_gj581_stellar','muscles_gj436_stellar', 'muscles_gj876_stellar', 'muscles_proxcen_stellar', 'muscles_gj1214_stellar','rugheimer2015_M8A_ep0_stellar']) #list of files to import and consider
    labels=np.array(['', 'Young \nSun','GJ 832','GJ 176','AD Leo','GJ 667C','GJ 581','GJ 436','GJ876','Prox Cen \n(MUSCLES)','GJ1214','R+2015 M8V', ''])
    numelt=len(elt_list)
    
    #Load data
    dose_120_200=np.zeros(np.shape(elt_list)) #surface radiance integrated 120-200 nm
    dose_200_300=np.zeros(np.shape(elt_list)) #surface radiance integrated 200-300 nm
    dose_ump_193=np.zeros(np.shape(elt_list)) #dose rate for UMP glycosidic bond cleavage, assuming lambda0=193
    dose_ump_230=np.zeros(np.shape(elt_list)) #dose rate for UMP glycosidic bond cleavage, assuming lambda0=230
    dose_ump_254=np.zeros(np.shape(elt_list)) #dose rate for UMP glycosidic bond cleavage, assuming lambda0=254
    dose_cucn3_254=np.zeros(np.shape(elt_list)) #dose rate for solvated electron production from tricyanocuprate, assuming lambda0=254
    dose_cucn3_300=np.zeros(np.shape(elt_list)) #dose rate for solvated electron production from tricyanocuprate, assuming lambda0=300
    
    num_elt=len(elt_list)
    for ind in range(0, num_elt):
        elt=elt_list[ind]
        dose_120_200[ind],dose_200_300[ind],dose_ump_193[ind],dose_ump_230[ind],dose_ump_254[ind],dose_cucn3_254[ind],dose_cucn3_300[ind]=np.genfromtxt('./DoseRates/dose_rates_generalprebioticatm_exponential_'+elt+'.dat', skip_header=1, skip_footer=0,usecols=(0,1,2,3,4,5,6), unpack=True)
    
    #Plot results
    fig, ax=plt.subplots(1, figsize=(8,6.))
    markersizeval=7.
    colors=cm.rainbow(np.linspace(0,1,6))
    
    abscissa=range(0, numelt+2)
    
    ax.plot(abscissa[1:-1], dose_ump_193/dose_cucn3_254, marker='s', markersize=markersizeval, linestyle='', color=colors[0], label=r'UMP-193/CuCN3-254')
    ax.plot(abscissa[1:-1], dose_ump_230/dose_cucn3_254, marker='s', markersize=markersizeval, linestyle='', color=colors[1], label=r'UMP-230/CuCN3-254')
    ax.plot(abscissa[1:-1], dose_ump_254/dose_cucn3_254, marker='s', markersize=markersizeval, linestyle='', color=colors[2], label=r'UMP-254/CuCN3-254')

    ax.plot(abscissa[1:-1], dose_ump_193/dose_cucn3_300, marker='s', markersize=markersizeval, linestyle='', color=colors[3], label=r'UMP-193/CuCN3-300')
    ax.plot(abscissa[1:-1], dose_ump_230/dose_cucn3_300, marker='s', markersize=markersizeval, linestyle='', color=colors[4], label=r'UMP-230/CuCN3-300')
    ax.plot(abscissa[1:-1], dose_ump_254/dose_cucn3_300, marker='s', markersize=markersizeval, linestyle='', color=colors[5], label=r'UMP-254/CuCN3-300')
    

    ax.set_xticks(abscissa)
    ax.set_xticklabels(labels, rotation=90, fontsize=16)
    
    ax.set_yscale('log')
    ax.set_ylabel(r'$\bar{D}_{UMP-X}/\bar{D}_{CuCN3-Y}$', fontsize=16)

    ax.legend(bbox_to_anchor=[0, 1.08, 1., .152], loc=3, ncol=2, mode='expand', borderaxespad=0., fontsize=14)
    plt.tight_layout(rect=(0,0,1,0.8))
    plt.subplots_adjust(wspace=0., hspace=0.2)
    plt.savefig('./Plots/results_steadystate_reldoserates.pdf', orientation='portrait',papertype='letter', format='pdf')

############################################
###II. Plot all TOA, BOA radiances for flares
############################################

if plot_flare_toa_surf:
    #Define file list
    elt_list=np.array(['general_youngsun_earth_highres_widecoverage_spectral','vpl_adleo_stellar','vpl_adleo_greatflare_stellar']) #list of files to import and consider
    labels=np.array(['Young Sun','AD Leo (Quiesc.)','AD Leo (Flare)'])
    
    numelt=len(elt_list)
    
    #Read in models
    wav_dict={}
    toa_dict={}    
    surf_dict={}
    
    for elt in elt_list:
        wav_dict[elt], toa_dict[elt], surf_dict[elt]=np.genfromtxt('./TwoStreamOutput/generalprebioticatm_exponential_'+elt+'.dat', skip_header=1, skip_footer=0,usecols=(2,3,6), unpack=True)# | 2:centers of wavelength bins, nm | 3:TOA flux, erg/cm2/s/nm | 6: surface radiance, erg/cm2/s/nm
    
    #Plot
    fig, (ax0, ax1)=plt.subplots(2, figsize=(cm2inch(16.5)*1.1,10.), sharex=True, sharey=True)
    markersizeval=4.
    colors=cm.rainbow(np.linspace(0,1,numelt-1))
    
    ax0.set_title('TOA Flux', fontsize=16)
    ax1.set_title('Surface Radiance', fontsize=16)
    for ind in range(0, numelt):
        elt=elt_list[ind]
        
        if elt=='general_youngsun_earth_highres_widecoverage_spectral':
            colorval='black'
        else:
            colorval=colors[ind-1]
        
        ax0.plot(wav_dict[elt], erg2phot(toa_dict[elt], wav_dict[elt]), marker='s', markeredgewidth=0., markersize=markersizeval, linestyle='-', linewidth=1, color=colorval, label=labels[ind])
        ax1.plot(wav_dict[elt], erg2phot(surf_dict[elt], wav_dict[elt]), marker='s', markeredgewidth=0., markersize=markersizeval, linestyle='-', linewidth=1, color=colorval, label=labels[ind])
        
    ax0.set_ylabel(r'Flux (photons s$^{-1}$cm$^{-2}$nm$^{-1}$)', fontsize=16)
    ax0.set_yscale('log')
    ax1.set_yscale('log')
    ax1.set_ylim([1.e5, 1.e15])
    ax1.set_ylabel(r'Surface Radiance (photons s$^{-1}$cm$^{-2}$nm$^{-1}$)', fontsize=16)
    ax1.set_xscale('linear')
    ax1.set_xlabel('Wavelength (nm)', fontsize=16)
    ax1.set_xlim([120., 400.])


    ax0.legend(bbox_to_anchor=[-0.15, 1.08, 1.2, .152], loc=3, ncol=4, mode='expand', borderaxespad=0., fontsize=14)
    plt.tight_layout(rect=(0,0,1,0.87))
    plt.subplots_adjust(wspace=0., hspace=0.2)
    plt.savefig('./Plots/results_flare_radiances_toa_boa_spectype.pdf', orientation='portrait',papertype='letter', format='pdf')

if plot_flare_doserates:
    #Define file list
    elt_list=np.array(['general_youngsun_earth_highres_widecoverage_spectral','vpl_adleo_stellar','vpl_adleo_greatflare_stellar']) #list of files to import and consider
    labels=np.array(['', 'Young Sun','AD Leo (Quiesc.)','AD Leo (Flare)', ''])
    
    numelt=len(elt_list)
    
    #Load data
    dose_120_200=np.zeros(np.shape(elt_list)) #surface radiance integrated 120-200 nm
    dose_200_300=np.zeros(np.shape(elt_list)) #surface radiance integrated 200-300 nm
    dose_ump_193=np.zeros(np.shape(elt_list)) #dose rate for UMP glycosidic bond cleavage, assuming lambda0=193
    dose_ump_230=np.zeros(np.shape(elt_list)) #dose rate for UMP glycosidic bond cleavage, assuming lambda0=230
    dose_ump_254=np.zeros(np.shape(elt_list)) #dose rate for UMP glycosidic bond cleavage, assuming lambda0=254
    dose_cucn3_254=np.zeros(np.shape(elt_list)) #dose rate for solvated electron production from tricyanocuprate, assuming lambda0=254
    dose_cucn3_300=np.zeros(np.shape(elt_list)) #dose rate for solvated electron production from tricyanocuprate, assuming lambda0=300
    
    num_elt=len(elt_list)
    for ind in range(0, num_elt):
        elt=elt_list[ind]
        dose_120_200[ind],dose_200_300[ind],dose_ump_193[ind],dose_ump_230[ind],dose_ump_254[ind],dose_cucn3_254[ind],dose_cucn3_300[ind]=np.genfromtxt('./DoseRates/dose_rates_generalprebioticatm_exponential_'+elt+'.dat', skip_header=1, skip_footer=0,usecols=(0,1,2,3,4,5,6), unpack=True)
    
    #Plot results
    fig, ax=plt.subplots(1, figsize=(7.5,6.))
    markersizeval=7.
    colors=cm.rainbow(np.linspace(0,1,7))
    
    abscissa=range(0, numelt+2)
    
    ax.plot(abscissa[1:-1], dose_200_300, marker='s',markersize=markersizeval, linestyle='', color=colors[1], label=r'Radiance 200-300 nm')    
    ax.plot(abscissa[1:-1], dose_ump_193, marker='s', markersize=markersizeval, linestyle='', color=colors[2], label=r'UMP Bond Cleavage ($\lambda_0=193$)')
    ax.plot(abscissa[1:-1], dose_ump_230, marker='s',markersize=markersizeval, linestyle='', color=colors[3], label=r'UMP Bond Cleavage ($\lambda_0=230$)')
    ax.plot(abscissa[1:-1], dose_ump_254, marker='s', markersize=markersizeval, linestyle='', color=colors[4], label=r'UMP Bond Cleavage ($\lambda_0=254$)')
    ax.plot(abscissa[1:-1], dose_cucn3_254, marker='s', markersize=markersizeval, linestyle='',color=colors[5], label=r'CuCN$_3$ Photoionization ($\lambda_0=254$)')
    ax.plot(abscissa[1:-1], dose_cucn3_300, marker='s', markersize=markersizeval, linestyle='', color=colors[6], label=r'CuCN$_3$ Photoionization ($\lambda_0=300$)')
    
    
    ax.set_xticks(abscissa)
    ax.set_xticklabels(labels, rotation=90, fontsize=16)
    
    ax.set_yscale('log')
    ax.set_ylabel(r'Relative Dose Rate $\bar{D}_i$', fontsize=16)

    ax.legend(bbox_to_anchor=[0, 1.08, 1., .152], loc=3, ncol=2, mode='expand', borderaxespad=0., fontsize=12)
    plt.tight_layout(rect=(0,0,1,0.8))
    plt.subplots_adjust(wspace=0., hspace=0.2)
    plt.savefig('./Plots/results_flare_doserates.pdf', orientation='portrait',papertype='letter', format='pdf')

if plot_flare_doserates_relative:
    #Define file list
    elt_list=np.array(['general_youngsun_earth_highres_widecoverage_spectral','vpl_adleo_stellar','vpl_adleo_greatflare_stellar']) #list of files to import and consider
    labels=np.array(['', 'Young Sun','AD Leo (Quiesc.)','AD Leo (Flare)',''])
    numelt=len(elt_list)
    
    #Load data
    dose_120_200=np.zeros(np.shape(elt_list)) #surface radiance integrated 120-200 nm
    dose_200_300=np.zeros(np.shape(elt_list)) #surface radiance integrated 200-300 nm
    dose_ump_193=np.zeros(np.shape(elt_list)) #dose rate for UMP glycosidic bond cleavage, assuming lambda0=193
    dose_ump_230=np.zeros(np.shape(elt_list)) #dose rate for UMP glycosidic bond cleavage, assuming lambda0=230
    dose_ump_254=np.zeros(np.shape(elt_list)) #dose rate for UMP glycosidic bond cleavage, assuming lambda0=254
    dose_cucn3_254=np.zeros(np.shape(elt_list)) #dose rate for solvated electron production from tricyanocuprate, assuming lambda0=254
    dose_cucn3_300=np.zeros(np.shape(elt_list)) #dose rate for solvated electron production from tricyanocuprate, assuming lambda0=300
    
    num_elt=len(elt_list)
    for ind in range(0, num_elt):
        elt=elt_list[ind]
        dose_120_200[ind],dose_200_300[ind],dose_ump_193[ind],dose_ump_230[ind],dose_ump_254[ind],dose_cucn3_254[ind],dose_cucn3_300[ind]=np.genfromtxt('./DoseRates/dose_rates_generalprebioticatm_exponential_'+elt+'.dat', skip_header=1, skip_footer=0,usecols=(0,1,2,3,4,5,6), unpack=True)
    
    #Plot results
    fig, ax=plt.subplots(1, figsize=(7.5,6.))
    markersizeval=7.
    colors=cm.rainbow(np.linspace(0,1,6))
    
    abscissa=range(0, numelt+2)
    
    ax.plot(abscissa[1:-1], dose_ump_193/dose_cucn3_254, marker='s', markersize=markersizeval, linestyle='', color=colors[0], label=r'UMP-193/CuCN3-254')
    ax.plot(abscissa[1:-1], dose_ump_230/dose_cucn3_254, marker='s', markersize=markersizeval, linestyle='', color=colors[1], label=r'UMP-230/CuCN3-254')
    ax.plot(abscissa[1:-1], dose_ump_254/dose_cucn3_254, marker='s', markersize=markersizeval, linestyle='', color=colors[2], label=r'UMP-254/CuCN3-254')

    ax.plot(abscissa[1:-1], dose_ump_193/dose_cucn3_300, marker='s', markersize=markersizeval, linestyle='', color=colors[3], label=r'UMP-193/CuCN3-300')
    ax.plot(abscissa[1:-1], dose_ump_230/dose_cucn3_300, marker='s', markersize=markersizeval, linestyle='', color=colors[4], label=r'UMP-230/CuCN3-300')
    ax.plot(abscissa[1:-1], dose_ump_254/dose_cucn3_300, marker='s', markersize=markersizeval, linestyle='', color=colors[5], label=r'UMP-254/CuCN3-300')    

    ax.set_xticks(abscissa)
    ax.set_xticklabels(labels, rotation=90, fontsize=16)
    
    ax.set_yscale('log')
    ax.set_ylabel(r'$\bar{D}_{UMP-X}/\bar{D}_{CuCN3-Y}$', fontsize=16)

    ax.legend(bbox_to_anchor=[0, 1.08, 1., .152], loc=3, ncol=2, mode='expand', borderaxespad=0., fontsize=14)
    plt.tight_layout(rect=(0,0,1,0.8))
    plt.subplots_adjust(wspace=0., hspace=0.2)
    plt.savefig('./Plots/results_flare_reldoserates.pdf', orientation='portrait',papertype='letter', format='pdf')

plt.show()    
