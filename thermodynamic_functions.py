# -*- coding: iso-8859-1 -*-
"""
This file defines functions that encode various thermodynamic parameters, especially of CO2 and H2O.

Abbreviation used: PPC=Principles of Planetary Climate, Ray Pierrehumbert, First Edition.

"""

########################
###Import useful libraries
########################
import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy.integrate
from matplotlib.pyplot import cm

def is_empty(any_structure):
	if any_structure:
		return False
	else:
		return True

########################
###Define useful constants, all in CGS (via http://www.astro.wisc.edu/~dolan/constants.html)
########################

#Unit conversions
km2m=1.e3 #1 km in m
km2cm=1.e5 #1 km in cm
cm2km=1.e-5 #1 cm in km
cm2m=1.e-2 #1 cm in m
amu2g=1.66054e-24 #1 amu in g
g2kg=1.e-3 #1 gram in kg
bar2atm=0.9869 #1 bar in atm
atm2bar=1./bar2atm #1 atm in bar
Pascal2bar=1.e-5 #1 Pascal in bar
bar2Pa=1.e5 #1 bar in Pascal
Pa2bar=1.e-5 #1 Pascal in bar
deg2rad=np.pi/180.
bar2barye=1.e6 #1 Bar in Barye (the cgs unit of pressure)
barye2bar=1.e-6 #1 Barye in Bar
micron2m=1.e-6 #1 micron in m
micron2cm=1.e-4 #1 micron in cm
metricton2kg=1000. #1 metric ton in kg

#Fundamental constants
c=2.997924e10 #speed of light, cm/s
h=6.6260755e-27 #planck constant, erg/s
k=1.380658e-16 #boltzmann constant, erg/K
sigma=5.67051e-5 #Stefan-Boltzmann constant, erg/(cm^2 K^4 s)
R_earth=6371.*km2m#radius of earth in m
R_sun=69.63e9 #radius of sun in cm
AU=1.496e13#1AU in cm

#Mean molecular masses
m_co2=44.01*amu2g #co2, in g
m_h2o=18.02*amu2g #h2o, in g

######Specific heat at constant pressures for gases
######Currently taken for 293K, 1 atm from http://www.engineeringtoolbox.com/specific-heat-capacity-gases-d_159.html. How do we get for non-STP?
#####c_p_co2=0.844*1.e7 #converted from kJ/(Kg*K) to erg/(g*K)
#####c_p_h2o=1.97*1.e7 #converted from kJ/(Kg*K) to erg/(g*K), measured at 1 atm and 338-589 F ()

######Mars parameters
#####g=371. #surface gravity of Mars, cm/s**2, from: http://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html

########################
###CO2 Saturation Pressure
########################
def p_sat_co2_cc(T):
	"""
	Takes: temperature in K
	Returns: CO2 partial pressure in Ba
	Based on Clausius-Clpayeron Relation, with  assumption of constant latent heat of sublimation for CO2
	"""
	T_char=3138. #L/R_co2, units of K, taken from PPC p100
	T_0=216.58 #temperature of triple point for CO2, from http://webbook.nist.gov/cgi/cbook.cgi?ID=C124389&Mask=4
	P_0=5.185*bar2barye #pressure of triple point for CO2, from http://webbook.nist.gov/cgi/cbook.cgi?ID=C124389&Mask=4
	
	p_sat_co2=P_0*np.exp(-T_char*(1./T-1./T_0))
	return p_sat_co2
	
def t_sat_co2_cc(P):
	"""
	Takes: CO2 partial pressure in Ba
	Returns: temperature in K
	Based on Clausius-Clpayeron Relation, with  assumption of constant latent heat of sublimation for CO2
	"""
	T_char=3138. #L/R_co2, units of K, taken from PPC p100
	T_0=216.58 #temperature of triple point for CO2, from http://webbook.nist.gov/cgi/cbook.cgi?ID=C124389&Mask=4
	P_0=5.185*bar2barye #pressure of triple point for CO2, from http://webbook.nist.gov/cgi/cbook.cgi?ID=C124389&Mask=4
	
	t_sat_co2=T_0/(1.-(T_0/T_char)*np.log(P/P_0))
	return t_sat_co2

def t_sat_co2_fanale(p):
	"""
	Takes: CO2 pressure in Ba
	Returns: temperature at which given pressure saturates
	Based on expression of Fanale+1982, via Forget+2013
	Did not find this expression in Fanale...
	Warning: seems to be a discontinuity at the T_c corresponding to the step condition?! Only ~2K, but still!
	"""
	p_pa=p*barye2bar*bar2Pa #convert from Ba to Pa

	if type(p)==float: #Have to handle non-arrays slightly differently
		if (p_pa<518000.):
			t_sat=-3167.8/(np.log(0.01*p_pa)-23.23)
		if (p_pa >=518000.):
			t_sat=684.2-92.3*np.log(p_pa)+4.32*np.log(p_pa)**2.
	else:
		t_sat=np.zeros(np.shape(p))
		
		inds1=np.where(p_pa<518000.) #branch 1, should be the only branch accessed for our work...
		inds2=np.where(p_pa >=518000.) #branch 2
		
		t_sat[inds1]=-3167.8/(np.log(0.01*p_pa[inds1])-23.23)
		t_sat[inds2]=684.2-92.3*np.log(p_pa[inds2])+4.32*np.log(p_pa[inds2])**2.
		
	return t_sat

def p_sat_co2_fanale(T):
	"""
	Takes: temperature in K
	Returns: saturation pressure of CO2 in Ba
	Based on inverting expression of Fanale+1982, via Forget+2013
	The T-p mapping is not one-to-one. We choose only the branch with p<5.18e5Pa=5.18e6 Ba=5.18 bar.
	"""
	p_pa_1=100.*np.exp(23.23-3167.8/T) #inversion of the expression from the P<518000 Pa branch
	
	p_pa=np.copy(p_pa_1)
	
	inds=np.where(p_pa_1>518000)
	
	if not(is_empty(np.squeeze(inds))): #if inds is not empty
		p_pa_2=np.exp((92.3+np.sqrt(92.3**2.-4*4.32*(684.2-T)))/(2*4.32)) #inversion of the expression from the P>518000 Pa branch. Choice of + value in quadratic equation forced by requirement that the presures be greater than 518000 Pa
		print 'WARNING: Saturation Pressure Computed Using P>=518000. Branch of Fanale+1982 expression, excercise extreme caution)'
		p_pa[inds]=p_pa_2[inds]
	p=p_pa*Pa2bar*bar2barye
	return p

def p_sat_co2_kasting(T):
	"""
	Takes: temperature in K
	Returns: saturation pressure of CO2 in Ba
	Based on expressions from Kasting+1991, which are in turn based on results from Vukalovich and Altunin 1968 p97
	"""
	inds1=np.where(T>=216.56) #upper branch
	inds2=np.where(T<216.56) #lower branch
	
	p_atm=np.zeros(np.shape(T))
	
	p_atm[inds1]=10.**(3.128082-867.2124/(T[inds1])+18.65612e-3*(T[inds1])-72.48820e-6*(T[inds1])**2.+93.e-9*(T[inds1])**3.)
	p_atm[inds2]=10.**(6.760956-1284.07/((T[inds2])-4.718)+1.256e-4*((T[inds2])-143.15))
	
	p=p_atm*atm2bar*bar2barye
	return p

###Test plot
#T_list=np.arange(150., 351.) #plausible range of temperatures
#P_list=10.**(np.arange(-3, 3.1, step=0.1))*bar2barye

#fig, (ax1, ax2)=plt.subplots(2, figsize=(8,10), sharex=True, sharey=True)
#ax1.set_title('CO2 Saturation Curves')
#ax1.plot(T_list, p_sat_co2_cc(T_list)*barye2bar, color='green', label='Constant-L+Clausius-Clapyeron P(T)')
#ax1.plot(T_list, p_sat_co2_kasting(T_list)*barye2bar, color='red', label='Kasting+1991 P(T)')
#ax1.plot(T_list, p_sat_co2_fanale(T_list)*barye2bar, color='blue', label='Fanale+1982 P(T)')
#ax1.set_ylabel('Pressure (bar)')
#ax1.set_xlabel('Temperature (K)')
#ax1.set_yscale('log')
#ax1.set_ylim(1.e-3, 1.e3)
#ax1.set_xlim(150., 350.)
#ax1.legend(loc=0)

#ax2.plot(t_sat_co2_cc(P_list), P_list*barye2bar, color='green', label='Constant-L+Clausius-Clapyeron T(P)')
#ax2.plot(t_sat_co2_fanale(P_list), P_list*barye2bar, color='blue', label='Fanale+1982 T(P)')
#ax2.set_ylabel('Pressure (bar)')
#ax2.set_xlabel('Temperature (K)')
#ax2.set_yscale('log')
#ax2.legend(loc=0)

#plt.savefig('./Plots/p_sat_co2.pdf', orientation='portrait',papertype='letter', format='pdf')

########################
###H2O Saturation Pressure
########################
def p_sat_h2o_cc(T):
	"""
	Takes: temperature in K
	Returns: H2O partial pressure in Ba
	Based on Clausius-Clapyeron Relation, with  assumption of constant latent heat of sublimation for H2O
	"""
	T_char=5420. #L/R_h2o, units of K, taken from PPC p100, applies near 300K
	T_0=273.16 #temperature of triple point for H2O, from LN 4
	P_0=611.*Pa2bar*bar2barye #pressure of triple point for H2O, from LN 4
	
	p_sat_h2o=P_0*np.exp(-T_char*(1./T-1./T_0))
	return p_sat_h2o
	
def t_sat_h2o_cc(P):
	"""
	Takes: H2O partial pressure in Ba
	Returns: temperature in K
	Based on Clausius-Clpayeron Relation, with  assumption of constant latent heat of sublimation for H2O
	"""
	T_char=5420. #L/R_h2o, units of K, taken from PPC p100, applies near 300K
	T_0=273.16 #temperature of triple point for H2O, from LN 4
	P_0=611.*Pa2bar*bar2barye #pressure of triple point for H2O, from LN 4
	
	t_sat_h2o=T_0/(1.-(T_0/T_char)*np.log(P/P_0))
	return t_sat_h2o

def p_sat_h2o_ice_wagner(T):
	"""
	Takes: temperature in K
	Returns: H2O vapor pressure (saturation pressure?) in Ba, for sublimation of water vapor from ice
	Based on eqn 2.21 (p. 401) of Wagner and Pruss 1995, which in turn took it from Wagner+1994
	"""
	T_n=273.16
	P_n=0.000611657*1.e6*Pa2bar*bar2barye #convert pressure from Mpa to barye
	
	theta=T/T_n
	
	vapor_pressure=P_n*np.exp(-13.928169*(1.-theta**(-1.5))+34.7078238*(1.-theta**(-1.25)))
	return vapor_pressure

def p_sat_h2o_liquid_wagner(T):
	"""
	Takes: temperature in K
	Returns: H2O vapor pressure (saturation pressure?) in Ba, for water evaporating from liquid water
	Based on equation 2.5 (pg 398) of Wagner and Pruss 1995.
	"""
	T_c=647.096 #K
	P_c=22.064*1.e6*Pa2bar*bar2barye #convert pressure from Mpa to barye
	
	theta=(1.-T/T_c) #intermediate variable
	a_1=-7.85951783
	a_2=1.84408259
	a_3=-11.7866497
	a_4=22.6807411
	a_5=-15.9618719
	a_6=1.80122502
	
	term=a_1*theta+a_2*theta**1.5+a_3*theta**3.+a_4*theta**3.5+a_5*theta**4+a_6*theta**7.5
	vapor_pressure=P_c*np.exp((T_c/T)*term)
	return vapor_pressure

def p_sat_h2o_wordsworth(T):
	"""
	Takes: temperature in K
	Returns: H2O saturation pressure in Ba.
	Based on the code get_psat_H2O in the file thermodynamics.f90, obtained from Robin Wordsworth 4/19/2016
	That file in turn credits Haar, Gallagher and Kell (1984) and Meyer, McClintock, Silvestri and Spencer (1967)
	"""
	T_ref=647.25 #K
	a=np.array([-7.8889166,2.5514255,-6.716169, 33.239495, -105.38479,174.35319,-148.39348, 48.631602])
	if type(T)==float:
		if T<314.0:
			p_mpa=0.1*np.exp(6.3573118 - 8858.843/T + 607.56335*T**(-0.6))
		else:
			v=T/T_ref
			w=np.abs(1.-v)
			b=0.
			for ind in range(0, 8):
				z=ind+1.
				b=b+a[ind]*w**((z+1.)/2.)
			q=b/v
			p_mpa=22.093*np.exp(q)
	else:
		inds0=np.where(T<314.)
		inds1=np.where(T>=314.)
		p_mpa=np.zeros(np.shape(T))
		
		p_mpa[inds0]=0.1*np.exp(6.3573118 - 8858.843/T[inds0] + 607.56335*T[inds0]**(-0.6))
		
		for ind1 in inds1:
			T_ind1=T[ind1]
			v=T_ind1/T_ref
			w=np.abs(1.-v)
			b=0.
			for ind in range(0, 8):
				z=ind+1.
				b=b+a[ind]*w**((z+1.)/2.)
			q=b/v
			p_mpa[ind1]=22.093*np.exp(q)
		
	p=p_mpa*1.e6*Pa2bar*bar2barye
	
	return p

def t_sat_h2o_wordsworth(p):
	"""
	Takes: pressure in Ba
	Returns: temperature in K
	Based on the code get_Tsat_H20 in the file thermodynamics.f90, obtained from Robin Wordsworth 4/19/2016
	That code in turn was based on a polynomial fit to the output of get_psat_H2O (which we have implemented as p_sat_h2o_wordsworth above), and is reported as good between 180 and 500 K (at least)
	"""
	if np.sum(p*barye2bar<1.9e-7):
		print "Warning: temperature values input to t_sat_h2o_wordsworth are too low, have entered unphysical branch"
	
	p_pa=p*barye2bar*bar2Pa #convert from barye to Pa, which are the units the empirical relation is calibrated for
	A=np.array([0.0046, -0.0507, 0.5631, 7.5861, 207.2399])
	logp=np.log(p_pa)
	
	T_sat=A[0]*logp**4. + A[1]*logp**3. + A[2]*logp**2. + A[3]*logp + A[4] 

	return T_sat

###Test plot
#T_list=np.arange(150., 351.) #plausible range of temperatures
#P_list=10.**(np.arange(-11, 0.1, step=0.1))*bar2barye

#fig, (ax1, ax2)=plt.subplots(2, figsize=(8,10), sharex=True, sharey=True)
#ax1.set_title('H2O Saturation Curves')
#ax1.plot(T_list, p_sat_h2o_cc(T_list)*barye2bar, color='green', label='Constant-L+Clausius-Clapyeron P(T)')
#ax1.plot(T_list, p_sat_h2o_ice_wagner(T_list)*barye2bar, color='red', label='Ice Sublimation, Wagner+1995')
#ax1.plot(T_list, p_sat_h2o_liquid_wagner(T_list)*barye2bar, color='blue', label='Water Evaporation, Wagner+1995')
#ax1.plot(T_list, p_sat_h2o_wordsworth(T_list)*barye2bar, color='orange', label='Wordsworth Code')
#ax1.set_ylabel('Pressure (bar)')
#ax1.set_xlabel('Temperature (K)')
#ax1.set_yscale('log')
#ax1.set_ylim(1.e-11, 1.e0)
#ax1.set_xlim(150., 350.)
#ax1.legend(loc=0)

#ax2.plot(t_sat_h2o_cc(P_list), P_list*barye2bar, color='green', label='Constant-L+Clausius-Clapyeron T(P)')
#ax2.plot(t_sat_h2o_wordsworth(P_list), P_list*barye2bar, color='orange', label='Wordsworth Code')
#ax2.set_ylabel('Pressure (bar)')
#ax2.set_xlabel('Temperature (K)')
#ax2.set_yscale('log')
#ax2.legend(loc=0)

#plt.savefig('./Plots/p_sat_h2o.pdf', orientation='portrait',papertype='letter', format='pdf')

##########################
####c_p(CO2)
##########################
#c_p as a function of temperature: use Shomate Equation (see PPC pg 115)
def c_p_shomate(gas, T):
	"""
	gas: currently can take values 'n2' and 'co2'
	T: temperature in K.
	Returns: c_p in units of erg/(g*K)
	Coefficients and equation from: PPC page 115 (CO2, N2)
	http://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=1#Thermo-Gas (H2O Vapor, valid 500-1700 K)
	"""
	#kludge to prevent shomate equations from being called outside their validity range
	if type(T)==float:
		if T<150.:
			T=150.
	else:
		T[T<150.]=150.
	
	if gas=='n2':
		A, B, C, D, E=np.array([931.857, 293.529, -70.576, 5.688, 1.587])
		c_p=A+B*(T/1000.)+C*(T/1000.)**2+D*(T/1000.)**3+E*(T/1000.)**(-2) #J/(Kg*K)
	elif gas=='co2':
		A, B, C, D, E=np.array([568.122, 1254.249, -765.713, 180.645, -3.105])
		c_p=A+B*(T/1000.)+C*(T/1000.)**2+D*(T/1000.)**3+E*(T/1000.)**(-2) #J/(Kg*K)
	elif gas=='h2o':
		A, B, C, D, E=np.array([30.09200, 6.832514, 6.793435, -2.534480, 0.082139])
		c_p=A+B*(T/1000.)+C*(T/1000.)**2+D*(T/1000.)**3+E*(T/1000.)**(-2) #J/(mol*K)
		c_p=c_p/(m_h2o/amu2g*g2kg) #divide by molar mass of water to get J/Kg/K

	else:
		print 'Error: wrong value for gas in function c_p_shomate'
		
	return c_p*1e4 #convert from J/(Kg*K) to erg/(g*K)

def cp_co2_wordsworth(T):
	"""
	Takes: temperature in K.
	Returns: c_p of CO2 in units of erg/(g*K)
	Based on the code cp_CO2 in the file thermodynamics.f90, obtained from Robin Wordsworth 4/19/2016
	That code in turn was based on a polynomial fit to data from http://www.engineeringtoolbox.com/carbon-dioxide-d_974.html
	Note that we have to multiply c_p by a 1000 relative what was in Robin's code. We attribute this to the source data being in kJ/Kg/K instead of J/Kg/K
	"""
	logT=np.log(T)
	P=np.array([-0.0542, 0.9851, -2.9930])
	
	c_p_mks=(P[0]*logT**2.+P[1]*logT+P[2])*1000. #units of J/(Kg*K)
	c_p_cgs=c_p_mks*1.e4 #convert from J/(Kg*K) to erg/(g*K)
	return c_p_cgs

###Test plot
#T_list=np.arange(150., 351.) #plausible range of temperatures
#fig, (ax1)=plt.subplots(1, figsize=(8,6))
#ax1.set_title('c_p(T) for CO2')
#ax1.plot(T_list, c_p_shomate('co2', T_list)*1e-4, color='green', label='Shomate Relation')
#ax1.plot(T_list, cp_co2_wordsworth(T_list)*1e-4, color='red', label='Wordsworth Code')
#ax1.set_ylabel('c_p (J/Kg/K)')
#ax1.set_xlabel('Temperature (K)')
#ax1.legend(loc=0)
#plt.savefig('./Plots/c_p_co2.pdf', orientation='portrait',papertype='letter', format='pdf')
#plt.show()

#pdb.set_trace()
##########################
####c_p(H2O)
##########################
#c_p as a function of temperature: use Shomate Equation (see PPC pg 115)
def cp_h2o_wordsworth(T):
	"""
	Takes: temperature in K.
	Returns: c_p of H2O in units of erg/(g*K)
	Based on the code cp_H2O in the file thermodynamics.f90, obtained from Robin Wordsworth 4/19/2016
	That code in turn was based on a polynomial fit to data from http://www.engineeringtoolbox.com/carbon-dioxide-d_974.html
	Note that we have to multiply c_p by a 1000 relative what was in Robin's code. We attribute this to the source data being in kJ/Kg/K instead of J/Kg/K	
	"""
	logT=np.log(T)
	P=np.array([-0.0878, 1.9262, -13.3665, 31.6932])
	
	c_p_mks=(P[0]*logT**3.+P[1]*logT**2.+P[2]*logT+P[3])*1000. #units of J/(Kg*K)
	c_p_cgs=c_p_mks*1.e4 #convert from J/(Kg*K) to erg/(g*K)
	return c_p_cgs

###Test plot
#T_list=np.arange(150., 351.) #plausible range of temperatures
#fig, (ax1)=plt.subplots(1, figsize=(8,6))
#ax1.set_title('c_p(T) for H2O')
#ax1.plot(T_list, c_p_shomate('h2o', T_list)*1e-4, color='green', label='Shomate Relation (T>500K)')
#ax1.plot(T_list, cp_h2o_wordsworth(T_list)*1e-4, color='red', label='Wordsworth Code')
#ax1.set_ylabel('c_p (J/Kg/K)')
#ax1.set_xlabel('Temperature (K)')
#ax1.legend(loc=0)
#plt.savefig('./Plots/c_p_h2o.pdf', orientation='portrait',papertype='letter', format='pdf')


##########################
####Is CO2 ideal?
##########################
###Bottcher+2012 in Environ. Earth Science find that for p<50 kg/m**3, the ideal gas law well-calculates the link between P,T and density for CO2 in the vapor and gas phases. We test whether p=10 bar, T=150 K  which is an upper bound on density for our cases conforms to this limit

#max_density=(10.*bar2barye)/(k*150.)*m_co2*g2kg/(cm2m)**3
#print max_density #value 35...we are in the clear!!!


#plt.show()

