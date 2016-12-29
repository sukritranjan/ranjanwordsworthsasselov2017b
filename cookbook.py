# -*- coding: iso-8859-1 -*-

def rebin_factor( a, newshape ):
	  '''Rebin an array to a new shape.
	  newshape must be a factor of a.shape.
	  
	  Source: http://wiki.scipy.org/Cookbook/Rebinning
	  '''
	  assert len(a.shape) == len(newshape)
	  assert not sometrue(mod( a.shape, newshape ))

	  slices = [ slice(None,None, old/new) for old,new in zip(a.shape,newshape) ]
	  return a[slices]
        
def rebin(a, *args):
	'''rebin ndarray data into a smaller ndarray of the same rank whose dimensions
	are factors of the original dimensions. eg. An array with 6 columns and 4 rows
	can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
	example usages:
	>>> a=rand(6,4); b=rebin(a,3,2)
	>>> a=rand(6); b=rebin(a,2)
	
	Source: http://wiki.scipy.org/Cookbook/Rebinning
	'''
	from numpy import *
	shape = a.shape
	lenShape = len(shape)
	factor = asarray(shape)/asarray(args)
	evList = ['a.reshape('] + \
		['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
		[')'] + ['.mean(%d)'%(i+1) for i in range(lenShape)]
	print ''.join(evList)
	return eval(''.join(evList))
	
	

def congrid(a, newdims, method='linear', centre=False, minusone=False):
	'''Arbitrary resampling of source array to new dimension sizes.
	Currently only supports maintaining the same number of dimensions.
	To use 1-D arrays, first promote them to shape (x,1).
	
	Uses the same parameters and creates the same co-ordinate lookup points
	as IDL''s congrid routine, which apparently originally came from a VAX/VMS
	routine of the same name.

	method:
	neighbour - closest value from original data
	nearest and linear - uses n x 1-D interpolations using
			    scipy.interpolate.interp1d
	(see Numerical Recipes for validity of use of n 1-D interpolations)
	spline - uses ndimage.map_coordinates

	centre:
	True - interpolation points are at the centres of the bins
	False - points are at the front edge of the bin

	minusone:
	For example- inarray.shape = (i,j) & new dimensions = (x,y)
	False - inarray is resampled by factors of (i/x) * (j/y)
	True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
	This prevents extrapolation one element beyond bounds of input array.
	
	Source: http://wiki.scipy.org/Cookbook/Rebinning
	'''
	import numpy as n
	import scipy.interpolate
	import scipy.ndimage

	
	if not a.dtype in [n.float64, n.float32]:
	      a = n.cast[float](a)

	m1 = n.cast[int](minusone)
	ofs = n.cast[int](centre) * 0.5
	old = n.array( a.shape )
	ndims = len( a.shape )
	if len( newdims ) != ndims:
		print "[congrid] dimensions error. " \
		      "This routine currently only support " \
		      "rebinning to the same number of dimensions."
		return None
	newdims = n.asarray( newdims, dtype=float )
	dimlist = []

	if method == 'neighbour':
		for i in range( ndims ):
		    base = n.indices(newdims)[i]
		    dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
				    * (base + ofs) - ofs )
		cd = n.array( dimlist ).round().astype(int)
		newa = a[list( cd )]
		return newa

	elif method in ['nearest','linear']:
		# calculate new dims
		for i in range( ndims ):
			base = n.arange( newdims[i] )
			dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
				    * (base + ofs) - ofs )
		# specify old dims
		olddims = [n.arange(i, dtype = n.float) for i in list( a.shape )]

		# first interpolation - for ndims = any
		mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
		newa = mint( dimlist[-1] )

		trorder = [ndims - 1] + range( ndims - 1 )
		for i in range( ndims - 2, -1, -1 ):
			newa = newa.transpose( trorder )

			mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
			newa = mint( dimlist[i] )

		if ndims > 1:
		      # need one more transpose to return to original dimensions
		      newa = newa.transpose( trorder )

		return newa
	elif method in ['spline']:
		oslices = [ slice(0,j) for j in old ]
		oldcoords = n.ogrid[oslices]
		nslices = [ slice(0,j) for j in list(newdims) ]
		newcoords = n.mgrid[nslices]

		newcoords_dims = range(n.rank(newcoords))
		#make first index last
		newcoords_dims.append(newcoords_dims.pop(0))
		newcoords_tr = newcoords.transpose(newcoords_dims)
		# makes a view that affects newcoords

		newcoords_tr += ofs

		deltas = (n.asarray(old) - m1) / (newdims - m1)
		newcoords_tr *= deltas

		newcoords_tr -= ofs

		newa = scipy.ndimage.map_coordinates(a, newcoords)
		return newa
	else:
		print "Congrid error: Unrecognized interpolation type.\n", \
		      "Currently only \'neighbour\', \'nearest\',\'linear\',", \
		      "and \'spline\' are supported."
		return None


def rebin_uneven(lefts_old, rights_old, vals_old, lefts_new, rights_new):
	"""
	Purpose of this script is to rebin a histogram onto a new x-axis. It's goal is to accomodate irregular bins. 
	
	This method AVERAGES bins together. So if you have two bins containing the numbers 2 and 3 and want to rebin them into one bin, the value of that bin would be 2.5. This is well suited to dealing with things that are rates. For example, suppose you have those two bins. Suppose each contains the flux in ergs/cm2/s/nm: 10 and 20 for 200-210 and 210-220 nm, respectively. Then if you want the flux average from 200-220 nm, you need to compute (10*10+20*10)/(10+10)=15 erg/cm2/s/nm. 
	
	lefts_old=left edges of old bins
	rights_old=right edges of old bins
	vals_old=values of old bins
	lefts_new=left edges of new bins
	rights_new=right edges of new bins
	
	returns: values of new bins
	Uses linear interpolation to split up bins.
	ASSUMES: abscissa monotonically increasing ###
	The bins defined by lefts_new and rights_new must be entirely enclosed in the bins defined by left_old and right_old
	"""
	import numpy as np
	import pdb
	vals_new=np.zeros(np.shape(lefts_new))
	binwidths_old=rights_old-lefts_old#widths of old bins. May be nonuniform.
	
	for ind in range(0, len(vals_new)):
		#Loop over each of the new bins
		val_new=0. #initialize the value of the new bin to zero.
		left_new=lefts_new[ind]
		right_new=rights_new[ind]

		#is the new bin contained entirely within an existing bin?
		fullycontained_ind=np.where((left_new>=lefts_old) & (right_new<=rights_old)) #if there is a bin for which both these conditions hold, then this bin is contained entirely within it.

		if np.size(fullycontained_ind)>0: #if such a bin exists...
			val_new=vals_old[fullycontained_ind]# then the value of the new bin is just the enclosing old bin.
		else: #otherwise, we know that the new bin must span at least 1 junction of bins, meaning there will be a left partial part and a right partial part
			val_new_numerator=0 #adding up multiple contributions so need to keep track of the numerators and denominators of the weighted sum.
			val_new_denominator=0
			
			#Given that we are not wholely contained within a single bin, there may exist a partial bin enclosed by the left edge, i.e. left_edge must be in the middle of some bin. Only case in which this is *not* true is if the left edge _perfectly_ overlaps with one of the old bin edges.
			leftedgemiddle_ind=np.where((left_new>lefts_old) & (left_new<rights_old))
			if np.size(leftedgemiddle_ind)>0: #This condition is met as long as the left edge does not perfectly overlap with another edge
				val_new_numerator=val_new_numerator+vals_old[leftedgemiddle_ind]*(rights_old[leftedgemiddle_ind]-left_new)
				val_new_denominator=val_new_denominator+rights_old[leftedgemiddle_ind]-left_new
			
			#As there must be a left edge in the middle of a bin somwhere, there must be a right edge in the middle of a bin somewhere, unless it perfectly overlaps with an old edge.
			rightedgemiddle_ind=np.where((right_new<rights_old) & (right_new>lefts_old))
			if np.size(rightedgemiddle_ind)>0: #condition met as long as there is no perfect overlap in the edges
				val_new_numerator=val_new_numerator+vals_old[rightedgemiddle_ind]*(right_new-lefts_old[rightedgemiddle_ind])
				val_new_denominator=val_new_denominator+right_new-lefts_old[rightedgemiddle_ind]
			
			#Lastly, there may exist whole old bins encapsulated within the new bin. Add those in as well.
			oldencapsulatedinds=np.where((left_new<=lefts_old) & (right_new>=rights_old)) #indices of old bins which are completely encapsulated by the new bins. Note this captures the cases where a new edge aligns with an old edge
			if np.size(oldencapsulatedinds) > 0: #if such encapsulated bins exist then...
			      val_new_numerator=val_new_numerator+np.sum(vals_old[oldencapsulatedinds]*binwidths_old[oldencapsulatedinds])
			      val_new_denominator=val_new_denominator+np.sum(binwidths_old[oldencapsulatedinds])
			
			val_new=val_new_numerator/val_new_denominator
		vals_new[ind]=val_new
		#pdb.set_trace()
	return vals_new

#"""Test the uneven rebinning function we have written. """
#import numpy as np

#########First family of tests: evenly-binned input (old data)
#lefts_old=np.array([0., 1., 2., 3., 4., 5.])
#rights_old=np.array([1., 2., 3., 4., 5., 6.])
#vals_old=np.array([100., 110., 120., 130., 140., 150.]) 

##First test: ensure we recover the same bins if we return the same edges
#lefts_new=lefts_old
#rights_new=rights_old
#vals_new_1=rebin_uneven(lefts_old, rights_old, vals_old, lefts_new, rights_new)
#print vals_new_1-vals_old #should be 0 if we did it right

##Second test: bin down the data by a factor of two.
#lefts_new=np.array([0., 2., 4.])
#rights_new=np.array([2., 4., 6.])
#vals_new_2=rebin_uneven(lefts_old, rights_old, vals_old, lefts_new, rights_new)
#print vals_new_2-np.array([105., 125., 145.]) #should be 0 if we did it right

##Third test: evenly spaced values, but displaced from edges
#lefts_new=np.array([0.5, 1.5, 2.5, 3.5, 4.5])
#rights_new=np.array([1.5, 2.5, 3.5, 4.5, 5.5])
#vals_new_3=rebin_uneven(lefts_old, rights_old, vals_old, lefts_new, rights_new)
#print vals_new_3-np.array([105., 115., 125., 135., 145.])

##Fourth test: evenly spaced values, displaced from edges, reduce resolution by factor of 2
#lefts_new=np.array([0.7, 2.7])
#rights_new=np.array([2.7, 4.7])
#vals_new_4=rebin_uneven(lefts_old, rights_old, vals_old, lefts_new, rights_new)
#print vals_new_4-np.array([(0.3*100+1.0*110.+0.7*120.)/(0.3+1.0+0.7), (0.3*120+1.0*130+0.7*140.)/(0.3+1+0.7)]) #Test reveals numpy floating-point errors at the <1e-14 level.

##Fifth test: unevenly spaced values
#lefts_new=np.array([0.0, 1.0, 1.3, 1.5, 2.0, 3.5]) #same, left overlap/right in same bin, same bin, left in same bin/right overlap, left overlap/right in next bin, left in one bin/right overlap in next bin.
#rights_new=np.array([1.0,1.3, 1.5, 2.0, 3.5, 5.0])
#vals_new_5=rebin_uneven(lefts_old, rights_old, vals_old, lefts_new, rights_new)
#print vals_new_5-np.array([100.,110, 110, 110., (120+0.5*130)/1.5, (0.5*130.+140)/1.5])

#lefts_new=np.array([0.3, 1.4]) #both in different bins, both in different bins with multiple bins in the middle
#rights_new=np.array([1.4, 4.1])
#vals_new_5=rebin_uneven(lefts_old, rights_old, vals_old, lefts_new, rights_new)
#print vals_new_5-np.array([(0.7*100+0.4*110)/(1.1), (0.6*110+120+130+0.1*140)/(0.6+2+0.1)]) #Test reveals numpy floating-point errors at the <1e-14 level.




###########Second family of tests: unevenly-binned input
#lefts_old=np.array([0., 1., 3., 3.5,4.5])
#rights_old=np.array([1.,3., 3.5,4.5, 6.])
#vals_old=np.array([100., 115., 130., 135., 146.666666667]) 

##First test: even output
#lefts_new=np.array([0., 1., 2., 3., 4., 5.])
#rights_new=np.array([1., 2., 3., 4., 5., 6.])
#vals_new_6=rebin_uneven(lefts_old, rights_old, vals_old, lefts_new, rights_new)
#shouldbe_6=np.array([100., 115., 115., (0.5*130+0.5*135), (0.5*135.+0.5*146.666666667), 146.666666667])
#print vals_new_6-shouldbe_6
##print np.sum(vals_old*(rights_old-lefts_old))-np.sum(vals_new_6) #should equal zero if everything conserved.

##Last test: uneven input, uneven output
#lefts_old=np.array
#lefts_old=np.array([ 0., 1., 3.,  4.5, 4.9, 5.5, 6.0])
#rights_old=np.array([1., 3., 4.5, 4.9, 5.5, 6.0, 10.])
#vals_old=np.array([  0, 10., 20., 30., 40,  50,  60.]) 

#lefts_new=np.array([ 0.5, 1.5, 3.1, 3.9, 5.0, 7.5])
#rights_new=np.array([1.5, 3.1, 3.9, 5.0, 7.5, 10.])
#vals_new_7=rebin_uneven(lefts_old, rights_old, vals_old, lefts_new, rights_new)
#shouldbe_7=np.array([(0.5*0+0.5*10), (1.5*10+0.1*20)/1.6, 20., (0.6*20.+0.4*30+0.1*40.)/(1.1), (0.5*40+0.5*50+1.5*60)/2.5, 60.])
#print vals_new_7-shouldbe_7

def get_bin_edges(x_centers):
	"""
	This function is used to generate bin edges for histogram-type data for which only the bin centers are available (e.g., data extracted from plots). It does so using the following algorithm:
	-The boundary between points x_i and x_i+1 is set at the midpoint between them.
	-The left edge of bin x_0 is equidistant from the center of the bin as the right edge
	-The right edge of bin x_n is equidistant from the center of the bin as the left edge
	
	Input: x_centers, bin centers
	
	Output: x_lefts, x_rights
	"""
	import numpy as np
	x_lefts=np.zeros(np.shape(x_centers))
	x_rights=np.zeros(np.shape(x_centers))
	
	for ind in range(0, len(x_centers)-1):
		boundaryval=0.5*(x_centers[ind]+x_centers[ind+1])
		
		x_lefts[ind+1]=boundaryval
		x_rights[ind]=boundaryval
	
	x_lefts[0]=x_centers[0]-(x_rights[0]-x_centers[0])
	x_rights[-1]=x_centers[-1]+(x_centers[-1]-x_lefts[-1])
	
	return x_lefts, x_rights

def smooth(x,window_len, window):
	"""
	From http://wiki.scipy.org/Cookbook/SignalSmooth
	smooth the data using a window with requested size.
	
	This method is based on the convolution of a scaled window with the signal.
	The signal is prepared by introducing reflected copies of the signal 
	(with the window size) in both ends so that transient parts are minimized
	in the begining and end part of the output signal.
	
	input:
	    x: the input signal 
	    window_len: the dimension of the smoothing window; should be an odd integer
	    window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
		flat window will produce a moving average smoothing.

	output:
	    the smoothed signal
	    
	example:

	t=linspace(-2,2,0.1)
	x=sin(t)+randn(len(t))*0.1
	y=smooth(x)
	
	see also: 
	
	numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
	scipy.signal.lfilter
    
	TODO: the window parameter could be the window itself if an array instead of a string
	NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
	"""
	import numpy

	if x.ndim != 1:
		raise ValueError, "smooth only accepts 1 dimension arrays."

	if x.size < window_len:
		raise ValueError, "Input vector needs to be bigger than window size."


	if window_len<3:
		return x


	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


	s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
	#print(len(s))
	if window == 'flat': #moving average
		w=numpy.ones(window_len,'d')
	else:
		w=eval('numpy.'+window+'(window_len)')

	y=numpy.convolve(w/w.sum(),s,mode='valid')
	return y

def movingaverage(interval, window_size):
	"""
	Moving average smoothing
	See http://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python/11352216#11352216 for details
	"""
	import numpy
	window = numpy.ones(int(window_size))/float(window_size)
	return numpy.convolve(interval, window, 'same')
