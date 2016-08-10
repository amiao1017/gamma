import circulartrack as ct
import h5py
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pdb


cutoff = [[.04+1.4537768715*10**9,.01+1.4537769123*10**9],[1+ 1.4537771131*10**9,.04+1.4537771697*10**9],[.2+1.4537775765*10**9, .3+1.4537776333*10**9],
[.007+1.4537778544*10**9, .0001998+1.4537779184*10**9],[.008+1.4537781861*10**9,.01+1.4537782785*10**9], [1+1.4537786903*10**9,.002+1.4537787701*10**9],
[.04+1.4537790299*10**9, .01+1.4537791085*10**9],[1+1.453779364*10**9, .015+1.4537794589*10**9],[.016+1.4537805236*10**9,.0015+1.453780584*10**9]
]
#Set up the bins using cutoff1 (just to standardize them)
def setbins():
	f = h5py.File('/home/msbandstra/radmap/singapore/2016-01-26_psa/gamma/SESSION2016-01-26T10_42_09+0800_nai.h5','r')
	energy  = np.array(f['energy'])
	detector = np.array(f['detector_id'])
	timestampHPGE = np.array(f['timestamp'])
	g= h5py.File('/home/msbandstra/radmap/singapore/gps/gpspipe_log_2016-01-26.h5','r')
	latitudes = g['latitude']
	longitudes = g['longitude']
	timestampsGPS = g['timestamps']
	p = timestampHPGE[(timestampHPGE<cutoff[0,1])&(timestampHPGE>cutoff[0,0])&(energy<3000)&(energy>50)]
	lat = np.interp(p,timestampsGPS,latitudes) 
	lon = np.interp(p,timestampsGPS,longitudes)
	vector1 = np.array([103.787671,103.787102,103.785704,103.786335])
	vector2	= np.array([1.266601,1.266958,1.264711,1.264339])
	origin = np.array([0,0])
	latandlon = ct.vectorize(lat,lon,origin)
	points_a_through_d = ct.vectorize(vector2,vector1,origin)
	distancearray = ct.pdistance(latandlon,points_a_through_d,lat,lon)
	mindp = np.amin(distancearray)
	maxdp = np.amax(distancearray)
	freq1,bins = np.histogram(distancearray,bins =100)
	return bins


def makehist(cutoff):
	f = h5py.File('/home/msbandstra/radmap/singapore/2016-01-26_psa/gamma/SESSION2016-01-26T10_42_09+0800_nai.h5','r')
	energy  = np.array(f['energy'])
	detector = np.array(f['detector_id'])
	timestampHPGE = np.array(f['timestamp'])
	g= h5py.File('/home/msbandstra/radmap/singapore/gps/gpspipe_log_2016-01-26.h5','r')
	latitudes = g['latitude']
	longitudes = g['longitude']
	timestampsGPS = g['timestamps']
	p = timestampHPGE[(timestampHPGE<cutoff[1])&(timestampHPGE>cutoff[0])&(energy<3000)&(energy>50)]
	lat = np.interp(p,timestampsGPS,latitudes) 
	lon = np.interp(p,timestampsGPS,longitudes)
	vector1 = np.array([103.787671,103.787102,103.785704,103.786335])
	vector2	= np.array([1.266601,1.266958,1.264711,1.264339])
	origin = np.array([0,0])
	latandlon = ct.vectorize(lat,lon,origin)
	points_a_through_d = ct.vectorize(vector2,vector1,origin)
	distancearray = ct.pdistance(latandlon,points_a_through_d,lat,lon)
	mindp = np.amin(distancearray)
	maxdp = np.amax(distancearray)
	
	freq,bins = np.histogram(distancearray,bins =setbins())

	times = np.interp(bins,distancearray,p)   
	time = times[1:] - times[:-1]
	countrate = freq/time
	return [bins[:-1],countrate,time,freq,times]

def plotsomestuff():
	
	plt.figure()
	plt.title('Gross Count')
	for i in xrange(9):
		y = makehist(cutoff[i])
		plt.plot(y[0],y[1])
	plt.show()

def sample():
	plt.figure()
	array = []
	for i in xrange(9):
		array.append(makehist(cutoff[i]))
	meancountrate = 1.0/9 * sum(array[0::,1])
	variance = meancountrate/8.0 * sum(1.0/np.array(array[0::,2]))
	c = np.array(array[0::,1]) - meancountrate
	sums = 0
	for i in c:
		sums += i**2
	samplevariance = sums/8.0
	stdev = samplevariance**0.5
	estdev = variance**0.5
	avgbins = firsthist[0]
	plt.plot(avgbins,stdev)
	plt.plot(avgbins,estdev)
	
	plt.show()

def subtraction(a):

	plt.figure()
	array = []
	for i in xrange(9):
		array.append(makehist(cutoff[i]))
	histarray = np.array(array)
	meancountrate = 1.0/9 * sum(histarray[0::,1])

	avgbins = firsthist[0]
	yourhist = histarray[(a-1)]
	plt.plot(avgbins,meancountrate)
	plt.plot(avgbins,yourhist[1])
	plt.title('Countrate/Distance' + str(a))
	plt.figure()
	meanvar = meancountrate/9.0 * sum(1.0/np.array(histarray[0::,2]))
	countdiff = meancountrate - yourhist[1] 
	plt.plot(avgbins,countdiff)
	stdev = (meancountrate / yourhist[2] + meanvar)**0.5
	plt.plot(avgbins, 2*stdev)
	plt.plot(avgbins,-2*stdev)	
	plt.title('Subtraction' + str(a))


array = []
for i in xrange(9):
	array.append(makehist(cutoff[i]))
array = np.array(array)


def greaterthan(n,array):
	statements =[]
	for i in array:
		t1 = i[4]
		statements.append(t1[(i[1]> n)])

	return statements

