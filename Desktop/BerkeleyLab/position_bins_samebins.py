# Start and stop times of each pass
import distancefrompoint as dp
import numpy as np
import h5py
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import gmaps 

# Make sure to change SetBins accordingly (Higher for NAI)

#use distancefrompoint.py to manually find the cutoff times 
#for each stretch of the road
cutoff1 = [.01+1.4537818916*10**9,.01+1.453781999*10**9]
cutoff2 = [.0015+1.453781999*10**9,0.5+1.45378221715*10**9]
cutoff3 = [0.12+1.4537822181*10**9, .0012+1.453782328*10**9]
cutoff4 = [.0012+1.453782328*10**9, 1.0 +1.453782522*10**9]

#Set up the bins using cutoff1 (just to standardize them)
def setbins():
	f = h5py.File('/home/msbandstra/radmap/singapore/2016-01-26_psa/gamma/SESSION2016-01-26T12_09_57+0800_hpge.h5','r')
	energy  = np.array(f['energy'])
	detector = np.array(f['detector_id'])
	timestampHPGE = np.array(f['timestamp'])
	g= h5py.File('/home/msbandstra/radmap/singapore/gps/gpspipe_log_2016-01-26.h5','r')
	latitudes = g['latitude']
	longitudes = g['longitude']
	timestampsGPS = g['timestamps']
	p = timestampHPGE[(timestampHPGE<cutoff1[1])&(timestampHPGE>cutoff1[0])&(energy<3000)&(energy>50)]
	lat = np.interp(p,timestampsGPS,latitudes) 
	lon = np.interp(p,timestampsGPS,longitudes)
	pointlat = 1.268286
	pointlon = 103.784789
	distancearray = dp.distance(lat,lon, pointlat,pointlon)
	mindp = np.amin(distancearray)
	maxdp = np.amax(distancearray)
	freq1,bins = np.histogram(distancearray,bins =100)
	return bins


def makehist(cutoff):
	f = h5py.File('/home/msbandstra/radmap/singapore/2016-01-26_psa/gamma/SESSION2016-01-26T12_09_57+0800_hpge.h5','r')
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



	pointlat = 1.268286
	pointlon = 103.784789
	distancearray = dp.distance(lat,lon, pointlat,pointlon)

	mindp = np.amin(distancearray)
	maxdp = np.amax(distancearray)
	#plt.figure()
	#plt.plot(p,distancearray)
	#plt.title('distance v time')
	
	freq,bins = np.histogram(distancearray,bins =setbins())

	times = np.interp(bins,distancearray,p) 
	time = times[1:] - times[:-1]
	countrate = freq/time
	return [bins[:-1],countrate,time,freq,times]





def plotsomestuff():
	plt.figure()
	plt.title('Total count rate, binned by distance')
	firsthist = makehist(cutoff1)
	#makehist(cutoff2)
	secondhist = makehist(cutoff3)
	plt.plot(firsthist[0],firsthist[1])
	plt.plot(secondhist[0],secondhist[1])
	#makehist(cutoff4)
	plt.figure()
	plt.title('subtraction')
	countdiff = secondhist[1]-firsthist[1]
	avgbins = (firsthist[0] + secondhist[0])/2
	plt.plot(avgbins,countdiff)
	stdev = (0.5*(secondhist[2])**(-2)*secondhist[3]+0.5*(firsthist[2])**(-2)*firsthist[3]+(secondhist[2]*firsthist[2])**(-1)*(firsthist[3]+secondhist[3]))**0.5
	plt.plot(avgbins, 2*stdev)
	plt.plot(avgbins,-2*stdev)	

	plt.show()


plotsomestuff()


def outliers():
	firsthist = makehist(cutoff1)
	secondhist = makehist(cutoff3)
	countdiff = secondhist[1]-firsthist[1]
	stdev = (0.5*(secondhist[2])**(-2)*secondhist[3]+0.5*(firsthist[2])**(-2)*firsthist[3]+(secondhist[2]*firsthist[2])**(-1)*(firsthist[3]+secondhist[3]))**0.5
	t1 = secondhist[4]
	t2 = firsthist[4]
	statements1 = t1[(countdiff**2)**(0.5) > 2*stdev]
	statements2 = t2[(countdiff**2)**(0.5) > 2*stdev]
	return [statements1,statements2]

for i in outliers()[0]:
	print i
for k in outliers()[1]:
	print k






