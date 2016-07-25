import numpy
import h5py
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import gmaps 

def printmeme(a,b,nyess):
	f = h5py.File('/home/msbandstra/radmap/singapore/2016-01-26_psa/gamma/SESSION2016-01-26T10_42_09+0800_hpge.h5','r')
	energy  = numpy.array(f['energy'])
	detector = numpy.array(f['detector_id'])
	timestampHPGE = numpy.array(f['timestamp'])
	x = numpy.array(energy)
	p = timestampHPGE[(a<x)&(x<b)]
	g= h5py.File('/home/msbandstra/radmap/singapore/gps/gpspipe_log_2016-01-26.h5','r')
	latitudes = g['latitude']
	longitudes = g['longitude']
	timestampsGPS = g['timestamps']

	pmax = numpy.amax(p)
	pmin = numpy.amin(p)
	bins = numpy.arange(int(pmin),int(pmax)+1,5)
	freq,bins = numpy.histogram(p,bins)

	bin_centers=0.5*(bins[:-1]+bins[1:])

	lat = numpy.interp(bin_centers,timestampsGPS,latitudes)
	lon = numpy.interp(bin_centers,timestampsGPS, longitudes)


	plt.figure()
	plt.title(nyess)
	gmaps.plot_autozoom(numpy.amax(lat) + 0.010, numpy.amin(lat) - 0.010, numpy.amin(lon) - 0.015, numpy.amax(lon) + 0.015,tiles=25)
	plt.scatter(lon,lat,c=freq,s=30,marker='o', vmin = 0, vmax = 17000)


printmeme(50,3000,'Gross Count Rate')
plt.show()