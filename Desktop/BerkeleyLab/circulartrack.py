import numpy as np
import h5py
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def magnitude(x):
	magnitude = (x[0]*x[0]+x[1]*x[1])**0.5
	return magnitude
def magnitude2(x):
	a = x*x
	g = (np.sum(a,axis=1))**0.5
	return g

def disttoline(m,n,lat,lon):
	v = m-n
	vperp = np.array([v[1],-v[0]])
	dp = vperp[0]*(lon - m[0])+ vperp[1]*(lat-m[1])
	return np.abs(dp)/magnitude(vperp)


def vectorize(lat,lon,origin):
	length = lat.size
	xorigin = origin[0]
	yorigin = origin[1]
	xvector = np.zeros(length)
	yvector = np.zeros(length)
	x = xvector + xorigin
	y = yvector + yorigin
	newlon = lon - x
	newlat = lat - y 
	return np.dstack((newlon,newlat))[0]
	
#######################################3
f = h5py.File('/home/msbandstra/radmap/singapore/2016-01-26_psa/gamma/SESSION2016-01-26T10_42_09+0800_hpge.h5','r')
energy  = np.array(f['energy'])
detector = np.array(f['detector_id'])
timestampHPGE = np.array(f['timestamp'])
g= h5py.File('/home/msbandstra/radmap/singapore/gps/gpspipe_log_2016-01-26.h5','r')
latitudes = g['latitude']
longitudes = g['longitude']
timestampsGPS = g['timestamps']
p = timestampHPGE
lat = np.interp(p,timestampsGPS,latitudes)
lon = np.interp(p,timestampsGPS,longitudes)

##############################

vector1 = np.array([103.787671,103.787102,103.785704,103.786335])
vector2	= np.array([1.266601,1.266958,1.264711,1.264339])
origin = np.array([0,0])
latandlon = vectorize(lat,lon,origin)
points_a_through_d = vectorize(vector2,vector1,origin)

def pdistance(latandlon,points_a_through_d,lat,lon):
	a = points_a_through_d[0]
	b = points_a_through_d[1]
	c = points_a_through_d[2]
	d = points_a_through_d[3]
	ab =magnitude(a-b)
	bc= magnitude(b-c)
	cd =magnitude(c-d)
	ad =magnitude(a-d)
	pdistancea = []
	#############
	abdistance = disttoline(a,b,lat,lon)
	bcdistance = disttoline(b,c,lat,lon)
	cddistance = disttoline(c,d,lat,lon)
	addistance = disttoline(a,d,lat,lon)
	distancematrix = np.vstack((abdistance,bcdistance,cddistance,addistance)).T
	narray = distancematrix.argmin(axis =1)
	#############
	pdistancea = lat #placeholder
	pdistancea[narray == 0] = magnitude2(latandlon[narray ==0] - b) + bc + cd+ ad
	pdistancea[narray == 1] = magnitude2(latandlon[narray ==1] - c) + cd + ad
	pdistancea[narray == 2] = magnitude2(latandlon[narray ==2] - d) + ad
	pdistancea[narray == 3] = magnitude2(latandlon[narray ==3] - a)
	return np.array(pdistancea)


pdistance1 = pdistance(latandlon,points_a_through_d,lat,lon)

plt.figure()
print p.size
print pdistance1.size
plt.plot(p,pdistance1)

plt.figure()
plt.hist(pdistance1,bins=100)
plt.show()





