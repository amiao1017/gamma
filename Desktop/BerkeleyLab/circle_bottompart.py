import circulartrack as ct
import h5py
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pdb


cutoff1 = [.04+1.4537768715*10**9,.01+1.4537769123*10**9]
cutoff2 = [1+ 1.4537771131*10**9,.04+1.4537771697*10**9]
cutoff3 = [.2+1.4537775765*10**9, .3+1.4537776333*10**9]
cutoff4 = [.007+1.4537778544*10**9, .0001998+1.4537779184*10**9]
cutoff5 = [.008+1.4537781861*10**9,.01+1.4537782785*10**9]
cutoff6 = [1+1.4537786903*10**9,.002+1.4537787701*10**9]
cutoff7 = [.04+1.4537790299*10**9, .01+1.4537791085*10**9]
cutoff8 = [1+1.453779364*10**9, .015+1.4537794589*10**9]
cutoff9 = [.016+1.4537805236*10**9,.0015+1.453780584*10**9]

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
	p = timestampHPGE[(timestampHPGE<cutoff1[1])&(timestampHPGE>cutoff1[0])&(energy<3000)&(energy>50)]
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
	#plt.title('Total count rate, binned by distance')
	firsthist = makehist(cutoff1)
	secondhist = makehist(cutoff2)
	thirdhist = makehist(cutoff3)
	fourthhist = makehist(cutoff4)
	fifthhist = makehist(cutoff5)
	sixhist = makehist(cutoff6)
	sevenhist = makehist(cutoff7)
	eighthist = makehist(cutoff8)
	ninehist = makehist(cutoff9)
	plt.plot(firsthist[0],firsthist[1])
	plt.plot(secondhist[0],secondhist[1])
	plt.plot(thirdhist[0],thirdhist[1])
	plt.plot(fourthhist[0],fourthhist[1])
	plt.plot(fifthhist[0],fifthhist[1])
	plt.plot(sixhist[0],sixhist[1])
	plt.plot(sevenhist[0],sevenhist[1])
	plt.plot(eighthist[0],eighthist[1])
	plt.plot(ninehist[0],ninehist[1])
	plt.show()

#plotsomestuff()

def sample():

	plt.figure()
	firsthist = makehist(cutoff1)
	secondhist = makehist(cutoff2)
	thirdhist = makehist(cutoff3)
	fourthhist = makehist(cutoff4)
	fifthhist = makehist(cutoff5)
	sixhist = makehist(cutoff6)
	sevenhist = makehist(cutoff7)
	eighthist = makehist(cutoff8)
	ninehist = makehist(cutoff9)
	meancountrate = (secondhist[1] + firsthist[1] + thirdhist[1] + fourthhist[1] + fifthhist[1] + sixhist[1] + sevenhist[1] + eighthist[1] + ninehist[1])/9
	variance = (meancountrate/secondhist[2] + meancountrate/firsthist[2] + meancountrate/thirdhist[2] + meancountrate/fourthhist[2] + meancountrate/fifthhist[2] + meancountrate/sixhist[2] + meancountrate/sevenhist[2] + meancountrate/eighthist[2] + meancountrate/ninehist[2])/8
	samplevariance = ((secondhist[1] -meancountrate)**2 + (firsthist[1]-meancountrate)**2 + (thirdhist[1]-meancountrate)**2 + (fourthhist[1]-meancountrate)**2 + (fifthhist[1]-meancountrate)**2 + (sixhist[1]-meancountrate)**2 + (sevenhist[1]-meancountrate)**2 + (eighthist[1]-meancountrate)**2 + (ninehist[1]-meancountrate)**2)/8
	stdev = samplevariance**0.5
	estdev = variance**0.5
	avgbins = firsthist[0]
	plt.plot(avgbins,stdev)
	plt.plot(avgbins,estdev)
	
	plt.show()

#sample()

def subtraction(a):

	plt.figure()
	firsthist = makehist(cutoff1)
	secondhist = makehist(cutoff2)
	thirdhist = makehist(cutoff3)
	fourthhist = makehist(cutoff4)
	fifthhist = makehist(cutoff5)
	sixhist = makehist(cutoff6)
	sevenhist = makehist(cutoff7)
	eighthist = makehist(cutoff8)
	ninehist = makehist(cutoff9)
	histarray = [firsthist,secondhist,thirdhist,fourthhist,fifthhist,sixhist,sevenhist,eighthist,ninehist]
	meancountrate = (secondhist[1] + firsthist[1] + thirdhist[1] + fourthhist[1] + fifthhist[1] + sixhist[1] + sevenhist[1] + eighthist[1] + ninehist[1])/9
	avgbins = firsthist[0]
	yourhist = histarray[(a-1)]
	plt.plot(avgbins,meancountrate)
	plt.plot(avgbins,yourhist[1])
	plt.title('Countrate/Distance' + str(a))
	plt.figure()
	meanvar = meancountrate/9.0 * (1/firsthist[2]+1/secondhist[2]+1/thirdhist[2]+1/fourthhist[2]+1/fifthhist[2]+1/sixhist[2]+1/sevenhist[2]+1/eighthist[2]+1/ninehist[2])
	countdiff = meancountrate - yourhist[1] 
	plt.plot(avgbins,countdiff)
	stdev = (meancountrate / yourhist[2] + meanvar)**0.5
	plt.plot(avgbins, 2*stdev)
	plt.plot(avgbins,-2*stdev)	
	plt.title('Subtraction' + str(a))


#subtraction(1)
#subtraction(2)
#subtraction(3)
#subtraction(4)
#subtraction(5)
#subtraction(6)
#subtraction(7)
#subtraction(8)
#subtraction(9)
#plt.show()

firsthist = makehist(cutoff1)
secondhist = makehist(cutoff2)
thirdhist = makehist(cutoff3)
fourthhist = makehist(cutoff4)
fifthhist = makehist(cutoff5)
sixhist = makehist(cutoff6)
sevenhist = makehist(cutoff7)
eighthist = makehist(cutoff8)
ninehist = makehist(cutoff9)

array = [firsthist,secondhist,thirdhist,fourthhist,fifthhist,sixhist,sevenhist,eighthist,ninehist]

def outliers(firsthist,secondhist,thirdhist,fourthhist,fifthhist,sixhist,sevenhist,eighthist,ninehist):
	
	meancountrate = (secondhist[1] + firsthist[1] + thirdhist[1] + fourthhist[1] + fifthhist[1] + sixhist[1] + sevenhist[1] + eighthist[1] + ninehist[1])/9
	meanvar = meancountrate/9.0 * (1/firsthist[2]+1/secondhist[2]+1/thirdhist[2]+1/fourthhist[2]+1/fifthhist[2]+1/sixhist[2]+1/sevenhist[2]+1/eighthist[2]+1/ninehist[2])

	#1
	countdiff1 = meancountrate-firsthist[1]
	stdev = (meanvar+meancountrate / firsthist[2])**0.5
	t1 = firsthist[4]
	statements1 = t1[(countdiff1 > 5*stdev)]
	#2
	countdiff2 = meancountrate-secondhist[1]
	stdev = (meanvar+meancountrate / secondhist[2])**0.5
	t2 = secondhist[4]
	statements2 = t2[(countdiff2 > 5*stdev)]
	#3
	countdiff3 = meancountrate-thirdhist[1]
	stdev = (meanvar+meancountrate / thirdhist[2])**0.5
	t3 = thirdhist[4]
	statements3 = t3[(countdiff3 > 5*stdev)]
	#4
	countdiff4 = meancountrate-fourthhist[1]
	stdev = (meanvar+ meancountrate / fourthhist[2])**0.5
	t4 = fourthhist[4]
	statements4 = t4[(countdiff4 > 5*stdev)]
	#5 
	countdiff5 = meancountrate-fifthhist[1]
	stdev = (meanvar+ meancountrate / fifthhist[2])**0.5
	t5 = fifthhist[4]
	statements5 = t5[(countdiff5 > 5*stdev)]
	#6
	countdiff6 = meancountrate-sixhist[1]
	stdev = (meanvar+ meancountrate / sixhist[2])**0.5
	t6 = sixhist[4]
	statements6 = t6[(countdiff6 > 5*stdev)]
	#7
	countdiff7 = meancountrate-sevenhist[1]
	stdev = (meanvar+meancountrate / sevenhist[2])**0.5
	t7 = sevenhist[4]
	statements7 = t7[(countdiff7> 5*stdev)]
	#8
	countdiff8 = meancountrate-eighthist[1]
	stdev = (meanvar+meancountrate / eighthist[2])**0.5
	t8 = eighthist[4]
	statements8 = t8[(countdiff8 > 5*stdev)]
	#9
	countdiff9 = meancountrate-ninehist[1]
	stdev = (meanvar+meancountrate / ninehist[2])**0.5
	t9 = ninehist[4]
	statements9 = t9[(countdiff9 > 5*stdev)]
	

	return [statements1,statements2,statements3,statements4,statements5,statements6,statements7,statements8,statements9]


def greaterthan(n,array):
	statements =[]
	for i in array:
		t1 = i[4]
		statements.append(t1[(i[1]> n)])
	

	return statements


def outliers2(firsthist,secondhist,thirdhist,fourthhist,fifthhist,sixhist,sevenhist,eighthist,ninehist,a):
	
	meancountrate = (secondhist[1] + firsthist[1] + thirdhist[1] + fourthhist[1] + fifthhist[1] + sixhist[1] + sevenhist[1] + eighthist[1] + ninehist[1])/9
	meanvar = meancountrate/9.0 * (1/firsthist[2]+1/secondhist[2]+1/thirdhist[2]+1/fourthhist[2]+1/fifthhist[2]+1/sixhist[2]+1/sevenhist[2]+1/eighthist[2]+1/ninehist[2])
	final =[]
	#1
	countdiff1 = meancountrate-firsthist[1]
	stdev = (meanvar+meancountrate / firsthist[2])**0.5
	t1 = firsthist[4]
	final.append((countdiff1 > a*stdev))
	#2
	countdiff2 = meancountrate-secondhist[1]
	stdev = (meanvar+meancountrate / secondhist[2])**0.5
	t2 = secondhist[4]
	final.append((countdiff2 > a*stdev))
	#3
	countdiff3 = meancountrate-thirdhist[1]
	stdev = (meanvar+meancountrate / thirdhist[2])**0.5
	t3 = thirdhist[4]
	final.append((countdiff3>a*stdev))
	#4
	countdiff4 = meancountrate-fourthhist[1]
	stdev = (meanvar+ meancountrate / fourthhist[2])**0.5
	t4 = fourthhist[4]
	final.append((countdiff4 > a*stdev))
	#5 
	countdiff5 = meancountrate-fifthhist[1]
	stdev = (meanvar+ meancountrate / fifthhist[2])**0.5
	t5 = fifthhist[4]
	final.append((countdiff5 > a*stdev))
	#6
	countdiff6 = meancountrate-sixhist[1]
	stdev = (meanvar+ meancountrate / sixhist[2])**0.5
	t6 = sixhist[4]
	final.append((countdiff6 > a*stdev))
	#7
	countdiff7 = meancountrate-sevenhist[1]
	stdev = (meanvar+meancountrate / sevenhist[2])**0.5
	t7 = sevenhist[4]
	final.append((countdiff7> a*stdev))
	#8
	countdiff8 = meancountrate-eighthist[1]
	stdev = (meanvar+meancountrate / eighthist[2])**0.5
	t8 = eighthist[4]
	final.append((countdiff8>a*stdev))
	#9
	countdiff9 = meancountrate-ninehist[1]
	stdev = (meanvar+meancountrate / ninehist[2])**0.5
	t9 = ninehist[4]
	final.append((countdiff9>a*stdev))
	

	return final

outl = outliers2(firsthist,secondhist,thirdhist,fourthhist,fifthhist,sixhist,sevenhist,eighthist,ninehist,1.0)
a = np.ravel(np.array(outl))
b = np.loadtxt('vehiclelocations.gz')

print (a[b==1]).size
print sum(a[b==1])

outl = outliers2(firsthist,secondhist,thirdhist,fourthhist,fifthhist,sixhist,sevenhist,eighthist,ninehist,1.5)
a = np.ravel(np.array(outl))
b = np.loadtxt('vehiclelocations.gz')

print (a[b==1]).size
print sum(a[b==1])


outl = outliers2(firsthist,secondhist,thirdhist,fourthhist,fifthhist,sixhist,sevenhist,eighthist,ninehist,2.0)
a = np.ravel(np.array(outl))
b = np.loadtxt('vehiclelocations.gz')

print (a[b==1]).size
print sum(a[b==1])

outl = outliers2(firsthist,secondhist,thirdhist,fourthhist,fifthhist,sixhist,sevenhist,eighthist,ninehist,2.5)
a = np.ravel(np.array(outl))
b = np.loadtxt('vehiclelocations.gz')

print (a[b==1]).size
print sum(a[b==1])

outl = outliers2(firsthist,secondhist,thirdhist,fourthhist,fifthhist,sixhist,sevenhist,eighthist,ninehist,3.0)
a = np.ravel(np.array(outl))
b = np.loadtxt('vehiclelocations.gz')

print (a[b==1]).size
print sum(a[b==1])

outl = outliers2(firsthist,secondhist,thirdhist,fourthhist,fifthhist,sixhist,sevenhist,eighthist,ninehist,3.5)
a = np.ravel(np.array(outl))
b = np.loadtxt('vehiclelocations.gz')

print (a[b==1]).size
print sum(a[b==1])

outl = outliers2(firsthist,secondhist,thirdhist,fourthhist,fifthhist,sixhist,sevenhist,eighthist,ninehist,4.0)
a = np.ravel(np.array(outl))
b = np.loadtxt('vehiclelocations.gz')

print (a[b==1]).size
print sum(a[b==1])

outl = outliers2(firsthist,secondhist,thirdhist,fourthhist,fifthhist,sixhist,sevenhist,eighthist,ninehist,4.5)
a = np.ravel(np.array(outl))
b = np.loadtxt('vehiclelocations.gz')

print (a[b==1]).size
print sum(a[b==1])

outl = outliers2(firsthist,secondhist,thirdhist,fourthhist,fifthhist,sixhist,sevenhist,eighthist,ninehist,5.0)
a = np.ravel(np.array(outl))
b = np.loadtxt('vehiclelocations.gz')

print (a[b==1]).size
print sum(a[b==1])