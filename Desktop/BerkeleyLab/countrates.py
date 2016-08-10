
import numpy as np
import h5py
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt 


f = h5py.File('/home/msbandstra/radmap/singapore/2016-01-26_psa/gamma/SESSION2016-01-26T10_42_09+0800_nai.h5','r')
energy  = np.array(f['energy'])
detector = np.array(f['detector_id'])
timestamp = np.array(f['timestamp'])
x = np.array(energy)
# i = np.array((np.round(x)==1460).nonzero())[0]

def countratehist(element,cutoff):
	p1 = timestamp[(((cutoff[0]<x)&(x<cutoff[1]))|((1127<x)&(x<1137)))]
	pmax1 = np.amax(p1)
	pmin1 = np.amin(p1)
	plt.figure()
	plt.title(element)
	plt.hist(p1,bins = np.arange(int(pmin1),int(pmax1)+1,2))

plt.show()


