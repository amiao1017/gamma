
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

print np.amax(timestamp)

p1 = timestamp[(((1168<x)&(x<1178))|((1127<x)&(x<1137)))]
pmax1 = np.amax(p1)
pmin1 = np.amin(p1)
plt.figure()
plt.title('Co60')
plt.hist(p1,bins = np.arange(int(pmin1),int(pmax1)+1,2))


p2 = timestamp[((657<x)&(x<667))]
pmax2 = np.amax(p2)
pmin2 = np.amin(p2)
plt.figure()
plt.title('Cs137')
plt.hist(p2,bins = np.arange(int(pmin2),int(pmax2)+1,2))

p3 = timestamp
pmax3 = np.amax(p3)
pmin3 = np.amin(p3)
plt.figure()
plt.title('total counts')
plt.hist(p3,bins = np.arange(int(pmin3),int(pmax3)+1,2))


p4 = timestamp[((1455<x)&(x<1465))]
pmax4 = np.amax(p4)
pmin4 = np.amin(p4)
plt.figure()
plt.title('K40')
plt.hist(p4,bins = np.arange(int(pmin4),int(pmax4)+1,2))

p5 = timestamp[(((2609<x)&(x<2619)))]
pmax5 = np.amax(p5)
pmin5 = np.amin(p5)
plt.figure()
plt.title('Tl208')
plt.hist(p5,bins = np.arange(int(pmin5),int(pmax5)+1,2))

p6 = timestamp[(((2200<x)&(x<2210)))]
pmax6 = np.amax(p6)
pmin6 = np.amin(p6)
plt.figure()
plt.title('Bi214')
plt.hist(p6,bins = np.arange(int(pmin6),int(pmax6)+1,2))


plt.show()


