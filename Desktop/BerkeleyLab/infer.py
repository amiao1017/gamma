# 
import circle_bottompart as cb 
import statsmodels.api as sm 
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import normalize
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import BayesianRidge

def normalizebyrow(x):
	return normalize(x,norm= 'l1',axis=1)

array = cb.array
counts = []
times = []
countrates = []
bins = array[0][0]
answerkey_fortest = array[8][1]


for i in array:
	counts.append(i[3])
	times.append(i[2])
	countrates.append(i[1])


counts = np.ravel(np.array(counts))
times = np.ravel(np.array(times))
countrates = np.ravel(np.array(countrates))


#actually won't need this
X_1 = np.loadtxt('hists_training4.gz')
hists = np.loadtxt('hists_test4.gz')
a = np.vstack((X_1,hists))

#deleted every 101st element
histdata = np.loadtxt('hist_data.gz')
histdata1 = normalizebyrow(histdata)


#counts,times,and histdata

traininghists = histdata[100:]
testhists = histdata[:100]

trainingcounts = counts[100:]
testcounts = counts[:100]

trainingrates = countrates[100:]
testrates = countrates[:100]

trainingtimes = times[100:]
testtimes = times[:100]

# using trainingcounts and training hists use log linear
#poisson_model = sm.GLM(trainingrates,
#						sm.tools.tools.add_constant(traininghists),
#						family =sm.families.Poisson(sm.genmod.families.links.log))
#results = poisson_model.fit()
#print(results.summary())

#x = results.predict(sm.tools.tools.add_constant(testhists))


clf = BayesianRidge(compute_score=True)
clf.fit(traininghists,trainingrates)
x = clf.predict(testhists)  

answer = testrates

plt.plot(bins,x)
plt.plot(bins,answer)
plt.show()


