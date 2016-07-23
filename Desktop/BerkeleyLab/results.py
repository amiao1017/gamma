from sklearn import svm 
import numpy as np 
import cv2 
from scipy.cluster.vq import vq,kmeans,whiten
import glob,os
from PIL import Image

from sklearn.preprocessing import normalize

bagofwords = np.loadtxt('/home/russellk/Documents/bagofwords2.gz')
X_1 = np.loadtxt('hists_training3.gz')
Y_1 = np.loadtxt('training_results3.gz')
answerkey = np.loadtxt('answer_key3.gz')
hists = np.loadtxt('hists_test3.gz')


clf = svm.SVC(kernel = 'poly')
clf2 = svm.SVC()
clf3 = svm.SVC(kernel = 'linear')

#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

#Log Regression
from sklearn import linear_model 
logreg = linear_model.LogisticRegression()



def bghist(image):
	img1 = cv2.imread(image)
	img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	kp1,des1 = sift.detectAndCompute(img2,None)
	p = vq(des1,bagofwords)
	labels = p[0]
	hist,bins = np.histogram(labels,bins=np.arange(-.5,200,1))
	return hist

#normalizing columns by dividing by largest value
def normalizebycolumn(x):
	i = 0 
	newx = x
	while i < 200:
		amax = np.amax(x[:,i])
		if amax != 0:
			newx[:,i] = x[:,i]/float(amax)
		i = i+1
	return newx

#row sums to 1
def normalizebyrow(x):
	return normalize(x,norm= 'l1',axis=1)


X_2 = normalizebycolumn(X_1)
hists2 = normalizebycolumn(hists)
X_3 = normalizebyrow(X_1)
hists3 = normalizebyrow(hists)

result = gnb.fit(X_3,Y_1).predict(hists3) 
result2 = logreg.fit(X_3,Y_1).predict(hists3)
result3 = clf.fit(X_3,Y_1).predict(hists3)
result4 = clf2.fit(X_3,Y_1).predict(hists3)
result5 = clf3.fit(X_3,Y_1).predict(hists3)


