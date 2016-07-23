from sklearn import svm 
import numpy as np 
import cv2 
import dates
from scipy.cluster.vq import vq,kmeans,whiten
import glob,os
from PIL import Image

directore = "/home/msbandstra/radmap/singapore/2016-01-26_psa/pi/SESSION2016-01-26T10_42_09+0800"
bagofwords = np.loadtxt('/home/russellk/Documents/bagofwords2.gz')
sift = cv2.SIFT()
clf = svm.SVC(kernel = 'linear')
imgse  = np.array(dates.gothroughtimes())

def bghist(image):
	img1 = cv2.imread(image)
	img2 = img1[360:480,:]
	kp1,des1 = sift.detectAndCompute(img2,None)
	p = vq(des1,bagofwords)
	labels = p[0]
	hist,bins = np.histogram(labels,bins=np.arange(-.5,200,1))
	return hist


def getimages(): #training data
	testimages = np.ravel(imgse)
	training  = []
	everything = []
	for file in glob.glob(directore+'/*'):
		everything.append(file)
		if file not in testimages:
			training.append(file)
	return [np.array(training),np.array(everything),testimages]



#do bghist on each training image then use SVC to train  

#training function
def train(data):
	responses =[]
	for i in data:
		img =Image.open(i)
		img.show()
		resp = input("Categorize as 1 or 0:")
		responses.append(resp)
	return responses

def hists(data):
	a = []
	for i in data:
		a.append(bghist(i))
	return np.array(a)



# second part of the loop
#os.chdir('/home/russellk/Documents')
#training = np.genfromtxt('/home/russellk/Documents/trainingdata.gz',dtype ='str')
#newtraining = []
#for i in training:
#	newtraining.append(directore+'/'+i)
#training = np.array(newtraining)
#########################################


x = np.ravel(imgse.T)
training = x[:450]
test = x[450:]


p = hists(test)
os.chdir('/home/russellk/Documents')
np.savetxt('hists_test3.gz',p)

y = train(test)
os.chdir('/home/russellk/Documents')
np.savetxt('answer_key3.gz',y)

n = hists(training)
os.chdir('/home/russellk/Documents')
np.savetxt('hists_training3.gz',n)

z = train(training)
os.chdir('/home/russellk/Documents')
np.savetxt('training_results3.gz',z)