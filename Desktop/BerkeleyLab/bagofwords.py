import numpy as np 
import cv2
import dates
from scipy.cluster.vq import vq,kmeans,whiten
import scipy as spicy

sift = cv2.SIFT()
imgse  = np.array(dates.gothroughtimes())
bagofwords = np.loadtxt('/home/russellk/Documents/bagofwords2.gz')

def siftdescriptors(image):
	images = np.ravel(image)
	sift = cv2.SIFT()
	descriptors = np.zeros(128)
	for i in images:
		img1 = cv2.imread(i)
		img1_1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
		kp1,des1 = sift.detectAndCompute(img1_1,None)
		descriptors=np.vstack((descriptors,des1))
	descriptors1 = descriptors[1:]
	return descriptors1[::5]



def docluster(n,imgse):
	des = siftdescriptors(imgse)
	bagofwords= kmeans(des,n)
	return bagofwords[0]


def bghist(image):
	img1 = cv2.imread(image)
	img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	kp1,des1 = sift.detectAndCompute(img2,None)
	p = vq(des1,bagofwords)
	labels = p[0]
	hist,bins = np.histogram(labels,bins=np.arange(-.5,200,1))
	return hist

def eucdistance(image1,image2):
	hist1 = bghist(image1)
	hist2 = bghist(image2)
	return spicy.spatial.distance.euclidean(hist1,hist2)

def corrdistance(image1,image2):
	hist1 = bghist(image1)
	hist2 = bghist(image2)
	return spicy.spatial.distance.correlation(hist1,hist2)
	
def match(array):
	answer =[]
	for i in array:
		summ =0
		for k in array:
			summ +=eucdistance(i,k)
		answer.append(summ)
	return answer


a = docluster(75,imgse)
np.savetxt('bagofwords3.gz',a)
#training data