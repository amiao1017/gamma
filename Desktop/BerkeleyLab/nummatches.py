import cv2
import numpy as np 
import matplotlib.pyplot as plt 

def nummatches(image1,image2):
	img1 = cv2.imread(image1)
	img1_1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY) #changed to grayscale
	img2 = cv2.imread(image2)
	img2_1 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY) #changed to grayscale
	sift = cv2.SIFT()
	kp1,des1 = sift.detectAndCompute(img1_1,None)
	kp2,des2 = sift.detectAndCompute(img2_1,None)
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2,k=2)
	good = []
	for m,n in matches:
    		if m.distance < 0.75*n.distance:
           	 	good.append([m])
    
	return len(good)


a= '/home/russellk/Documents/TestImages/test1.jpg'
b= '/home/russellk/Documents/TestImages/test2.jpg'
c = '/home/russellk/Documents/TestImages/test3.jpg'
d = '/home/russellk/Documents/TestImages/test4.jpg'

e= '/home/russellk/Documents/TestImages/test2_1.jpg'
f= '/home/russellk/Documents/TestImages/test2_2.jpg'
g = '/home/russellk/Documents/TestImages/test2_3.jpg'
h = '/home/russellk/Documents/TestImages/test2_4.jpg'
p = '/home/russellk/Documents/TestImages/test2_5.jpg'
r = '/home/russellk/Documents/TestImages/test2_6.jpg'



array1= [a,b,c,d]
array2 = [e,f,g,h,p,r]

def match(array):
	answer =[]
	for i in array:
		summ =0
		for k in array:
			summ +=nummatches(i,k)
		answer.append(summ - nummatches(i,i))
	return answer
