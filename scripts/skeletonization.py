import cv2
import numpy as np
from scipy import ndimage
import scipy


img = cv2.imread('7.png',0)


def make_skeletonization(img):
	size = np.size(img)
	skel = np.zeros(img.shape,np.uint8)

	ret,img = cv2.threshold(img,127,255,0)
	element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	done = False

	while (not done):
	    eroded = cv2.erode(img,element)
	    temp = cv2.dilate(eroded,element)
	    temp = cv2.subtract(img,temp)
	    skel = cv2.bitwise_or(skel,temp)
	    img = eroded.copy()

	    zeros = size - cv2.countNonZero(img)
	    if zeros == size:
	        done = True

	skel += img
	cv2.imshow("skel",skel)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return skel


def make_Sobel_filtration(img):
	im = img.astype('int32')
	dx = ndimage.sobel(im, 0)  # horizontal derivative
	dy = ndimage.sobel(im, 1)  # vertical derivative
	mag = np.hypot(dx, dy)  # magnitude
	mag *= 255.0 / np.max(mag)  # normalize (Q&D)
	mag = mag.astype(np.uint8)
	cv2.imshow('Sobel', mag)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return mag


def make_Canny_filtration(img):
	im = img.astype('int32')
	edges = cv2.Canny(img, 100, 200)
	cv2.imshow('Canny', edges)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return edges



skeleton = make_skeletonization(img)
#Sobel = make_Sobel_filtration(img)
Canny = make_Canny_filtration(img)

skeleton += Canny
cv2.imshow('Sum', skeleton)
cv2.waitKey(0)
cv2.destroyAllWindows()