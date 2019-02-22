import numpy as np
import matplotlib.pyplot
from matplotlib import pyplot as plt
from copy import deepcopy
import cv2 as cv2
UBIT = "tkaushik"
np.random.seed(sum([ord(c) for c in UBIT]))

def imagequant(k):
    #Reading the image and initialising random centroids
    img1 = cv2.imread('baboon.jpg')
    Mu=np.random.randint(img1.shape[0], size=k)
    centroid=[]
    for i in range(k):
        centroid.append(img1[Mu[i]][Mu[i]])
    #placeholder for storing the cluster value for each point
    clusters= np.zeros((len(img1), len(img1[0]),1))
    #Function to return the nearest cluster for each pixel
    def dist(a, b):
        dist=list()
        for j in range(k):
            distance = np.sqrt(np.sum((a-b[j])**2))
            distance=np.asscalar(distance)
            dist.append(distance)
        dist=np.array(dist)
        cluster_index = np.argmin(dist)
        return cluster_index
    #Running the K means algorithm
    for iter in range(20):
        for i in range(len(img1)):
            for j in range(len(img1[0])):
                cluster_index=dist(img1[i][j],centroid)
                clusters[i][j]=cluster_index
        prev_centroid=deepcopy(centroid)
        #Mapping each point to its respective cluster
        for a in range(k):
            pts=list()
            for b in range(len(img1)):
                for c in range(len(img1[0])):
                    if(np.asscalar(clusters[b][c])==a):
                        pts.append(img1[b][c].tolist())
            pts=np.array(pts)
            #For the centroid calculated computing the mean of the cluster points
            #and checking its euclidean distance distance from the previous centroid
            centroid[a] = np.mean(pts, axis=0)
            distance=np.sqrt(np.sum((centroid[a]-prev_centroid[a])**2))
            if(distance==0):
                break
    #For the final centroids, the cluster points are mapped to the
    #centroid color value
    for a in range(len(centroid)):
        for b in range(len(img1)):
            for c in range(len(img1[0])):
                if(np.asscalar(clusters[b][c])==a):
                    img1[b][c]=centroid[a]
    return img1

task3_baboon_3=imagequant(3)
task3_baboon_5=imagequant(5)
task3_baboon_10=imagequant(10)
task3_baboon_20=imagequant(20)
cv2.imwrite('task3_baboon_3.jpg',task3_baboon_3)
cv2.imwrite('task3_baboon_5.jpg',task3_baboon_5)
cv2.imwrite('task3_baboon_10.jpg',task3_baboon_10)
cv2.imwrite('task3_baboon_20.jpg',task3_baboon_20)
