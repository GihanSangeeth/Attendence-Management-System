import numpy as np 
import cv2 
from matplotlib import pyplot as plt
import glob

fpath = 'testImage/1/'
path = glob.glob(fpath+'*.png')
count = 1

for image in path:
    query_img = cv2.imread('quaryImage/1.png') 
    train_img = cv2.imread(image) 
    
    query_img = cv2.resize(query_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    train_img = cv2.resize(train_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('img',query_img)
    
    query_img_gray = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY) 
    train_img_gray = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY) 

#  Initialize the ORB detector algorithm 
    orb = cv2.ORB_create(1000) 

# Detect keypoints (features) cand calculate the descriptors
    query_keypoints, query_descriptors = orb.detectAndCompute(query_img_gray,None) 
    train_keypoints, train_descriptors = orb.detectAndCompute(train_img_gray,None) 

#     print(len(query_keypoints), len(train_keypoints))
    x = len(query_keypoints)
    y = len(train_keypoints)
    
    
    if (x < y):
        matchingPer = (x/y)*100
        
    elif (x>y):
        matchingPer = (y/x)*100
        
    else: matchingPer = 100

    
    if matchingPer > 70:
        print('matched image'+str(count))
            
    else: print('Not matched image'+str(count))
        
    count+=1

 

cv2.waitKey(0)
cv2.destroyAllWindows()