import cv2 
import numpy as np

Img1 = cv2.imread('Img1.jpg')         
Img2 = cv2.imread('Img2.jpg') 
rows1,cols1,dim = Img1.shape
rows2,cols2,dim = Img2.shape
# Initiate SIFT detector
sift = cv2.SIFT_create()
KeyPoint1,Desc1 = sift.detectAndCompute(Img1,None)
KeyPoint2,Desc2 = sift.detectAndCompute(Img2,None)
           
Keypoints1 = cv2.drawKeypoints(Img1,KeyPoint1,np.array([]),color=(0,128,0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
Keypoints2 = cv2.drawKeypoints(Img2,KeyPoint2,np.array([]),color=(0,128,0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
Keypoints  = np.zeros([rows1,cols1+cols2,3])
Keypoints  [0:rows1,0:cols1,:] = Keypoints1[0:rows1,0:cols1]
Keypoints  [rows1//2-rows2//2:rows1//2+rows2//2,cols1:cols1+cols2,:] = Keypoints2[0:rows2,0:cols2]
cv2.imwrite('Corners.jpg',Keypoints)

FLANN_INDEX_KDTREE = 1
param1 = dict(algorithm=FLANN_INDEX_KDTREE, trees=20)
param2 = dict(checks=100)
flann = cv2.FlannBasedMatcher(param1,param2)
matches = flann.knnMatch(Desc1,Desc2,k=2)
Goodpoints = []
for i,j in matches:
    if i.distance < 0.75*j.distance:
        Goodpoints.append(i)
Pts1 = np.int32([ KeyPoint1[i.queryIdx].pt for i in Goodpoints ]).reshape(-1,2)
Pts2 = np.int32([ KeyPoint2[i.trainIdx].pt for i in Goodpoints ]).reshape(-1,2)

Correspondence1 = Keypoints1.copy()
Correspondence2 = Keypoints2.copy()
for i in range(np.size(Pts1,axis=0)):
    Correspondence1 = cv2.circle(Correspondence1,(Pts1[i][0],Pts1[i][1]),3,(255,0,0),3)
    Correspondence2 = cv2.circle(Correspondence2,(Pts2[i][0],Pts2[i][1]),3,(255,0,0),3) 
 
Correspondence  = np.zeros([rows1,cols1+cols2,3])
Correspondence  [0:rows1,0:cols1,:] = Correspondence1
Correspondence  [rows1//2-rows2//2:rows1//2+rows2//2,cols1:cols1+cols2,:] = Correspondence2
cv2.imwrite('Correspondences.jpg',Correspondence)   

Matchpoints = np.zeros([rows1,cols1+cols2,3])
Matchpoints [0:rows1,0:cols1,:] = Img1
Matchpoints [rows1//2-rows2//2:rows1//2+rows2//2,cols1:cols1+cols2,:] = Img2
for i in range(np.size(Pts1,axis=0)):
        Matchpoints = cv2.line(Matchpoints,(Pts1[i][0],Pts1[i][1]),(Pts2[i][0]+cols1,Pts2[i][1]+rows1//2-rows2//2),(255,0,0),1)  
cv2.imwrite ('Matches.jpg',Matchpoints)
 
Matchpoints2 =  np.zeros([rows1,cols1+cols2,3])
Matchpoints2 [0:rows1,0:cols1,:] = Img1
Matchpoints2 [rows1//2-rows2//2:rows1//2+rows2//2,cols1:cols1+cols2,:] = Img2
for i in range(20):
        Matchpoints2 = cv2.line(Matchpoints2,(Pts1[i][0],Pts1[i][1]),(Pts2[i][0]+cols1,Pts2[i][1]+rows1//2-rows2//2),(255,0,0),1)  
cv2.imwrite ('Matches-2.jpg',Matchpoints2)

M, mask = cv2.findHomography(Pts2,Pts1,cv2.RANSAC,10,100)
Inlier1 = Img1.copy()
Inlier2 = Img2.copy()
for i in range(np.size(Pts1,axis=0)):
    if mask[i,0]==1:
        Inlier1 = cv2.circle(Inlier1,(Pts1[i][0],Pts1[i][1]),3,(0,0,255),3)
        Inlier2 = cv2.circle(Inlier2,(Pts2[i][0],Pts2[i][1]),3,(0,0,255),3) 
    else:
        Inlier1 = cv2.circle(Inlier1,(Pts1[i][0],Pts1[i][1]),3,(255,0,0),3)
        Inlier2 = cv2.circle(Inlier2,(Pts2[i][0],Pts2[i][1]),3,(255,0,0),3) 
Inlier  = np.zeros([rows1,cols1+cols2,3])
Inlier  [0:rows1,0:cols1,:] = Inlier1
Inlier  [rows1//2-rows2//2:rows1//2+rows2//2,cols1:cols1+cols2,:] = Inlier2
cv2.imwrite('Inlier.jpg',Inlier)  
deltax,deltay = np.matmul(M,np.array([0,0,1]).T)[0:2]
# I change this line  if i.distance < 0.75*j.distance: with  if i.distance < 1.1*j.distance: and i create res18.jpg
deltax,deltay = np.matmul(M,np.array([0,0,1]).T)[0:2]
t = np.array([[1,0,-deltax],
              [0,1,-deltay],
              [0,0,1]])
MH = np.matmul(t,M)
Warp_Im = cv2.warpPerspective(Img2,MH,(9000,4000))
cv2.imwrite ('Warp_Im.jpg',Warp_Im)
