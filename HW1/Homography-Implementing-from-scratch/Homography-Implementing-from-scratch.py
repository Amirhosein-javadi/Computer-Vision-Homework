import cv2 
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import random
import math

def Find_Match(descriptor1,descriptor2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    return good  

def Get_Random_Point(src,dst):
    n = random.randint(0,np.size(src,axis=1)-1)
    src_point = src[:,n]
    dst_point = dst[:,n]
    return src_point,dst_point

def Find_Difference(src_indx,dst_indx,h):
    new_indx = np.append(dst_indx, np.ones([1,np.size(dst_indx,axis=1)]),axis=0)    
    new_indx = np.matmul(h,new_indx)       
    new_indx[0:2,:] = (new_indx[0:2,:]/new_indx[2,:])
    new_indx = np.delete(new_indx,2,0) 
    diff = new_indx-src_indx
    diff = np.array(diff[0,:]**2+diff[1,:]**2)
    return diff

Img1 = cv2.imread('im03.jpg')         
Img2 = cv2.imread('im04.jpg')  
Sift = cv2.SIFT_create()
kp1, des1 = Sift.detectAndCompute(Img1,None)
kp2, des2 = Sift.detectAndCompute(Img2,None)

Goodpoints = Find_Match(des1,des2)        
Src_pts = np.int32([kp1[m.queryIdx].pt for m in Goodpoints]).reshape(-1,2).T  # first column x second column y
Dst_pts = np.int32([kp2[m.trainIdx].pt for m in Goodpoints]).reshape(-1,2).T 

P = 0.99
S=4
Counter=0
W=0
Wmin=0
A = np.zeros([8,9]) 
Thrshld = 25**2
N = 1e9

while Counter < N:
    A = np.zeros([8,9]) 
    for i in range(1,5):
        globals()["spoint" + str(i)],globals()["dpoint" + str(i)] = Get_Random_Point(Src_pts,Dst_pts)  
        A[2*(i-1),3:6]   =-globals()["dpoint"+str(i)][0],-globals()["dpoint"+str(i)][1],-1
        A[2*(i-1),6:9]   = globals()["spoint"+str(i)][1]* globals()["dpoint"+str(i)][0],globals()["spoint" + str(i)][1]*globals()["dpoint" + str(i)][1],globals()["spoint" + str(i)][1]    
        A[2*(i-1)+1,0:3] = globals()["dpoint"+str(i)][0], globals()["dpoint"+str(i)][1],1 
        A[2*(i-1)+1,6:9] =-globals()["spoint"+str(i)][0]* globals()["dpoint"+str(i)][0],-globals()["spoint" + str(i)][0]*globals()["dpoint" + str(i)][1],-globals()["spoint" + str(i)][0]            
    B = -A[:,-1]  
    A = A[:,:-1]    
    status,h = cv2.solve(A,B,flags=cv2.DECOMP_SVD)
    H = np.append(h, np.ones([1,1]), axis=0).reshape(-1,3)  
    Diff = Find_Difference(Src_pts,Dst_pts,H)     
    W = np.sum(Diff<Thrshld)/np.size(Src_pts,axis=1) 
    if W>Wmin :
        Wmin = W.copy()
        BestH = H.copy()
        N = math.log(1-P)/math.log(1-W**S)
    Counter +=1 
    
deltax,deltay = np.matmul(BestH,np.array([0,0,1]).T)[0:2]
T = np.array([[1,0,-deltax],
              [0,1,-deltay],
              [0,0,1]]) 
MT = np.matmul(T,BestH)   
Warp_Im = cv2.warpPerspective(Img2,MT,(9000,4000))
cv2.imwrite ('res20.jpg',Warp_Im) 
