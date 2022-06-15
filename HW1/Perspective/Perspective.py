import cv2
import numpy as np
import math

im1 = cv2.imread('logo.png')
rows,cols,dim = im1.shape
f = 500
k = np.array([[f,0,cols//2],[0,f,rows//2],[0,0,1]])
t = 40
d = 25
phi = np.arctan(40/25)
n = np.reshape(np.array([0,-np.sin(phi),-np.cos(phi)]),(1,3))
R= np.array([[1,0,0],
             [0,np.cos(phi),-np.sin(phi)],
             [0,np.sin(phi),np.cos(phi)]])

t = np.array([0,40,0]).reshape(3,1)
H = np.matmul(k,R-np.matmul(t,n)/d)
H = np.matmul(H,np.linalg.inv(k))
T = np.array([[1,0,150],
              [0,1,700],
              [0,0,1]])

TH = np.matmul(T,H)
Warp_Im = cv2.warpPerspective(im1,TH,(570,1200))
cv2.imwrite ('res12.jpg',Warp_Im)





