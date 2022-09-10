import  numpy as np
import  cv2
from sklearn.linear_model import LinearRegression, RANSACRegressor
        
def Find_VanishingPoint(lines,min_x):  
    pic = Image.copy()
    xcoordinate = np.zeros([1,2])
    ycoordinate = np.zeros([1,2])
    zcoordinate = np.zeros([1,2])
    for i in range(np.size(lines,axis=0)):
        x1, y1, x2, y2 = lines[i][0]  
        if x1!=x2:
            slope = (y1-y2)/(x1-x2)
            if (x1>min_x) and (x2>min_x):
                if slope>0.25 and slope<0.3:
                    new_point = np.array([slope,y1-slope*x1]).reshape([1,2])
                    xcoordinate = cv2.vconcat([xcoordinate,new_point])
                elif slope<0 and slope>-0.1:
                    new_point = np.array([slope,y1-slope*x1]).reshape([1,2])
                    ycoordinate = cv2.vconcat([ycoordinate,new_point])
                elif slope>30:
                    new_point = np.array([slope,y1-slope*x1]).reshape([1,2])
                    zcoordinate = cv2.vconcat([zcoordinate,new_point])
         
    xcoordinate = np.delete(xcoordinate,0,axis=0)
    ycoordinate = np.delete(ycoordinate,0,axis=0)    
    zcoordinate = np.delete(zcoordinate,0,axis=0) 
    n_x = np.size(xcoordinate,axis=0)
    xcollision  = np.zeros([n_x*(n_x-1)//2,2])
    n_y = np.size(ycoordinate,axis=0)
    ycollision  = np.zeros([n_y*(n_y-1)//2,2])
    n_z = np.size(zcoordinate,axis=0)
    zcollision  = np.zeros([n_z*(n_z-1)//2,2]) 
    xcounter = 0
    ycounter = 0
    zcounter = 0
    for i in range(n_x-1):
        for j in range(i+1,n_x):
            if (xcoordinate[i,0]==xcoordinate[j,0]):
                continue
            xcollision[xcounter,0] =  (xcoordinate[j,1]-xcoordinate[i,1])/(xcoordinate[i,0]-xcoordinate[j,0])
            xcollision[xcounter,1] =  xcollision[xcounter,0]*xcoordinate[i,0]+xcoordinate[i,1]
            xcounter += 1
    for i in range(n_y-1):
        for j in range(i+1,n_y):
            if (ycoordinate[i,0]==ycoordinate[j,0]):
                continue            
            ycollision[ycounter,0] =  (ycoordinate[j,1]-ycoordinate[i,1])/(ycoordinate[i,0]-ycoordinate[j,0])
            ycollision[ycounter,1] =  ycollision[ycounter,0]*ycoordinate[i,0]+ycoordinate[i,1]
            ycounter += 1
    for i in range(n_z-1):
        for j in range(i+1,n_z):
            if (zcoordinate[i,0]==zcoordinate[j,0]):
                continue
            zcollision[zcounter,0] =  (zcoordinate[j,1]-zcoordinate[i,1])/(zcoordinate[i,0]-zcoordinate[j,0])
            zcollision[zcounter,1] =  zcollision[zcounter,0]*zcoordinate[i,0]+zcoordinate[i,1]
            zcounter += 1  
            
    Vx = np.ones([3,1]).astype(np.int32)   
    Vy = np.ones([3,1]).astype(np.int32)  
    Vz = np.ones([3,1]).astype(np.int32)  
     
    RansacEstimator = RANSACRegressor(base_estimator=LinearRegression(),min_samples=n_x-1,
                                        max_trials=100,loss='absolute_loss',random_state=100)       
    X = xcollision[:,0].reshape(-1,1)
    Y = xcollision[:,1].reshape(-1,1)
    RansacEstimator.fit(X,Y)
    Mask = RansacEstimator.inlier_mask_
    X = X[Mask]
    Y = Y[Mask]
    Vx[0,0] = np.average(X)
    Vx[1,0] = np.average(Y) 
    RansacEstimator = RANSACRegressor(base_estimator=LinearRegression(),min_samples=100,
                                        max_trials=100,loss='absolute_loss',random_state=100)   
    X = ycollision[:,0].reshape(-1,1)
    Y = ycollision[:,1].reshape(-1,1)
    RansacEstimator.fit(X,Y)
    Mask = RansacEstimator.inlier_mask_
    X = X[Mask]
    Y = Y[Mask]
    Vy[0,0] = np.average(X)
    Vy[1,0] = np.average(Y)     
    X = zcollision[:,0].reshape(-1,1)
    Y = zcollision[:,1].reshape(-1,1)
    RansacEstimator.fit(X,Y)
    Mask = RansacEstimator.inlier_mask_
    X = X[Mask]
    Y = Y[Mask]
    Vz[0,0] = np.average(X)
    Vz[1,0] = np.average(Y)      
    return Vx,Vy,Vz    


    
Image = cv2.imread("vns.jpg")
gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
row,col,dim = Image.shape
min_x = col//4
Edge = cv2.Canny(gray, 150, 300, None, 3)
lines = cv2.HoughLinesP(Edge, 1, np.pi / 360, 300, None, minLineLength=50, maxLineGap=35)
Vx,Vy,Vz = Find_VanishingPoint(lines,min_x)
Vx.dump("Vx.dat")
Vy.dump("Vy.dat")
Vz.dump("Vz.dat")
Vxx = np.array([[0,-Vx[2,0],Vx[1,0]],
                [Vx[2,0],0,-Vx[0,0]],
                [-Vx[1,0],Vx[0,0],0]])
H = np.matmul(Vxx,Vy)
H = H /(H[0]**2+ H[1]**2)**0.5
x_indx = np.zeros([2]).astype(np.int32) 
x_indx[0] = 0
x_indx[1] = col-1 
y_indx = np.zeros([2]).astype(np.int32) 
y_indx[0] = -H[2,0]/H[1,0]
y_indx[1] = -(H[2,0]+H[0,0]*x_indx[1])/H[1,0]
Bigrow = 8000
Bigcol = 8000
VanishLine = np.ones([Bigrow,Bigcol,3]).astype(np.uint8)*255
VanishLine[Bigrow//2-row//2:Bigrow//2+row//2+1,Bigcol//2-col//2:Bigcol//2+col//2,:] = Image
VanishLine = cv2.line(VanishLine,tuple([Bigcol//2-col//2+x_indx[0],Bigrow//2-row//2+y_indx[0]]),tuple([Bigcol//2-col//2+x_indx[1],Bigrow//2-row//2+y_indx[1]]),(0,0,255),20)
cv2.imwrite('res01.jpg',VanishLine)
minimumratio = 10
Bigrow = (max(Vx[1],Vy[1],Vz[1]) - min(Vx[1],Vy[1],Vz[1])+1000)[0]//minimumratio
Bigcol = (max(Vx[0],Vy[0],Vz[0]) - min(Vx[0],Vy[0],Vz[0])+1000)[0]//minimumratio
rowshift = (-(min(Vx[1],Vy[1],Vz[1])[0])+500)//minimumratio
colshift = (-(min(Vx[0],Vy[0],Vz[0])[0])+500)//minimumratio
VanishPoint = np.ones([Bigrow,Bigcol,3]).astype(np.uint8)*255
VanishPoint = cv2.rectangle(VanishPoint,(colshift-col//minimumratio,rowshift-row//minimumratio),(colshift+col//minimumratio,rowshift+row//minimumratio),(0,255,0),20)
VanishPoint = cv2.circle(VanishPoint,(colshift+Vx[0][0]//minimumratio,rowshift+Vx[1][0]//minimumratio), 20, (255,0,0), 20)
VanishPoint = cv2.circle(VanishPoint,(colshift+Vy[0][0]//minimumratio,rowshift+Vy[1][0]//minimumratio), 20, (255,0,0), 20)
VanishPoint = cv2.circle(VanishPoint,(colshift+Vz[0][0]//minimumratio,rowshift+Vz[1][0]//minimumratio), 20, (255,0,0), 20)
VanishPoint = cv2.line(VanishPoint,(colshift+Vx[0][0]//minimumratio,rowshift+Vx[1][0]//minimumratio),
                                    (colshift+Vy[0][0]//minimumratio,rowshift+Vy[1][0]//minimumratio),
                                    (0,0,255), 5)
cv2.imwrite('res02.jpg',VanishPoint)