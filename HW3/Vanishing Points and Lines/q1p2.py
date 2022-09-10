import  numpy as np
import  cv2
from sympy.solvers import solve
from sympy import Symbol
import math

def Solver(vx,vy,vz):
    a1,b1,c1 = vx[:,0]
    a2,b2,c2 = vy[:,0]
    a3,b3,c3 = vz[:,0]
    Px = Symbol('Px')
    Py = Symbol('Py')
    F  = Symbol('F')    
    equation1 = -c1*c2*F**2-c1*c2*Px**2-c1*c2*Py**2+(a2*c1+a1*c2)*Px+(b2*c1+b1*c2)*Py-(a1*a2+b1*b2)
    equation2 = -c1*c3*F**2-c1*c3*Px**2-c1*c3*Py**2+(a3*c1+a1*c3)*Px+(b3*c1+b1*c3)*Py-(a1*a3+b1*b3)
    equation3 = -c2*c3*F**2-c2*c3*Px**2-c2*c3*Py**2+(a3*c2+a2*c3)*Px+(b2*c3+b3*c2)*Py-(a3*a2+b3*b2)
    answer = solve([equation1,equation2,equation3],Px,Py,F)
    px = np.float(abs(answer[0][0]))
    py = np.float(abs(answer[0][1]))
    f  = np.float(abs(answer[0][2]))
    return px,py,f

image = cv2.imread("vns.jpg")
Vx = np.load("Vx.dat",allow_pickle=True)
Vy = np.load("Vy.dat",allow_pickle=True)
Vz = np.load("Vz.dat",allow_pickle=True)
Px,Py,F = Solver(Vx,Vy,Vz)
Px = np.int16(Px)
Py = np.int16(Py)
F = np.int16(F)
K = np.array([[F,0,Px],
              [0,F,Py],
              [0,0,1]])
Final_Im = image.copy()
Final_Im = cv2.circle(Final_Im,(Px,Py), 20, (255,0,0), 20)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(Final_Im,f'F is {F}',(1900,400), font, 5,(0,0,255),15)
cv2.imwrite('res03.jpg',Final_Im)
XBackProjection = np.matmul(np.linalg.inv(K),Vx)
YBackProjection = np.matmul(np.linalg.inv(K),Vy)
ZBackProjection = np.matmul(np.linalg.inv(K),Vz)
Z_Angle = -math.atan((Vx[1]-Vy[1])/(Vx[0]-Vy[0]))
R1 = np.array([[math.cos(Z_Angle),-math.sin(Z_Angle),0],
              [math.sin(Z_Angle),math.cos(Z_Angle),0],
              [0,0,1]])
ZRotated = np.matmul(R1,ZBackProjection)
X_Angle = -math.atan(ZRotated[0]/ZRotated[1])
R2 = np.array([[math.cos(X_Angle),-math.sin(X_Angle),0],
              [math.sin(X_Angle),math.cos(X_Angle),0],
              [0,0,1]])
ZRotated2 = np.matmul(R2,ZRotated)
R = np.matmul(R2,R1)
R.dump("R.dat")
