import  numpy as np
import  cv2
image = cv2.imread("vns.jpg")
row,col,dim = image.shape
R = np.load("R.dat",allow_pickle=True)
Warp_Im = cv2.warpPerspective(image,R,(col,row))
cv2.imwrite('res04.jpg',Warp_Im)