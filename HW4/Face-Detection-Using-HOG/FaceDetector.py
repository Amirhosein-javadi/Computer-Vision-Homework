import cv2
import numpy as np
import pickle
from skimage.feature import hog

def Non_Max_Suppression(boxes,Threshold):
	pick = []
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,0] + boxes[:,2]
	y2 = boxes[:,1] + boxes[:,2]
	S = (x2 - x1 + 1) * (y2 - y1 + 1)
	index = np.argsort(y2)
	while len(index) > 0:
		last = len(index) - 1
		i = index[last]
		pick.append(i)
		suppress = [last]
		for pos in range(0, last):
			j = index[pos]
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
			overlap = float(w * h) / S[j]
			if overlap > Threshold:
				suppress.append(pos)
		index = np.delete(index, suppress)
	return boxes[pick]  
        
    
def FaceDetector(image,Filename):
    cell_size = 5
    block_size = 3
    image_size = 64
    scale = 1.05
    SVM = pickle.load(open('FinalSVM.sav', 'rb'))
    gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    row,col = gray.shape
    windows = []
    windowsscale = 64
    flag=True
    while windowsscale< min(row,col):
        for r in range(0,row-windowsscale,10):
            for c in range(0,col-windowsscale,10): 
                img = gray[r:r+windowsscale,c:c+windowsscale]
                img = cv2.resize(img,(image_size,image_size))
                feature = hog(img, orientations=9,pixels_per_cell=(cell_size,cell_size),cells_per_block=(block_size,block_size)).reshape(1,-1)
                Result = SVM.predict(feature)
                if Result[0] == 1 :
                    windows.append([r,c,windowsscale])
                    if flag==True :
                        Feature = feature.copy()
                        flag = False
                        continue
                    Feature = cv2.vconcat([Feature,feature])
        windowsscale = int(windowsscale * scale)
    score = SVM.decision_function(Feature)
    Threshold = 1
    state = score>Threshold
    indx = list(np.where(state==True)[0])
    filtered_windows = [windows[indx[0]]]
    for i in range(1,len(indx)):
        filtered_windows.append(windows[indx[i]])
    box = Non_Max_Suppression(np.array(filtered_windows),0.5)
    for i in range(0,len(box)):
        image = cv2.rectangle(image,(box[i][1],box[i][0]), (box[i][1]+box[i][2],box[i][0]+box[i][2]), (255,255,255), 2)
    cv2.imwrite(Filename,image)    
    return

Img = cv2.imread("melli.jpg")
Filename = 'melli-detected-faces.jpg'
FaceDetector(Img,Filename)
Img = cv2.imread("Persepolis.jpg")
Filename = 'Persepolis-detected-faces.jpg'
FaceDetector(Img,Filename)
Img = cv2.imread("Esteghlal.jpg")
Filename = 'Esteghlal-detected-faces.jpg'
FaceDetector(Img,Filename)