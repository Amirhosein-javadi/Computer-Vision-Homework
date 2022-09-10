import numpy as np
import cv2
import glob

def Find_K(filenames):
    PatternSize = (6,9)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)
    flag = None
    ImageCoordinate = []
    ObjectCoordinate = []
    for filename in filenames:
        pic = cv2.imread(filename)
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        retval, corners = cv2.findChessboardCorners(pic,PatternSize,flag)
        corners = cv2.cornerSubPix(pic, corners,(100,100), (-1,-1), criteria )
        ImageCoordinate.append(corners)
        ObjectCoordinate.append(objp)    
    ret, matrix, distortion , rotation , translation  = cv2.calibrateCamera(ObjectCoordinate, ImageCoordinate, (pic.shape[1],pic.shape[0]),cv2.CALIB_USE_INTRINSIC_GUESS, None)
    return  matrix

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
Filenames = glob.glob('1/*.jpg')
M1 = Find_K(Filenames)
Filenames = glob.glob('2/*.jpg')
M2 = Find_K(Filenames)
Filenames = glob.glob('3/*.jpg')
M3 = Find_K(Filenames)
Filenames = glob.glob('4/*.jpg')
M4 = Find_K(Filenames)
        



