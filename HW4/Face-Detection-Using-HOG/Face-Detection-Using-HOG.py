import cv2
import numpy as np
import glob
from skimage.feature import hog
import sklearn.svm
import random
import pickle
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score


def Hog_Feature(k,Fldernames,Num,cell_size,block_size,flag):
    for i in range(k):
        random_class = random.randrange(Num)
        Filenames = glob.glob(f'{Fldernames[random_class]}/*.jpg')
        Num_Of_Images = len(Filenames)
        random_image = random.randrange(Num_Of_Images)
        Img = cv2.resize(cv2.imread(Filenames[random_image],0),(image_size,image_size))
        feature = hog(Img, orientations=9,pixels_per_cell=(cell_size,cell_size),cells_per_block=(block_size,block_size)).reshape(1,-1)
        if flag:
            Feature = feature.copy()
            flag = False
            continue
        Feature = cv2.vconcat([Feature,feature])
    return Feature

def Draw_precision_recall_curve(Y_true,Y_scores):
    precision,recall,_ = precision_recall_curve(Y_true,Y_scores)
    fig = plt.figure()
    plt.plot(precision,recall)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("precision-recall curve")
    fig.savefig("res2.jpg")
    plt.close()
    return

def Draw_ROC_curve(Y_true,Y_scores):
    fpr,tpr,_ = roc_curve(Y_true,Y_scores)
    fig = plt.figure()
    plt.plot(fpr,tpr)
    plt.title("ROC curve")
    fig.savefig("res1.jpg")
    plt.close()
    return

image_size = 64
max_val = 0
Flag = True
for kernel in ('linear', 'poly', 'rbf'):
    for cell_size in range(5,15):
        for block_size in range(1,5):
            Foldernames = glob.glob('nonface/*')
            Num_Of_Class = len(Foldernames)
            k = 1000
            NegativeFeatures = Hog_Feature(k,Foldernames,Num_Of_Class,cell_size,block_size,Flag)
            Foldernames = glob.glob('lfw/*')
            Num_Of_Class = len(Foldernames)
            PositiveFeature = Hog_Feature(k,Foldernames,Num_Of_Class,cell_size,block_size,Flag)
            Feature = cv2.vconcat([NegativeFeatures,PositiveFeature])
            Labels = cv2.vconcat([np.zeros([k]),np.ones([k])]).reshape(-1)    
            SVM = sklearn.svm.SVC(kernel=kernel)
            SVM.fit(Feature,Labels)
            t = 300
            Foldernames = glob.glob('nonface/*')
            Num_Of_Class = len(Foldernames)
            NegativeTest = Hog_Feature(t,Foldernames,Num_Of_Class,cell_size,block_size,Flag)
            Result = SVM.predict(NegativeTest)
            TrueResult = np.sum(Result==0)
            Foldernames = glob.glob('lfw/*')
            Num_Of_Class = len(Foldernames)
            PositiveTest = Hog_Feature(t,Foldernames,Num_Of_Class,cell_size,block_size,Flag)
            Result = SVM.predict(PositiveTest)
            TrueResult += np.sum(Result==1)
            if TrueResult>max_val:
                Final_kernel = kernel
                Final_cell = cell_size
                Final_block = block_size
                max_val = TrueResult
            


##########    testing     ##########    
image_size = 64
max_val = 0
Flag = True
Final_kernel = 'poly'
Final_cell = 5
Final_block = 3
Foldernames = glob.glob('nonface/*')
Num_Of_Class = len(Foldernames)
k = 10000
NegativeFeatures = Hog_Feature(k,Foldernames,Num_Of_Class,Final_cell,Final_block,Flag)
Foldernames = glob.glob('lfw/*')
Num_Of_Class = len(Foldernames)
PositiveFeature = Hog_Feature(k,Foldernames,Num_Of_Class,Final_cell,Final_block,Flag)
Feature = cv2.vconcat([NegativeFeatures,PositiveFeature])
Labels = cv2.vconcat([np.zeros([k]),np.ones([k])]).reshape(-1)    
with open('Feature.npy', 'rb') as f:
    Feature = np.load(f)  
with open('Labels.npy', 'rb') as f:
    Labels = np.load(f)        
FinalSVM = sklearn.svm.SVC(kernel=Final_kernel)
FinalSVM.fit(Feature,Labels)
Filename = 'FinalSVM.sav'
pickle.dump(FinalSVM, open(Filename, 'wb'))
# Filename = 'FinalSVM.sav'
# FinalSVM = pickle.load(open(Filename, 'rb'))
t = 1000
Foldernames = glob.glob('nonface/*')
Num_Of_Class = len(Foldernames)
NegativeTest = Hog_Feature(t,Foldernames,Num_Of_Class,Final_cell,Final_block,Flag)
NegativeResult = FinalSVM.predict(NegativeTest)
TrueResult = np.sum(NegativeResult==0)
Foldernames = glob.glob('lfw/*')
Num_Of_Class = len(Foldernames)
PositiveTest = Hog_Feature(t,Foldernames,Num_Of_Class,Final_cell,Final_block,Flag)
PositiveResult = FinalSVM.predict(PositiveTest)
TrueResult += np.sum(PositiveResult==1)
print(f'precision is {TrueResult/(2*t)*100}% succeed,\n kernel is {Final_kernel},\n block_size is {Final_block}\n and cell_size is {Final_cell}')     
Average_Precision = average_precision_score(cv2.vconcat([np.zeros([t]),np.ones([t])]),cv2.vconcat([NegativeResult,PositiveResult]))
Draw_precision_recall_curve(cv2.vconcat([np.zeros([t]),np.ones([t])]),cv2.vconcat([NegativeResult,PositiveResult]))
Draw_ROC_curve(cv2.vconcat([np.zeros([t]),np.ones([t])]),cv2.vconcat([NegativeResult,PositiveResult]))
