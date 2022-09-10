import  numpy as np
import  cv2
import glob
import statistics
import pickle
import sklearn.svm
from sklearn.metrics import confusion_matrix

def Find_Label():
    Label = np.zeros([2985])
    Foldernames = glob.glob('Data/Train/*')
    counter = 0
    for i in range(15):
        folder = Foldernames[i]
        Filenames = glob.glob(f'{folder}/*.jpg')
        Label[counter:counter+np.size(Filenames)]= i
        counter=counter+np.size(Filenames)
    return Label  

def Find_Correct_Answer(matrix):  
    correct = np.zeros_like(matrix)
    for i in range(15):
        correct[100*i:100*(i+1)] = (matrix[100*i:100*(i+1)]==i)*1   
    return correct   

Descriptor_Train2 = np.float32(np.load("Descriptor_Train.dat",allow_pickle=True))

TrainHistogram2 = np.load("TrainHistogram.dat",allow_pickle=True)
Labels = np.int16(Find_Label())
SVM = sklearn.svm.LinearSVC(C=16.8,random_state=0,tol=1e-6,multi_class='ovr') #1.9428
SVM.fit(TrainHistogram2,Labels)
TestHistogram2 = np.load("TestHistogram.dat",allow_pickle=True)
Result2 = SVM.predict(TestHistogram2)
True_Val = np.zeros([1500])
for i in range(15):
    for j in range(100):
        True_Val[j+i*100]=i
Confusion_Matrix = confusion_matrix(True_Val,Result2)
Correct2 = Find_Correct_Answer(Result2)
print(f'{np.sum(Correct2)/1500*100} percent of test imeages classifed correctly with SVD')