import  numpy as np
import  cv2
import glob
import sklearn
from sklearn.cluster import KMeans
import pickle

def Find_Num_of_Classes():
    NumofClasses = np.zeros([15])
    Foldernames = glob.glob('Data/Train/*')
    for i in range(15):
        folder = Foldernames[i]
        Filenames = glob.glob(f'{folder}/*.jpg')
        if i==0 :
            NumofClasses[i]= np.size(Filenames)
        else:
            NumofClasses[i]= np.size(Filenames)+NumofClasses[i-1]
    return NumofClasses

def Find_Descriptor(foldernames):
    sift = cv2.SIFT_create()
    Descriptor = np.zeros([1,128]).astype(np.uint8)
    Counter = np.zeros([1,1]).astype(np.uint32)
    for folder in foldernames:
        Filenames = glob.glob(f'{folder}/*.jpg')
        for filename in Filenames:
            pic = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
            keyPoint,desc = sift.detectAndCompute(pic,None)
            Descriptor = np.concatenate((Descriptor,desc),axis=0)
            c = np.array(np.size(keyPoint)).reshape(1,1)
            Counter = np.concatenate((Counter,c),axis=0)
    Descriptor = np.delete(Descriptor,0,axis=0)
    Counter = np.delete(Counter,0,axis=0)
    return  Descriptor,Counter  
    return Descriptor,Counter 
 
def Create_Learner(Descriptor,k):
    Learner =  sklearn.cluster.KMeans(n_clusters=k,random_state=0).fit(Descriptor)
    return Learner

def Create_Histogram(Desc,k,Number,Learner):
    hist_size = np.shape(Number)[0]
    Histogram = np.zeros([hist_size,k])
    counter = 0     
    i = 0
    for j in range(Number[i][0]):
        sample = Desc[j,:].reshape(1, -1)
        data = int(Learner.predict(sample))
        Histogram[i,data] += 1
    counter += Number[i][0]
    Histogram[i,:] = Histogram[i,:] / np.sum(Histogram[i,:])
    for i in range(1,hist_size):
        for j in range(Number[i][0]):
            sample = Desc[j+counter,:].reshape(1, -1)
            data = int(Learner.predict(sample))
            Histogram[i,data] += 1
        counter += Number[i][0]
        Histogram[i,:] = Histogram[i,:] / np.sum(Histogram[i,:])
    return Histogram  

def Create_Knn_Learner(Histogram,Num):
    knn = cv2.ml.KNearest_create()
    Histogram = np.float32(Histogram)
    label = np.zeros([np.size(Histogram,axis=0),1]).astype(np.float32)
    for i in range(1,15):
        for j in range(Num[i-1],Num[i]):
            label[j] = i
    knn.train(Histogram, cv2.ml.ROW_SAMPLE,label)
    return knn

def Find_Nearest_Histogram(knn,Histogram):
    Histogram = np.float32(Histogram)
    Result = np.zeros([1500,1])
    Result = knn.findNearest(Histogram,39)
    return Result[1]

def Find_Correct_Answer(matrix):  
    correct = np.zeros_like(matrix)
    for i in range(15):
        correct[100*i:100*(i+1)] = (matrix[100*i:100*(i+1)]==i)*1   
    return correct

Foldernames = glob.glob('Data/Train/*')
Descriptor_Train,Train_Number = Find_Descriptor(Foldernames)
Descriptor_Train = np.float32(Descriptor_Train)
NumofClasses = np.int16(Find_Num_of_Classes())
Foldernames = glob.glob('Data/Test/*')
Descriptor_Test,Test_Number = Find_Descriptor(Foldernames)
Descriptor_Test = np.float32(Descriptor_Test)
K = 100  
Learner = Create_Learner(Descriptor_Train,K)
with open("Learner120.pkl", "wb") as f:
    pickle.dump(Learner, f)
TrainHistogram = Create_Histogram(Descriptor_Train,K,Train_Number,Learner)
TrainHistogram.dump("TrainHistogram.dat")
TestHistogram = Create_Histogram(Descriptor_Test,K,Test_Number,Learner)
TestHistogram.dump("TestHistogram.dat")
Knn = Create_Knn_Learner(TrainHistogram,NumofClasses)
Result = Find_Nearest_Histogram(Knn,TestHistogram)
Correct = Find_Correct_Answer(Result)
print(f'{np.sum(Correct)/1500*100} percent of test imeages classifed correctly with KNN')
# NumofClasses = np.int16(Find_Num_of_Classes())
# Descriptor_Train = np.float32(np.load("Descriptor_Train.dat",allow_pickle=True))  
# Train_Number = np.load("Train_Number.dat",allow_pickle=True)
# Descriptor_Test = np.float32(np.load("Descriptor_Test.dat",allow_pickle=True))  
# Test_Number = np.load("Test_Number.dat",allow_pickle=True)
# K = 100  
# with open("Learner.pkl", "rb") as f:
#     Learner = pickle.load(f)
# TrainHistogram = np.load("TrainHistogram.dat",allow_pickle=True)    
# TestHistogram = np.load("TestHistogram.dat",allow_pickle=True)
# Knn = Create_Knn_Learner(TrainHistogram,NumofClasses)
# Result = Find_Nearest_Histogram(Knn,TestHistogram)
# Correct = Find_Correct_Answer(Result)
# print(f'{np.sum(Correct)/1500*100} percent of test imeages classifed correctly with KNN')
