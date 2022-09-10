import  numpy as np
import  cv2
import glob
import time
import statistics


def CreateFeatureVector(counter,FeatureVector,filenames):
    for filename in filenames:
        pic = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
        resized_pic = cv2.resize(pic,(n,n), interpolation = cv2.INTER_CUBIC)
        feature= np.reshape(resized_pic,(1,-1))
        feature = (feature - np.mean(feature)) / np.std(feature)
        FeatureVector[counter,:] = feature
        counter = counter + 1         
    return counter,FeatureVector



def Find_Nearest_Neighbor(k):
    BestMatch = np.zeros([1500,k])
    Result = np.zeros([1500,1])
    for i in range(1500):
        testvector =  np.reshape(np.array(list(TestVector[i,:])*2985),(2985,-1))
        diff = np.sum(np.abs(FeatureVector-testvector),axis=1)
        for t in range(k):
            testresult = np.where(diff==np.min(diff))[0][0]
            diff[testresult]=np.max(diff)  
            for m in range(15):
                if testresult<NumofClasses[m]:
                    BestMatch[i,t] = m
                    break
        Result[i]=   statistics.mode(BestMatch[i,:])    
    return Result
                
    
def Find_Correct_Answer(matrix):  
    correct = np.zeros_like(matrix)
    for i in range(15):
        correct[100*i:100*(i+1)] = (matrix[100*i:100*(i+1)]==i)*1   
    return correct   

n = 16
FeatureVector = np.zeros([2985,n**2])
ClassesCounter = 0
Foldernames = glob.glob('Data/Train/*')
NumofClasses = np.zeros([np.size(Foldernames)])
counter = 0
for folder in Foldernames:
    Filenames = glob.glob(f'{folder}/*.jpg')
    counter,FeatureVector = CreateFeatureVector(counter,FeatureVector,Filenames)
    NumofClasses[ClassesCounter] = counter
    ClassesCounter = ClassesCounter + 1

counter = 0
TestVector = np.zeros([1500,n**2])
Foldernames = glob.glob('Data/Test/*')
for folder in Foldernames:
    Filenames = glob.glob(f'{folder}/*.jpg')
    counter,TestVector = CreateFeatureVector(counter,TestVector,Filenames)
 
# NN
Result = Find_Nearest_Neighbor(1)
Correct = Find_Correct_Answer(Result)
print(f'{np.sum(Correct)/1500*100} percent of test imeages classifed correctly with K = 1')

# KNN K=3 
Result = Find_Nearest_Neighbor(3)
Correct = Find_Correct_Answer(Result) 
print(f'{np.sum(Correct)/1500*100} percent of test imeages classifed correctly with K = 3')

# KNN K=5   
Result = Find_Nearest_Neighbor(5)
Correct = Find_Correct_Answer(Result) 
print(f'{np.sum(Correct)/1500*100} percent of test imeages classifed correctly with K = 5')

# KNN K=9   
Result = Find_Nearest_Neighbor(9)
Correct = Find_Correct_Answer(Result)
print(f'{np.sum(Correct)/1500*100} percent of test imeages classifed correctly with K = 9')

# KNN K=15   
Result = Find_Nearest_Neighbor(15)
Correct = Find_Correct_Answer(Result)
print(f'{np.sum(Correct)/1500*100} percent of test imeages classifed correctly with K = 15')

# KNN K=21   
Result = Find_Nearest_Neighbor(21)
Correct = Find_Correct_Answer(Result)
print(f'{np.sum(Correct)/1500*100} percent of test imeages classifed correctly with K = 21')
