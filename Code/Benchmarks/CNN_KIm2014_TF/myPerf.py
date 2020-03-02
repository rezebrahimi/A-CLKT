'''
Created on Sep 28, 2017

@author: eb
'''
import numpy
def calcPerf(predicted, truthFile, ignoreThreshold=None):
    perf = {"acc":0,"pre":0,"rec":0,"f1":0 }
    FP=0
    FN=0
    TP=0
    TN=0    
    
    counter=0
    for p,t in zip(predicted,truthFile):        
        if (counter==ignoreThreshold):
            break
        elif p == t == 1:
            TP=TP+1
        elif p==0 and t==1:
            FN=FN+1
        elif p==1 and t==0:
            FP = FP +1
        else:
            TN = TN+1
        counter = counter +1;
    
    Precision = float(TP)/(TP+FP)
    Recall= float(TP)/(TP+FN)
    
    Accuracy = float(TP+TN)/(TP+TN+FP+FN)
    F_Measure = 2*Precision*Recall/(Precision+Recall)
    
    perf["pre"] = Precision
    perf["rec"]= Recall
    
    perf["acc"] = Accuracy
    perf["f1"] = F_Measure
    
    return perf

if __name__ == '__main__':
    
## Fr
#     predictions = []
#     with open('/home/eb/WSPY/CNN_KIm2014_TF/runs/prediction9.csv','r') as f:
#         for line in f:
#             if line.strip() == "1.0":
#                 predictions.append(1)
#             else:
#                 predictions.append(0)
#     print ('The size of predicted labels is: ' + str(len(predictions)))
#     y_val_Fr=[]
#     with open('/home/eb/dnm_data/MultiLingual/French-Labeled/test/DNM-test-Fr.cat','r') as f:
#         for line in f:
#             if line.strip() == "pos":
#                 y_val_Fr.append(1)
#             else:
#                 y_val_Fr.append(0)
#     print ('The size of validation labels is: ' + str(len(y_val_Fr)))
#     y_val_Fr = numpy.array(y_val_Fr)
## Ru
    predictions = []
    with open('/home/eb/WSPY/CNN_KIm2014_TF/runs/prediction9.csv','r') as f:
        for line in f:
            if line.strip() == "1.0":
                predictions.append(1)
            else:
                predictions.append(0)
    print ('The size of predicted labels is: ' + str(len(predictions)))
    y_val_Ru=[]
    with open('/home/eb/dnm_data/MultiLingual/Rus7030/val/DNM-RUS-test.cat','r') as f:
        for line in f:
            if line.strip() == "pos":
                y_val_Ru.append(1)
            else:
                y_val_Ru.append(0)
    print ('The size of validation labels is: ' + str(len(y_val_Ru)))
    y_val_Ru = numpy.array(y_val_Ru)    
    
    print (calcPerf(predictions, y_val_Ru))

