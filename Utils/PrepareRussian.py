'''
Created on Jun 25, 2018

@author: eb
'''
from keras.preprocessing import text
from keras.preprocessing import sequence
import numpy as np

def prepareRussianData():
    ################ Russian Data Preparation
    #### Build train data (list of texts) - 380
    trainPosFile = open("/home/eb/dnm_data/MultiLingual/Rus7030/train/DNM-Rus-pos.txt", "r")
    trainNegFile = open("/home/eb/dnm_data/MultiLingual/Rus7030/train/DNM-Rus-neg.txt", "r")
    
    trainTexts=[]
    for l in trainPosFile:
        trainTexts.append(l)
    trainPosFile.close()
    for l in trainNegFile:
        trainTexts.append(l)
    trainNegFile.close()
    
    print trainTexts[0]
    print ('train set size is: ' +str(len(trainTexts)))
    
    ##Build the labels (Pos=1, Neg=0)
    y_train=[]
    with open('/home/eb/dnm_data/MultiLingual/Rus7030/train/DNM-RUS-train.cat','r') as f:
        for line in f:
            if line.strip() == "pos":
                y_train.append(1)
            else:
                y_train.append(0)
    print ('The size of training labels is: ' + str(len(y_train)))
    
    
    #### Build validation (test) data  - 55 + 117= 172
    valPosFile = open("/home/eb/dnm_data/MultiLingual/Rus7030/val/DNM-Rus-pos.txt", "r")
    valNegFile = open("/home/eb/dnm_data/MultiLingual/Rus7030/val/DNM-Rus-neg.txt", "r")
    
    valTexts=[]
    for l in valPosFile:
        valTexts.append(l)
    valPosFile.close()
    for l in valNegFile:
        valTexts.append(l)
    valNegFile.close()
    
    print ('validation set size is: ' +str(len(valTexts)))
    
    y_val=[]
    with open('/home/eb/dnm_data/MultiLingual/Rus7030/val/DNM-RUS-test.cat','r') as f:
        for line in f:
            if line.strip() == "pos":
                y_val.append(1)
            else:
                y_val.append(0)
    print ('The size of validation labels is: ' + str(len(y_val)))
    
    # Build an indexed sequence for each repument
    vocabSize = 20000
    
    tokenizer = text.Tokenizer(num_words=vocabSize,
                       filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                       lower=True,
                       split=" ",
                       char_level=False)
    
    # Build the word to int mapping based on the training data
    tokenizer.fit_on_texts(trainTexts)
    
    # Build the sequence (Keras built-in) for LSTM
    trainTextsSeq = tokenizer.texts_to_sequences(trainTexts)
    print (trainTextsSeq[0])
    print (len(trainTextsSeq))
    
    valTextsSeq = tokenizer.texts_to_sequences(valTexts)
    print (valTextsSeq[0])
    print (len(valTextsSeq))
    
    # for non-sequence vectorization such as tfidf --> SVM
    trainVecMatrix = tokenizer.sequences_to_matrix(trainTextsSeq, mode='tfidf')
    #print (trainVecMatrix)
    print ('training vector length: '+str(len(trainVecMatrix)))
    print ('training vector columns: '+str(len(trainVecMatrix[0])))
    
    valVecMatrix = tokenizer.sequences_to_matrix(valTextsSeq, mode='tfidf')
    print ('validation vector length: '+str(len(valVecMatrix)))
    print ('validation vector columns: '+str(len(valVecMatrix[0])))
    
    return trainTextsSeq, valTextsSeq, y_train, y_val



