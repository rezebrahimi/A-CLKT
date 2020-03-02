'''
Created on Jun 25, 2018

@author: eb
'''
from keras.preprocessing import text
from keras.preprocessing import sequence
import numpy as np

def prepareFrenchData():
    ################ French Data Preparation
    ################
    #### Build train data (list of texts) - 53 + 94 = 147
    trainPosFile_Fr = open("/home/eb/dnm_data/MultiLingual/French-Labeled/train/DNM-Fr-train-Pos.txt", "r")
    trainNegFile_Fr = open("/home/eb/dnm_data/MultiLingual/French-Labeled/train/DNM-Fr-train-Neg.txt", "r")
    trainTexts_Fr=[]
    for l in trainPosFile_Fr:
        trainTexts_Fr.append(l)
    trainPosFile_Fr.close()
    for l in trainNegFile_Fr:
        trainTexts_Fr.append(l)
    trainNegFile_Fr.close()
     
    print (trainTexts_Fr[0])
    print ('train set size is: ' +str(len(trainTexts_Fr)))
     
    ##Build the labels (Pos=1, Neg=0)
    y_train_Fr=[]
    with open('/home/eb/dnm_data/MultiLingual/French-Labeled/train/DNM-Fr-train.cat','r') as f:
        for line in f:
            if line.strip() == "pos":
                y_train_Fr.append(1)
            else:
                y_train_Fr.append(0)
    print ('The size of training labels is: ' + str(len(y_train_Fr)))
    y_train_Fr = np.array(y_train_Fr) 
     
    #### Build validation (test) data  - 21 + 43 = 64
    
    valPosFile_Fr = open("/home/eb/dnm_data/MultiLingual/French-Labeled/test/DNM-Fr-test-Pos.txt", "r")
    valNegFile_Fr = open("/home/eb/dnm_data/MultiLingual/French-Labeled/test/DNM-Fr-test-Neg.txt", "r")
     
    valTexts_Fr=[]
    for l in valPosFile_Fr:
        valTexts_Fr.append(l)
    valPosFile_Fr.close()
    for l in valNegFile_Fr:
        valTexts_Fr.append(l)
    valNegFile_Fr.close()
     
    print ('validation set size is: ' +str(len(valTexts_Fr)))
     
    y_val_Fr=[]
    with open('/home/eb/dnm_data/MultiLingual/French-Labeled/test/DNM-test-Fr.cat','r') as f:
        for line in f:
            if line.strip() == "pos":
                y_val_Fr.append(1)
            else:
                y_val_Fr.append(0)
    print ('The size of validation labels is: ' + str(len(y_val_Fr)))
    y_val_Fr = np.array(y_val_Fr)
     
    # Build an indexed sequence for each repument
    vocabSize_Fr = 20000
     
    tokenizer_Fr = text.Tokenizer(num_words=vocabSize_Fr,
                       filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                       lower=True,
                       split=" ",
                       char_level=False)
     
    # Build the word to int mapping based on the training data
    tokenizer_Fr.fit_on_texts(trainTexts_Fr)
     
    # Build the sequence (Keras built-in) for LSTM
    trainTextsSeq_Fr = tokenizer_Fr.texts_to_sequences(trainTexts_Fr)
    print (trainTextsSeq_Fr[0])
    print (len(trainTextsSeq_Fr))
    trainTextsSeq_Fr = np.array(trainTextsSeq_Fr)
    
     
    valTextsSeq_Fr = tokenizer_Fr.texts_to_sequences(valTexts_Fr)
    print (valTextsSeq_Fr[0])
    print (len(valTextsSeq_Fr))
    valTextsSeq_Fr = np.array(valTextsSeq_Fr)
     
    # for non-sequence vectorization such as tfidf --> SVM
    trainVecMatrix_Fr = tokenizer_Fr.sequences_to_matrix(trainTextsSeq_Fr, mode='tfidf')
    #print (trainVecMatrix)
    print ('training vector length: '+str(len(trainVecMatrix_Fr)))
    print ('training vector columns: '+str(len(trainVecMatrix_Fr[0])))
     
    valVecMatrix_Fr = tokenizer_Fr.sequences_to_matrix(valTextsSeq_Fr, mode='tfidf')
    print ('validation vector length: '+str(len(valVecMatrix_Fr)))
    print ('validation vector columns: '+str(len(valVecMatrix_Fr[0])))
    
    return trainTextsSeq_Fr, valTextsSeq_Fr, y_train_Fr, y_val_Fr


