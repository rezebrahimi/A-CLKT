import numpy
import h5py
numpy.random.seed(7)
from keras.models import Sequential

from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional,GRU

from keras.layers import Input
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.preprocessing import text
from keras import optimizers
from keras.callbacks import EarlyStopping

from keras.utils import plot_model

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

import myPerf
import os

from keras import regularizers

os.environ['CUDA_VISIBLE_DEVICES']='3'

lang='En'


################ Russian Data Preparation
################
#### Build train data (list of texts) - 380
trainPosFile = open("./dnm_data/MultiLingual/Setting1_trainWithExpert/DNM-train-pos.txt", "r")
trainNegFile = open("./dnm_data/MultiLingual/Setting1_trainWithExpert/DNM-train-neg.txt", "r")

trainTexts=[]
y_train=[]
for l in trainPosFile:
    trainTexts.append(l)
    y_train.append(1)
trainPosFile.close()
for l in trainNegFile:
    trainTexts.append(l)
    y_train.append(0)
trainNegFile.close()

print (trainTexts[0])
print ('train set size is: ' +str(len(trainTexts)))

##Build the labels (Pos=1, Neg=0)

print ('The size of training labels is: ' + str(len(y_train)))


#### Build validation (test) data  - 55 + 117= 172
valPosFile = open("./dnm_data/MultiLingual/Setting1_trainWithExpert/DNM-test-pos.txt", "r")
valNegFile = open("./dnm_data/MultiLingual/Setting1_trainWithExpert/DNM-test-neg.txt", "r")

valTexts=[]
y_val=[]
for l in valPosFile:
    valTexts.append(l)
    y_val.append(1)
valPosFile.close()
for l in valNegFile:
    valTexts.append(l)
    y_val.append(0)
valNegFile.close()

print ('validation set size is: ' +str(len(valTexts)))



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

################ English Data Preparation
#### Build train data (list of texts)

max_features=10000

## Building models
max_rep_length = 150

class_weight = {0: 1.,1: 1}

N_train=len(trainTextsSeq)
TextsSeq_E=[]
Y_E=[]
temp_i=0

for i in range(N_train):
  temp_i+=1
  if y_train[i]==1:
    for j in range(1):
      TextsSeq_E.append(trainTextsSeq[i])
      Y_E.append(1)
  elif y_train[i]==0:
    if temp_i%1==0:
      TextsSeq_E.append(trainTextsSeq[i])
      Y_E.append(0)
    else:
      pass
  
  else:
    print('error')
    exit()

N_val=len(valTextsSeq)

temp_i=0
for i in range(N_val):
  if y_val[i]==1:
    for j in range(1):
      TextsSeq_E.append(valTextsSeq[i])
      Y_E.append(1)
  elif y_val[i]==0:
    if temp_i%1==0:
      TextsSeq_E.append(valTextsSeq[i])
      Y_E.append(0)
    else:
      pass
      #trainTextsSeq_E.append(valTextsSeq[i])
      #train_Y_E.append(0)
    temp_i+=1
  else:
    print('error')
    exit()



N_new=len(Y_E)
index=numpy.arange(N_new)
numpy.random.shuffle(index)
index=list(index)
TextsSeq_E=[TextsSeq_E[i] for i in index]
Y_E=[Y_E[i] for i in index]
TextsSeq_E=TextsSeq_E
Y_E=Y_E

fold=5
fold_size=int(N_new/fold)

folds=5
result_cv=[]
each_fold=0
for fold_i in range(0,N_new,fold_size):

  trainTextsSeq=TextsSeq_E[:fold_i]+TextsSeq_E[fold_size+fold_i:]
  valTextsSeq=TextsSeq_E[fold_i:fold_i+fold_size]

  y_train=Y_E[:fold_i]+Y_E[fold_size+fold_i:]
  y_val=Y_E[fold_i:fold_i+fold_size]

  if len(y_val)<3:
    continue

  trainTextsSeq = sequence.pad_sequences(trainTextsSeq, maxlen=max_rep_length)
  valTextsSeq = sequence.pad_sequences(valTextsSeq, maxlen=max_rep_length)


  y_val=numpy.array(y_val)
  y_train=numpy.array(y_train)

  ratio_test=0.5
  temp_train=numpy.concatenate((trainTextsSeq,y_train.reshape((-1,1))),axis=-1)

  unq, unq_idx = numpy.unique(temp_train[:, -1], return_inverse=True)
  unq_cnt = numpy.bincount(unq_idx)
  cnt = numpy.max(unq_cnt)
  out = numpy.empty((cnt+int(cnt*ratio_test),) + temp_train.shape[1:], temp_train.dtype)

  indices = numpy.random.choice(numpy.where(unq_idx==0)[0], cnt)
  out[0:cnt] = temp_train[indices]
  indices = numpy.random.choice(numpy.where(unq_idx==1)[0], int(ratio_test*cnt))
  out[cnt:int((1+ratio_test)*cnt)] = temp_train[indices]

  numpy.random.shuffle(out)
  selected=int(0.4*len(out))
  trainTextsSeq=out[:selected,:-1]
  y_train=out[:selected,-1]

  model = Sequential()
  model.add(Embedding(max_features, 64, input_length=max_rep_length))
  model.add((LSTM(64,return_sequences=True)))
  #model.add((LSTM(200,return_sequences=True,kernel_regularizer=regularizers.l2(0.0001), 
  #  recurrent_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.0001))))
  model.add((LSTM(64,return_sequences=False,kernel_regularizer=regularizers.l2(0.0001), 
    recurrent_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.0001))))
  #model.add(Dropout(0.5))
  #model.add(Bidirectional(GRU(64)))
  #model.add((LSTM(64,return_sequences=True)))

  #model.add(Dropout(0.5))
  model.add(Dense(1, activation='sigmoid'))

  early_stopping = EarlyStopping(monitor='val_loss', patience=3)

  # try using different optimizers and different optimizer configs
  model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

  print('Train...')
  print(y_train.shape)
  print(y_val.shape)
  print(trainTextsSeq.shape)
  print(valTextsSeq.shape)


  model.fit(trainTextsSeq, y_train,
            batch_size=16,
            epochs=10,
            validation_data=[valTextsSeq, y_val],
            callbacks=[early_stopping],class_weight=class_weight)

  label_pr_p = model.predict(valTextsSeq)
  auc=metrics.roc_auc_score(y_val,label_pr_p)

  #print(label_pr_p)
  performance=[]
  for threshold in numpy.arange(0.1,1,0.01):
      label_pr=[]
      for i in range(len(label_pr_p)):
        if label_pr_p[i]>threshold:
          label_pr.append(1)
        else:
          label_pr.append(0)

      label_pr=numpy.array(label_pr)

      #label_pr = model.predict_classes(valTextsSeq)

      acc=metrics.accuracy_score(y_val,label_pr)
      rec=metrics.recall_score(y_val,label_pr)
      prec=metrics.precision_score(y_val,label_pr)
      f1=metrics.f1_score(y_val,label_pr)
      #print(label_pr)
      #print(y_val)
      performance.append([threshold,acc,prec,rec,f1,auc])
  performance=numpy.array(performance)

  performance_sort=performance[performance[:, -2].argsort()]

  print(performance_sort)

  label_pr_p = model.predict(valTextsSeq)
  label_pr = model.predict_classes(valTextsSeq)
  auc=metrics.roc_auc_score(y_val,label_pr_p)
  auc2=metrics.average_precision_score(y_val,label_pr_p)
  acc=metrics.accuracy_score(y_val,label_pr)
  rec=metrics.recall_score(y_val,label_pr)
  prec=metrics.precision_score(y_val,label_pr)
  f1=metrics.f1_score(y_val,label_pr)
  #print(label_pr)
  #print(y_val)
  result_cv.append(performance_sort[-1])

  numpy.savetxt('output_adv_NoMT/%s_train_label_%d.txt'%(lang,each_fold),y_train,fmt='%d')
  numpy.savetxt('output_adv_NoMT/%s_test_label_%d.txt'%(lang,each_fold),y_val,fmt='%d')

  model.layers.pop();
  #Save the states via predict for Russian
  inp = model.input

  print(model.layers)

  out = model.layers[-2].output
  model_RetreiveS = Model(inp, out)
  #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  states_model = model_RetreiveS.predict(trainTextsSeq, batch_size=32, verbose=True)
  print(states_model.shape)

  hf = h5py.File("output_adv_NoMT/%s_states_%d.hdf5"%(lang,each_fold), "w")
  hf.create_dataset('states', data=states_model)
  hf.close()

  ###### Obtain Russian Representation for Russain Test (Applying the learned model from Russian train on Russian test data
  states_model_test = model_RetreiveS.predict(valTextsSeq, batch_size=32, verbose =True)
  print(states_model_test.shape)

  hf = h5py.File("output_adv_NoMT/%s_test_states_%d.hdf5"%(lang,each_fold), "w")
  hf.create_dataset('states_%s'%lang, data=states_model_test)
  hf.close()
  each_fold+=1

#numpy.savetxt('results_performance/%s_LSTM.txt'%lang,numpy.array(result_cv),fmt='%.4f')
result_cv = (numpy.mean(numpy.array(result_cv),axis=0))

print(result_cv)