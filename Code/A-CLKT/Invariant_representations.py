'''
Created on Jun 21, 2018

@author: eb
'''
from __future__ import print_function, division

from keras import backend as K
from sklearn import metrics
if 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    
import json
from keras.layers import LSTM,LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Masking
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from keras.models import load_model
from keras.models import model_from_json

import matplotlib.pyplot as plt
import h5py
from keras.utils import print_summary
import sys
import os
from keras.utils import plot_model

os.environ['CUDA_VISIBLE_DEVICES']='5'  
import numpy as np
from tensorflow.python.ops.nn_ops import leaky_relu

import myPerf
from keras import optimizers

lang='It'
fold_i=3 #2
def myprint(s):
    with open('modelsummary.txt','w+') as f:
        print(s, file=f)

class GAN():
    def __init__(self):
        self.rep_size = 150
        self.rep_dim = 200
        self.rep_shape = (self.rep_size, self.rep_dim)
        self.latent_dim = 100

        optimizer = Adam(0.00002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        
        

        # The generator takes noise as input and generates imgs
        self.generator1 = self.build_generator1()
        rep1 = Input(shape=(150,64,))
        generated_rep1 = self.generator1(rep1)
        self.discriminator.trainable = False
        # The discriminator takes generated images as input and determines validity
        validity1 = self.discriminator(generated_rep1)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined1 = Model(rep1, validity1)
        self.combined1.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        self.generator2 = self.build_generator2()
        rep2 = Input(shape=(150,64,))
        generated_rep2 = self.generator2(rep2)
        self.discriminator.trainable = False
        validity2 = self.discriminator(generated_rep2)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined2 = Model(rep2, validity2)
        self.combined2.compile(loss='binary_crossentropy', optimizer=optimizer)
        

    def build_generator1(self):

        model = Sequential()
        #model.add(Masking(mask_value=0., input_shape=self.rep_shape))
#         model.add(Conv1D(256, strides=1, kernel_size=1, activation=leaky_relu,input_shape=self.rep_shape))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(Conv1D(512, strides=1, kernel_size=1, activation=leaky_relu,input_shape=self.rep_shape))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(BatchNormalization(momentum=0.8))
        model.add(Bidirectional(LSTM(100, return_sequences=True),input_shape=(150,64,)))
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        #model.add(LSTM(200, return_sequences=True))
        #model.add(LSTM(200, return_sequences=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        for l in model.layers:
            l.supports_masking = True
        #plot_model(model, to_file='model.png')
        #plot_model(model, to_file='model.png', show_shapes=True,show_layer_names=True)
        #for layer_i in model.layers:
        #    print(layer_i)
        #print(model.to_yaml())
        #print(model.summary())

#         model = Sequential()
#         model.add(Conv1D(256, strides=2, kernel_size=3, activation=leaky_relu,input_shape=self.rep_shape))
#         model.add(Dense(256, input_dim=self.rep_shape))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(Dense(512))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(Dense(1024))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(Dense(np.prod(self.rep_shape), activation='tanh'))
#         model.summary()
#         print ("QQQQQQQ")
#         print (np.prod(self.rep_shape))
#         print (self.rep_shape)
#         model.add(Reshape(self.rep_shape))


        rep1 = Input(shape=(150,64,))
        generated_rep1 = model(rep1)

        return Model(rep1, generated_rep1)
    
    def build_generator2(self):
        
        model = Sequential()
        #model.add(Masking(mask_value=0., input_shape=self.rep_shape))
#         model.add(Conv1D(256, strides=1, kernel_size=1, activation=leaky_relu,input_shape=self.rep_shape))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(Conv1D(512, strides=1, kernel_size=1, activation=leaky_relu,input_shape=self.rep_shape))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(BatchNormalization(momentum=0.8))
        model.add(Bidirectional(LSTM(100, return_sequences=True),input_shape=(150,64,)))
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        #model.add(LSTM(200, return_sequences=True))
        #model.add(LSTM(200, return_sequences=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        for l in model.layers:
            l.supports_masking = True
        
#         model = Sequential()
#         model.add(Dense(256, input_dim=self.rep_shape))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(Dense(512))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(Dense(1024))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(Dense(np.prod(self.rep_shape), activation='tanh'))
#         model.add(Reshape(self.rep_shape))
#        model.summary()
        #print(model.to_yaml())
        #print(model.summary())
        #exit()

        rep2 = Input(shape=(150,64,))
        generated_rep2 = model(rep2)

        return Model(rep2, generated_rep2)    

    def build_discriminator(self):

        #model.add(Masking(mask_value=0., input_shape=self.rep_shape))
        
        model = Sequential()
        #model.add(Embedding(10000, 32, input_length=200))

        #model.add(LSTM(32))
        print(self.rep_shape)
        #exit()
        model.add(Bidirectional(LSTM(128,return_sequences=True), input_shape=self.rep_shape))
        model.add(Bidirectional(LSTM(32,return_sequences=False)))
        #model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1,activation='sigmoid'))
        '''
        model = Sequential()
        model.add(Flatten(input_shape=self.rep_shape))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        #model.summary()
        '''
        #print(self.rep_shape)
        #exit()
        #print(model.to_yaml())
        #print(model.summary())
        #exit()

        rep = Input(shape=self.rep_shape)
        validity = model(rep)

        return Model(rep, validity)

    def train(self, epochs, batch_size=32,fold_i=1):

        # Load the dataset
        f1 = h5py.File("./output_adv_NoMT/En_states_%d.hdf5"%fold_i, 'r')

        # List all groups
        print("Keys: %s" % f1.keys())
        states_En = list(f1.keys())[0]
        
        # Get the data
        rep1 = list(f1[states_En])
        rep1=np.array(rep1)
        
        f2 = h5py.File("./output_adv_NoMT/%s_states_%d.hdf5"%(lang,fold_i), 'r')

        # List all groups
        print("Keys: %s" % f2.keys())
        states_Ru = list(f2.keys())[0]
        
        # Get the data
        rep2 = list(f2[states_Ru])
        rep2 = np.array(rep2)
        # Rescale -1 to 1
        #X_train = X_train / 127.5 - 1.
        #rep1 = np.expand_dims(rep1, axis=3)
        #rep2 = np.expand_dims(rep2, axis=3)

        # Adversarial ground truths
        r1_label = np.ones((batch_size, 1))
        r2_label = np.zeros((batch_size, 1))

        loss_log=[]
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, rep1.shape[0], batch_size)
            reps1 = rep1[idx]
            idx = np.random.randint(0, rep2.shape[0], batch_size)
            reps2 = rep2[idx]

            # noise = np.random.normal(0, 1, (batch_size, self.latent_dim)) # We do not need noise for initialization

            # Generate a batch of new images
            gen_reps1 = self.generator1.predict(reps1)
            gen_reps2 = self.generator2.predict(reps2)

            # Train the discriminator
            for train_n in range(1):
                d_loss_real = self.discriminator.train_on_batch(gen_reps1, r1_label)
                d_loss_fake = self.discriminator.train_on_batch(gen_reps2, r2_label)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            #noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss1 = self.combined1.train_on_batch(reps1, r2_label)
            g_loss2 = self.combined2.train_on_batch(reps2, r1_label)
            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss1: %f] [G loss2: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss1, g_loss2))

            loss_log.append((d_loss[0], 100*d_loss[1], g_loss1, g_loss2))

        loss_log=np.array(loss_log)
        return loss_log
if __name__ == '__main__':
    gan = GAN()

    

    loss_log=gan.train(epochs=20, batch_size=128,fold_i=fold_i)
    np.savetxt('./results_performance/loss_log_%s.txt'%lang,loss_log,fmt='%.4f')
    # Save generator 2
    gan.generator2.save('./generators_saved/gen2_saved_ru_1k.h5')
    
    # Apply Generator 2 on Russian train and test to project it into new obtained representation
    f3 = h5py.File("./output_adv_NoMT/%s_states_%d.hdf5"%(lang,fold_i), 'r')
    states_train_Ru = list(f3.keys())[0]
    rep_Ru_train = list(f3[states_train_Ru])
    rep_Ru_train = np.array(rep_Ru_train)
    projected_rep_Ru_train = gan.generator2.predict(rep_Ru_train)
    hf = h5py.File("output_adv_NoMT/projected_rep_%s_train_1k.hdf5"%(lang), "w")
    hf.create_dataset('states_Ru', data=projected_rep_Ru_train)
    hf.close()
    
    # Load Russian_test rep
    f4 = h5py.File("./output_adv_NoMT/%s_test_states_%d.hdf5"%(lang,fold_i), 'r')
    states_test_Ru = list(f4.keys())[0]
    rep_Ru_test = list(f4[states_test_Ru])
    rep_Ru_test = np.array(rep_Ru_test)
    projected_rep_Ru_test = gan.generator2.predict(rep_Ru_test)
    hf = h5py.File("output_adv_NoMT/projected_rep_%s_test_1k.hdf5"%lang, "w")
    hf.create_dataset('states_Ru', data=projected_rep_Ru_test)
    hf.close()
    
    # Apply generator2 (Build a model that get representations and outputs class label. Train it on rep of Russian_train and test it on rep of Russian_val)
    #_, _, y_train_Ru, y_val_Ru = PrepareRussian.prepareRussianData()
    #print(y_train_Ru)
    y_train_Ru=np.loadtxt('output_adv_NoMT/%s_train_label_%d.txt'%(lang,fold_i))
    y_val_Ru=np.loadtxt('output_adv_NoMT/%s_test_label_%d.txt'%(lang,fold_i))
    #print(y_train_Ru)
    print(len(y_train_Ru))

    # Load Russian_train rep again
    f_train = h5py.File("./output_adv_NoMT/projected_rep_%s_train_1k.hdf5"%lang, 'r')
    states_train = list(f_train.keys())[0]
    projed_rep_Ru_train = list(f_train[states_train])
    projed_rep_Ru_train = np.array(projed_rep_Ru_train)
    

    # Load Russian test projected data
    f_test = h5py.File("./output_adv_NoMT/projected_rep_%s_test_1k.hdf5"%lang, 'r')
    states_test = list(f_test.keys())[0]
    projed_rep_Ru_test = list(f_test[states_test])
    projed_rep_Ru_val = np.array(projed_rep_Ru_test)


    
    # Building model 
    myInput = Input(shape=(150,200))
    LSTM_Russian = Bidirectional(LSTM(100,return_sequences=False))(myInput)
    #LSTM_Russian=Bidirectional(LSTM(32, return_sequences=False))(LSTM_Russian)
    predictions = Dense(1, activation='sigmoid')(LSTM_Russian)
    model_Ru = Model(inputs=myInput, outputs=predictions)
    model_Ru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model_Ru.summary())

    print(len(projed_rep_Ru_train))
    print(len(y_train_Ru))
    class_weight = {0: 1.,1: 1}
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model_Ru.fit(projed_rep_Ru_train, y_train_Ru, epochs=10, batch_size=32,
        validation_data=[projed_rep_Ru_val, y_val_Ru],
        callbacks=[early_stopping],class_weight=class_weight)
    
    # Load Russian test projected data
    
    pred_train = model_Ru.predict(projed_rep_Ru_train, batch_size= 32)
    #pred_train_p = model_Ru.predict_proba(projed_rep_Ru_train, batch_size= 32)

    #print (pred_res)

    #pred_train = map(lambda x: 1 if x[1]>x[0] else 0, pred_train)
    #pred_res = map(lambda x: 1 if x>0.5 else 0, pred_res)
    
    #print(pred_train_p)
   
    auc=metrics.roc_auc_score(y_train_Ru,pred_train)
    #print(pred_train)
    performance=[]
    for threshold in np.arange(0.1,1,0.001):
        label_pr=[]
        for i in range(len(pred_train)):
          if pred_train[i]>threshold:
            label_pr.append(1)
          else:
            label_pr.append(0)

        label_pr=np.array(label_pr)

        #label_pr = model.predict_classes(valTextsSeq)

        acc=metrics.accuracy_score(y_train_Ru,label_pr)
        rec=metrics.recall_score(y_train_Ru,label_pr)
        prec=metrics.precision_score(y_train_Ru,label_pr)
        f1=metrics.f1_score(y_train_Ru,label_pr)
        #print(label_pr)
        #print(y_train_Ru)
        performance.append([threshold,acc,prec,rec,f1,auc])
    performance=np.array(performance)

    performance_sort=performance[performance[:, -2].argsort()]

    print(performance_sort[:-10])
    
    
    pred_res = model_Ru.predict(projed_rep_Ru_val, batch_size= 32)
    auc=metrics.roc_auc_score(y_val_Ru,pred_res)
    #print(pred_res)
    performance=[]
    for threshold in np.arange(0.0,1,0.01):
        label_pr=[]
        for i in range(len(pred_res)):
          if pred_res[i]>threshold:
            label_pr.append(1)
          else:
            label_pr.append(0)

        label_pr=np.array(label_pr)

        #label_pr = model.predict_classes(valTextsSeq)

        acc=metrics.accuracy_score(y_val_Ru,label_pr)
        rec=metrics.recall_score(y_val_Ru,label_pr)
        prec=metrics.precision_score(y_val_Ru,label_pr)
        f1=metrics.f1_score(y_val_Ru,label_pr)
        #print(label_pr)
        #print(y_train_Ru)
        performance.append([threshold,acc,prec,rec,f1,auc])
    performance=np.array(performance)

    performance_sort=performance[performance[:, -2].argsort()]

    print(performance_sort[:-10])
    

