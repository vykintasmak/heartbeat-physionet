"""
Created by Vykinitas Maknickas (vykintas.mak[at]gmail.com)
LinkedIn: https://lt.linkedin.com/in/vykintasmak

@author: Vykintas Maknickas

To try out the code you need to create two separate folders for Normal and Abnormal .wav files and insert them into code below
To launch it:
python hb_prediction_v2.py -o train
python hb_prediction_v2.py -o test -f filename

"""

from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
import hbdata_predict_v2 as dataimport
from keras.models import model_from_json
import argparse

def train(normal_folder='',
          abnormal_folder='',
          max_features=5000, 
          maxlen=100, 
          batch_size=32, 
          embedding_dims=100, 
          nb_filter=90, 
          hidden_dims=256, 
          nb_epoch=500, 
          nb_classes=2, 
          optimizer='sgd', 
          loss='categorical_crossentropy',
          test_split=0.2,
          seed=1955,
          model_json='hb_model_orthogonal_experiment_norm.json',
          weights='hb_weights_orthogonal_experiment_norm.hdf5',
          load_weights=False,
          normal_path='',
          abnormal_path=''):

    print('Loading data...')
    #Loads training data from specified folders for normal and abnormal sound files. 
    #Transfrorms data using short-time Fourier transform, logscales the result and splits it into 129x129 squares.
    #Randomizes data.
    #Splits the data into train and test arrays
    (X_train, y_train), (X_test, y_test) = dataimport.load_data(normal_path=normal_path, abnormal_path=abnormal_path, test_split=test_split, width=129, height=256, seed=seed)
    
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)
    
    print('Build model...')
    model = Sequential()
    
    # We start off with using Convolution2D for a frame
    # The filter is 3x57
    model.add(Convolution2D(nb_filter=nb_filter,
                            nb_row=3,
                            nb_col=57,
                            init='orthogonal',
                            border_mode='valid',
                            W_regularizer=l2(0.0001),
                            W_constraint = maxnorm(2),
                            input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    model.add(Activation('relu'))
    
    # dropout to reduce overfitting:
    model.add(Dropout(0.25))
    
    # we use standard max pooling (halving the output of the previous layer):
    model.add(MaxPooling2D(pool_size=(3, 4), strides=(1, 3)))
    
    # the second convolution layer is 1x3
    model.add(Convolution2D(nb_filter, 
                            nb_row=1,
                            nb_col=3,
                            init='orthogonal', 
                            W_regularizer=l2(0.0001), 
                            W_constraint=maxnorm(2)))

    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    # we use max pooling again:
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))
    
    # We flatten the output of the conv layer,
    # so that we can add a vanilla dense layer:
    model.add(Flatten())

    # we add two hidden layers:
    # increasing number of hidden layers may increase the accuracy, current number is designed for the competition 
    model.add(Dense(hidden_dims, 
                    init='orthogonal', 
                    W_regularizer=l2(0.0001), 
                    W_constraint = maxnorm(2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(hidden_dims, 
                    init='orthogonal', 
                    W_regularizer=l2(0.0001), 
                    W_constraint = maxnorm(2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    # We project onto a binary output layer to determine the category (Currently: normal/abnormal, but you can try train on the exact abnormality also)
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    # you can load pre-trained weights to quicken the training
    if(load_weights):
        model.load_weights(weights)
    
    # Prints summary of the model
    model.summary()
    
    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    
    # Saving model to Json (its easier to test it this way)    
    json_string = model.to_json()
    open(model_json, 'w').write(json_string)
    
    # Each time the loss will drop it will save weights file
    checkpointer = ModelCheckpoint(filepath=weights, verbose=1, save_best_only=True)
    
    # Start training
    model.fit(X_train, Y_train, batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              callbacks=[checkpointer],
              validation_data=(X_test, Y_test))
              
              
    return True


def write_answer(filename, result, resultfile="answers.txt"):
        fo = open(resultfile, 'a')
        fo.write(str(filename) + "," + str(result) + "\n")
        fo.close()

        return True    
def test(filename, model_json='hb_model_orthogonal_experiment_norm.json', 
         weights='hb_weights_orthogonal_experiment_norm.hdf5', 
         optimizer='sgd', 
         loss='categorical_crossentropy'):
    print('Build model...')
    
    #Loads model from Json file
    model = model_from_json(open(model_json).read())
    
    #Loads pre-trained weights for the model
    model.load_weights(weights)

    #Compiles the model
    model.compile(loss=loss, optimizer=optimizer)
    
    #loads filename, transfrorms data using short-time Fourier transform, logscales the result and splits it into 129x129 squares
    X = dataimport.data_from_file(filename=str(filename)+".wav", width=129, height=256, max_frames=10)
    
    predictions = np.zeros(len(X))
    z = 0
    
    #Makes predictions for each 129x129 square
    for frame in X:
        predict_frame = np.zeros((1, 3, 129, 129))
        predict_frame[0] = frame
        predictions_all = model.predict_proba(predict_frame, batch_size=batch_size)
        predictions[z] = predictions_all[0][1]

        z += 1
    
    #Averages the results of per-frame predictions    
    average = np.average(predictions)
    average_prediction = round(average)
    		
    #Prints the result
    if int(average_prediction) == 0.0:
        #append file with -1
        write_answer(filename=filename, result="-1")
        
        print('Result for '+filename+': '+'Normal (-1)')
        
    else:
        #append file with 1
        write_answer(filename=filename, result="1")    
        
        print('Result for '+filename+': '+'Abnormal (1)')        
    
    return int(average_prediction)
    
parser = argparse.ArgumentParser(description='This is a script to train and test PhysioNet 2016 challenge data.')
parser.add_argument('-o','--option', help='Options may be train or test',required=True)
parser.add_argument('-i','--inputfile', help='Input file name (full path) without .wav ending',required=False)

args = parser.parse_args()
# set parameters:
max_features = 5000
maxlen = 100
batch_size = 32
embedding_dims = 100
nb_filter = 90
hidden_dims = 256
nb_epoch = 500
nb_classes = 2
sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9)
loss = 'categorical_crossentropy'
model_json='hb_model_orthogonal_experiment_norm.json'
weights='hb_weights_orthogonal_experiment_norm.hdf5'
seed = 1995
test_split = 0.2
normal_path = '/your/folder/here/'
abnormal_path = '/your/abnormal/folder/here/'

if(args.option == 'train'):
    train(max_features=max_features, 
              maxlen=maxlen, 
              batch_size=batch_size, 
              embedding_dims=embedding_dims, 
              nb_filter=nb_filter,  
              hidden_dims=hidden_dims, 
              nb_epoch=nb_epoch, 
              nb_classes=nb_classes, 
              optimizer=sgd, 
              loss=loss,
              test_split=test_split,
              seed=seed,
              model_json=model_json,
              weights=weights,
              load_weights=False,
              normal_path=normal_path,
              abnormal_path=abnormal_path)
elif(args.option == 'test'):
    test(model_json=model_json, 
         weights=weights, 
         optimizer=sgd, 
         loss=loss, 
         filename=str(args.inputfile))
else:
    print('You need to choose between train and test arguments')

