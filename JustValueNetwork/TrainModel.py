#!/usr/bin/python3

import Const

# to be incremented over time (quick beginning precise later)
EPOCHSTEPS = 100 # number of minibatch samples
EPOCHSNUM = 1000000 # number of epochs to go for
VALIDATIONSTEPS = 100 # number of minibatch samples to be given for validation (one sample given back for each generator call)
SAMPLENUM = 500 # number of data in a generator sample


from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import ELU
from keras.layers import LeakyReLU
from keras.layers import merge
from keras.layers import add
from keras.layers import concatenate
from keras.layers import Lambda
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.utils import plot_model
from time import time
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import glob, shutil
import pickle
import os
import sys
import subprocess


epdfile=None
currentline=0


def get_chess_training_positions(pickledirectory, validationset=False, includerange=False, excluderange=False):
# for multithread see: http://anandology.com/blog/using-iterators-and-generators/
    global currentline
    if not validationset:
        if epdfile==None:
            if not os.path.exists(Const.ALREADYPROCESSEDDIR): os.mkdir(Const.ALREADYPROCESSEDDIR)
        if currentline==None:
            currentline=0
    while(True):
        sn = 0
        X = [] ; Y = []
        for file in glob.glob(pickledirectory+"/*.pickle"):
            #print("Loading: "+file)
            try:
                (epd, X1, Y1) = pickle.load(open(file, "rb"))
            except:
                print("Error on file: "+str(file))
            else:
                if not (includerange and (Y1<=includerange[0] or Y1>=includerange[1])) and not (excluderange and (Y1>excluderange[0] and Y1<excluderange[1])):
                    X.append(X1)
                    if Y1 < -Const.INFINITECP:
                        Y1 = -Const.INFINITECP
                    elif Y1 > Const.INFINITECP:
                        Y1 = Const.INFINITECP
                    Y1 = Y1 / Const.INFINITECP # normalization for tanh activation!
                    Y.append(Y1)
                    sn = sn+1
            if not validationset:
                try:
                    if epdfile!=None:
                        os.remove(file) # created just when needed
                    else:
                        shutil.move(file, Const.ALREADYPROCESSEDDIR)
                except:
                    print("Error moving or removing")
            if sn>=SAMPLENUM: break
        if len(X)>0:
            yield (np.array(X),np.array(Y))
        else:
            print("Not enough elements")
            sleep(160)
        if epdfile!=None and not validationset:
            tcurrentline=currentline; currentline+=EPOCHSTEPS # almost atomic...
            print("./PrepareInput.py "+epdfile+" "+str(currentline)+" "+str(EPOCHSTEPS*SAMPLENUM))
            subprocess.call(['./PrepareInput.py',epdfile,str(tcurrentline),str(EPOCHSTEPS*SAMPLENUM)])


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


#MAIN:

# fix initial seed for reproducibility
seed = 53
np.random.seed(seed)

# create from scratch the model or load it with parameters
if os.path.isfile(Const.MODELFILE+".hdf5"):

    # load it
    model = load_model(Const.MODELFILE+".hdf5")
    (modelname,includerange,excluderange) = pickle.load(open(Const.MODELFILE+".pickle","rb")) # model attributes
    print("Loaded model and weights from file")

else:


    #@modelbegin
    #----------

    modelname = "Test-20180501"

    input = Input(shape=((8, 8, Const.NUMFEATURES) if K.image_dim_ordering()=="tf" \
               else (Const.NUMFEATURES, 8, 8)))

    net = Dense(Const.NUMFEATURES * 8, use_bias=False, activation='relu') (input)

    net = Flatten() (net)

    net = Dense(1, activation='tanh') (net)

    model = Model(inputs=input, outputs=net)

    model.compile(
        loss='mean_absolute_percentage_error', 
        optimizer='nadam', 
        metrics=["mse", "mae", "mape", "cosine"])

    # include/exclude evaluations in certain ranges of centipawns
    includerange = False
    excluderange = (-50,50) # in order to limit error percentage

    #----------
    #@modelend


# activations:
# softmax(axis=-1)
# softplus
# softsign
# elu(alpha=1.0)
# selu
# relu
# tanh
# sigmoid
# hard_sigmoid
# linear
# PReLU
# LeakyReLU

# losses:
# mean_squared_logarithmic_error, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
# squared_hinge, hinge, logcosh, categorical_hinge,
# categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy,
# kullback_leibler_divergence, poisson, cosine_proximity

# optimizers:
# SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
# RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# Adagrad(lr=0.01, epsilon=None, decay=0.0)
# Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
# Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
# Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

# metrics:
# regression: mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, cosine_proximity, rmse
# classification: binary_accuracy, categorical_accuracy, sparse_categorical_accuracy, top_k_categorical_accuracy(k), sparse_top_k_categorical_accuracy(k)


    plot_model(model, to_file=Const.MODELFILE+'.png', show_shapes=True, show_layer_names=True)
    pickle.dump((modelname,includerange,excluderange), open(Const.MODELFILE+".pickle","wb")) # model attributes
    subprocess.call(["./extract-model.sh",__file__]) # write model to txt file for document purposes
    print("Newly created model with empty weights")


if len(sys.argv)>=2: epdfile=sys.argv[1]
if len(sys.argv)>=3: currentline=int(sys.argv[2])

# TRAIN!
model.fit_generator(
    get_chess_training_positions(Const.TOBEPROCESSEDDIR, includerange=includerange, excluderange=excluderange),
    steps_per_epoch=EPOCHSTEPS,
    epochs=EPOCHSNUM,
    verbose=1,
    callbacks=[
        ModelCheckpoint(Const.MODELFILE+".hdf5", monitor='loss', verbose=1, save_best_only=False, period=1),
        CSVLogger(Const.MODELFILE+".log", separator=";", append=True),
        TensorBoard(log_dir="__logs/{}".format(time())),
        #LossHistory(),
        ],
    validation_data=get_chess_training_positions(Const.VALIDATIONDATADIR, validationset=True),
    validation_steps=VALIDATIONSTEPS,
    class_weight=None,
    max_queue_size=1000,
    workers=1,
    use_multiprocessing=False,
    initial_epoch=0 )
