#!/usr/bin/python3

# to be adjusted over time (quick beginning precise later)
EPOCHSTEPS = 3 # number of minibatch samples
EPOCHSNUM = 1000000 # number of epochs to go for
VALIDATIONSTEPS = 200 # number of minibatch samples to be given for validation (one sample given back for each generator call)
SAMPLENUM = 777 # number of data in a generator sample

import Const

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from time import time
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import glob, shutil
import pickle
import os
import sys
import math
import subprocess
import logging
import optparse
import chess
import FeaturesExtraction as fe

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
        X = [] # features
        Y1 = [] # position value
        Y2 = [] # move probability matrix
        for file in glob.glob(pickledirectory+"/*.pickle"):
            #logging.debug("Loading: "+file)
            try:
                (epdposition, X1, Y11x, Y21) = pickle.load(open(file, "rb"))
            except:
                logging.warning("Error on file: "+str(file))
            else:
                board = chess.Board()
                board.set_epd(epdposition)
                if board.turn != chess.WHITE:
                    board.apply_mirror() # board features were always extracted with white perspective even if in the past black epd was also written
                if len(X1)!=8 or len(X1[0])!=8 or len(X1[0][0])!=fe.NUMFEATURES:
                    X1 = fe.extract_features(board) # if absent or not coherent on the file calculate (good to force if changing features...)
                Y11 = Y11x[0]
                if not math.isnan(Y11):
                    if not( ( includerange and (Y11<=includerange[0] or Y11>=includerange[1]) ) \
                            or ( excluderange and (Y11>excluderange[0] and Y11<excluderange[1]) ) ):
                        X.append(X1)
                        if Y11 < -Const.INFINITECP:
                            Y11 = -Const.INFINITECP
                        elif Y11 > Const.INFINITECP:
                            Y11 = Const.INFINITECP
                        Y11 = Y11 / Const.INFINITECP # normalization for tanh activation!
                        Y1.append( [Y11] )
                        Y2.append( Y21 )
                        sn = sn+1
                        logging.debug(" y("+str(Y11*Const.INFINITECP)+")", end="")
                    else:
                        logging.debug(" n("+str(Y11)+")", end="")
                else:
                    logging.warning(file+" value is nan")
            if not validationset:
                try:
                    if epdfile!=None:
                        os.remove(file) # created just when needed
                    else:
                        shutil.move(file, Const.ALREADYPROCESSEDDIR)
                except:
                    logging.warning("Error moving or removing")
            if sn>=SAMPLENUM: break
        if len(X)>0:
            yield ( np.array(X), {"value": np.array(Y1), "policy": np.array(Y2)} )
        else:
            print("Not enough elements", flush=True)
            exit(1)
        if epdfile!=None and not validationset:
            tcurrentline=currentline; currentline+=EPOCHSTEPS # almost atomic...
            print("./PrepareInput.py "+epdfile+" "+str(currentline)+" "+str(EPOCHSTEPS*SAMPLENUM))
            subprocess.call(['./PrepareInput.py',epdfile,str(tcurrentline),str(EPOCHSTEPS*SAMPLENUM)])

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

#MAIN:

LOGGING_LEVELS = {'critical': logging.CRITICAL,
                  'error': logging.ERROR,
                  'warning': logging.WARNING,
                  'info': logging.INFO,
                  'debug': logging.DEBUG}
parser = optparse.OptionParser(usage="usage: %prog [options] [epdfile [initialline]]")
parser.add_option('-l', '--logging-level', help='Logging level')
parser.add_option('-f', '--logging-file', help='Logging file name')
(options, args) = parser.parse_args()
logging_level = LOGGING_LEVELS.get(options.logging_level, logging.WARNING)
logging.basicConfig(level=logging_level, filename=options.logging_file,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
# TODO: to change all print function with "logging.debug" or "logging.info"...

# fix initial seed for reproducibility
seed = 53
np.random.seed(seed)

# create from scratch the model or load it with parameters
oldmodels = glob.glob(Const.MODELFILE+"-v*.hdf5")
if len(oldmodels)!=0:
    # load the last version
    oldmodels.sort(reverse=True)
    lastmodel = oldmodels[0]
    ver = int( lastmodel[(lastmodel.index("-v")+len("-v")):lastmodel.index("-loss")] ) # "__model-v{:0>6d}.hdf5"
    model = load_model(lastmodel)
    (modelname,includerange,excluderange) = pickle.load(open(Const.MODELFILE+".pickle","rb")) # model attributes
    print("Loaded model and weights from file ", lastmodel)
else:
    ver = 0
    #@modelbegin
    #----------
    modelname = "Policy-Mattew"
    initializer = "random_uniform"
    input_tensor = Input(shape=(8, 8, fe.NUMFEATURES))
    #----------
    network = input_tensor
    network = Conv2D(64, \
                     kernel_initializer=initializer, \
                     kernel_size=(7,7), \
                     strides=(1,1), \
                     padding='same', \
                     activation=None, \
                     use_bias=False) \
                  (network)
    network = ELU() \
                  (network)
    network = Conv2D(64, \
                     kernel_initializer=initializer, \
                     kernel_size=(5,5), \
                     strides=(1,1), \
                     padding='same', \
                     activation=None, \
                     use_bias=False) \
                  (network)
    network = ELU() \
                  (network)
    network = Conv2D(64, \
                     kernel_initializer=initializer, \
                     kernel_size=(3,3), \
                     strides=(1,1), \
                     padding='same', \
                     activation=None, \
                     use_bias=False) \
                  (network)
    network = ELU() \
                  (network)
    network = Dropout(0.05) \
                  (network)
    #----------
    network_value = network
    #network_value = Flatten() \
    #                    (network_value)
    #
    #network_value = Dense(256, \
    #                kernel_initializer=initializer, \
    #                use_bias=False, \
    #                activation=None) \
    #                    (network_value)
    #
    #network_value = ELU() \
    #                    (network_value)
    #network_value = Conv2D(64, \
    #                kernel_initializer=initializer, \
    #                kernel_size=(3,3), \
    #                strides=(1,1), \
    #                padding='same', \
    #                activation=None, \
    #                use_bias=False) \
    #                    (network_value)
    #network_value = ELU() \
    #                    (network_value)
    network_value = Conv2D(64, \
                    kernel_initializer=initializer, \
                    kernel_size=(1,1), \
                    strides=(1,1), \
                    padding='same', \
                    activation=None, \
                    use_bias=False) \
                        (network_value)
    network_value = Flatten() \
                        (network_value)   
    network_value = Dense(1, \
                    kernel_initializer=initializer, \
                    activation=None, \
                    use_bias=False) \
                        (network_value)
    network_value = Activation("tanh", \
                    name="value") \
                        (network_value)
    #----------
    network_policy = network
    #network_policy = Conv2D(64, \
    #                 kernel_initializer=initializer, \
    #                 kernel_size=(3,3), \
    #                 strides=(1,1), \
    #                 padding='same', \
    #                 activation=None, \
    #                 use_bias=False) \
    #                     (network_policy)
    #network_policy = ELU() \
    #                    (network_policy)
    network_policy = Conv2D(64, \
                     kernel_initializer=initializer, \
                     kernel_size=(1,1), \
                     strides=(1,1), \
                     padding='same', \
                     activation=None, \
                     use_bias=False) \
                         (network_policy)
    network_policy = Reshape((8,8,8,8)) \
                         (network_policy)
    network_policy = Activation("sigmoid", \
                                name="policy") \
                         (network_policy)
    #----------
    model = Model(inputs=input_tensor, outputs=[network_value, network_policy])
    model.compile(
        loss={"value": "mean_absolute_percentage_error", "policy": "categorical_crossentropy"},
        loss_weights={"value": 0.01, "policy": 1}, # value will be in pawn
        optimizer='nadam')
    # evaluations in certain ranges of centipawns can be included or excluded
    includerange = False
    excluderange = False # (-25,25)
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

    # initializers:
    # Identity(gain=1.0)
    # Constant(value=0)
    # RandomNormal(mean=0.0, stddev=0.05)
    # RandomUniform(minval=-0.05, maxval=0.05)
    # TruncatedNormal(mean=0.0, stddev=0.05)
    # VarianceScaling(scale=1.0, mode='fan_in', distribution='normal')
    # Orthogonal(gain=1.0)
    # lecun_normal()
    # lecun_uniform()
    # glorot_normal()
    # glorot_uniform()
    # he_normal()
    # he_uniform()

    # losses:
    # mean_squared_logarithmic_error 
    # mean_squared_error 
    # mean_absolute_error 
    # mean_absolute_percentage_error
    # squared_hinge 
    # hinge 
    # logcosh 
    # categorical_hinge
    # categorical_crossentropy
    # sparse_categorical_crossentropy
    # binary_crossentropy
    # kullback_leibler_divergence
    # poisson
    # cosine_proximity

    # optimizers:
    # SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    # RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    # Adagrad(lr=0.01, epsilon=None, decay=0.0)
    # Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    # Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    # Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    # metrics regression: 
    # mean_squared_error, 
    # mean_absolute_error, 
    # mean_absolute_percentage_error, 
    # cosine_proximity, 
    # rmse
    
    # metrics classification: 
    # binary_accuracy, 
    # categorical_accuracy, 
    # sparse_categorical_accuracy, 
    # top_k_categorical_accuracy(k), 
    # sparse_top_k_categorical_accuracy(k)

    plot_model(model, to_file=Const.MODELFILE+'.png', show_shapes=True, show_layer_names=True)
    pickle.dump((modelname,includerange,excluderange), open(Const.MODELFILE+".pickle","wb")) # model attributes
    subprocess.call(["./extract-model.sh",__file__]) # write model to txt file for document purposes
    print("Newly created model with empty weights")

if len(args)>=1: epdfile=args[0]
if len(args)>=2: currentline=int(args[1])

# TRAIN!
model.fit(
    get_chess_training_positions(Const.TOBEPROCESSEDDIR, includerange=includerange, excluderange=excluderange),
    steps_per_epoch=EPOCHSTEPS,
    epochs=EPOCHSNUM,
    verbose=1,
    callbacks=[
        ModelCheckpoint(Const.MODELFILE+"-v{epoch:05d}-loss{val_loss:09.2f}.hdf5", monitor='loss', verbose=1, save_best_only=False, period=1),
        CSVLogger(Const.MODELFILE+"-log.csv", separator=";", append=True),
        TensorBoard(log_dir="__logs/{}".format(time())),
        #LossHistory(),
        ],
    validation_data=get_chess_training_positions(Const.VALIDATIONDATADIR, validationset=True),
    validation_steps=VALIDATIONSTEPS,
    class_weight=None,
    max_queue_size=1000,
    workers=1,
    use_multiprocessing=False,
    initial_epoch=ver )
