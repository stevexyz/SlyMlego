#!/usr/bin/python3

import Const

# to be incremented over time (quick beginning precise later)
EPOCHSTEPS = 100 # number of minibatch samples
EPOCHSNUM = 1000000 # number of epochs to go for
VALIDATIONSTEPS = 20 # number of minibatch samples to be given for validation (one sample given back for each generator call)
SAMPLENUM = 400 # number of data in a generator sample


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


#class LossHistory(keras.callbacks.Callback):
#    def on_train_begin(self, logs={}):
#        self.accs = []
#        self.val_accs = []
#        self.losses = []
#        self.val_losses = []
#    def on_batch_end(self, batch, logs={}):
#        self.accs.append(logs.get('acc'))
#        self.val_accs.append(logs.get('val_acc'))
#        self.losses.append(logs.get('loss'))
#        self.val_losses.append(logs.get('val_loss'))
#        #
#        plt.plot(self.accs)
#        plt.plot(self.val_accs)
#        plt.title('model accuracy')
#        plt.ylabel('accuracy')
#        plt.xlabel('epoch')
#        plt.legend(['train', 'test'], loc='upper left')
#        plt.savefig(Const.MODELFILE+'-log-acc.png')
#        plt.plot(self.losses)
#        plt.plot(self.val_losses)
#        plt.title('model loss')
#        plt.ylabel('loss')
#        plt.xlabel('epoch')
#        plt.legend(['train', 'test'], loc='upper left')
#        plt.savefig(Const.MODELFILE+'-log-loss.png')


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

    modelname = "Resnet2-v001-alfa"

    '''
    ideas from:
    - https://ctmakro.github.io/site/on_learning/resnet_keras.html
    - https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce
    -
    '''

    cardinality = 8

    def normalization_and_activation(y):
        y = BatchNormalization(axis=-1)(y)
        y = ELU()(y) # LeakyReLU()(y)
        return y

    def residual_block(y, nb_channels_in, nb_channels_out, strides=(1, 1), _project_shortcut=False):
        shortcut = y
        # we modify the residual building block as a bottleneck design to make the network more economical
        y = Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(y)
        y = normalization_and_activation(y)
        # ResNeXt (identical to ResNet when `cardinality` == 1)
        if cardinality == 1:
            y = Conv2D(nb_channels_in, kernel_size=(3, 3), strides=strides, padding='same', use_bias=False)(y)
        else:
            assert not nb_channels_in % cardinality
            _d = nb_channels_in // cardinality
            groups = []
            for j in range(cardinality):
                group = Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
                groups.append(Conv2D(_d, kernel_size=(3, 3), strides=strides, padding='same', use_bias=False)(group))
            y = concatenate(groups)
        y = normalization_and_activation(y)
        y = Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(y)
        y = BatchNormalization()(y)
        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            shortcut = Conv2D(nb_channels_out, kernel_size=(1, 1), strides=strides, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
        y = add([shortcut, y])
        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = LeakyReLU()(y)
        return y

    input_tensor = Input(shape=(8, 8, Const.NUMFEATURES))
    net_size = 512
    network = Conv2D(net_size, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(input_tensor)
    network = normalization_and_activation(network)
    #x = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    #for i in range(3):
    #    project_shortcut = True if i == 0 else False
    #    x = residual_block(x, 128, 1024, _project_shortcut=project_shortcut)
    for i in range(2):
        network = residual_block(network, net_size, net_size, strides=(1, 1))
    for i in range(2):
        network = residual_block(network, net_size, net_size, strides=(1, 1))
    network = GlobalAveragePooling2D()(network)
    network = Dense(1)(network)
    network = Activation("tanh")(network)
    model = Model(inputs=[input_tensor], outputs=[network])

    # add loss function and optimizer
    model.compile(
        loss='mean_absolute_percentage_error', # mean_squared_logarithmic_error, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
                                   # squared_hinge, hinge, logcosh, categorical_hinge,
                                   # categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy,
                                   # kullback_leibler_divergence, poisson, cosine_proximity
        optimizer='nadam', # SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
                           # RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
                           # Adagrad(lr=0.01, epsilon=None, decay=0.0)
                           # Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
                           # Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
                           # Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
                           # Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        metrics=["mse", "mae", "mape", "cosine" # mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, cosine_proximity, rmse
                 # binary_accuracy, categorical_accuracy, sparse_categorical_accuracy, top_k_categorical_accuracy(k), sparse_top_k_categorical_accuracy(k)
                ])

    # include/exclude evaluations in certain ranges of centipawns
    includerange = False
    excluderange = (-50,50) # in order to limit error percentage

    #@modelend


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
        ModelCheckpoint(Const.MODELFILE+".hdf5", monitor='acc', verbose=1, save_best_only=False, mode='max'),
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
