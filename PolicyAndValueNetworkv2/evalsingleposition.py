#!/usr/bin/python3

import Const
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
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
import chess
import chess.engine
import FeaturesExtraction as fe

epdfile=None
currentline=0

#MAIN:

if len(sys.argv)==1:
    print('Usage: python3 ', sys.argv[0], '<fenstring>')
    #fenstring = '1rnb1q2/p5pk/PpQ4p/3pPp2/7P/1P2B1P1/5PB1/R2R2K1 w - -'
    exit(1)
else:
    fenstring=sys.argv[1]

oldmodels = glob.glob(Const.MODELFILE+"-v*.hdf5")
if len(oldmodels)!=0:
    # load the last version
    oldmodels.sort(reverse=True)
    lastmodel = oldmodels[0]
    modeleval = load_model(lastmodel)
    print("Loaded model and weights from file ", lastmodel)
else:
    raise ValueError("No model file found!")

b = chess.Board()
b.set_fen(fenstring)
if b.turn != chess.WHITE:
    b = b.board.mirror()

y = modeleval.predict(np.array([fe.extract_features(b)]), batch_size=1, verbose=0) 

print(y[0][0][0])
for m in b.generate_legal_moves():
    a = str(m).upper()
    ff = ord(a[0])-65
    fr = ord(a[1])-49
    tf = ord(a[2])-65
    tr = ord(a[3])-49
    print(a, y[1][0][ff,fr,tf,tr])

