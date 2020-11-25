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
    b = b.mirror()
    print("MIRRORING!")

print("Using engine", Const.ENGINE1, "vs model", lastmodel)
engine1 = chess.engine.SimpleEngine.popen_uci(Const.ENGINE1)
engine1.configure({"Threads": Const.ENGINETHREADS})
engine1.configure({"Hash": Const.HASHSIZE})
info = engine1.analyse(b, chess.engine.Limit(time=Const.MOVETIME), multipv=99)

maxscore = -9999
yms = {} # moves and score dictionary
for pos in info:
    score = pos["score"]
    score = str(score.pov(score.turn))
    if score[0]=="#":
        if score[1]=='-': score = - Const.INFINITECP
        else: score = Const.INFINITECP
    else: score = int(score)
    if score > maxscore: maxscore = score
    yms[str(pos["pv"][0]).upper()] = score/100.0 # value in pawns (and not centipawns) prevent softmax to explode

y = modeleval.predict(np.array([fe.extract_features(b)]), batch_size=1, verbose=0) 

valuesum = 0
print(maxscore, "vs", y[0][0][0])
for m in b.generate_legal_moves():
    a = str(m).upper()
    ff = ord(a[0])-65
    fr = ord(a[1])-49
    tf = ord(a[2])-65
    tr = ord(a[3])-49
    value = y[1][0][ff,fr,tf,tr]
    valuesum += value
    print(a, yms[a], "vs", value)

for m in b.generate_legal_moves():
    a = str(m).upper()
    ff = ord(a[0])-65
    fr = ord(a[1])-49
    tf = ord(a[2])-65
    tr = ord(a[3])-49
    value = y[1][0][ff,fr,tf,tr]
    valuesum += value
    print(a, yms[a], "vs", value)

print('Policy "legal moves" total: ', valuesum)
valuesumnl = 0
for i0 in range(7):
    for i1 in range(7):
        for i2 in range(7):
            for i3 in range(7):
                if not chr(i0+65)+chr(i1+49)+chr(i2+65)+chr(i3+49) in yms:
                    valuesumnl += y[1][0][i0,i1,i2,i3]
print('Policy "not legal" total: ', valuesumnl)

exit(0)