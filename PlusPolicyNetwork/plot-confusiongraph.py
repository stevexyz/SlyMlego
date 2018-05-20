#!/usr/bin/python3

import Const
import matplotlib.pyplot as plt

import numpy as np
import glob, shutil
import pickle
import os
import sys
import time

from keras.models import load_model
from FeaturesExtraction import extract_features
import numpy as np
import chess.uci



if len(sys.argv)<2:
    print("Plot the confusion graph using validation samples")
    print("Usage: "+sys.argv[0]+" [<numsamples> [<modelfile>]]")
    exit(1)

if len(sys.argv)>=2:
    numsamples = int(sys.argv[1])
else:
    numsamples = 999999999

if len(sys.argv)>=3: 
    modeleval = load_model(sys.argv[2])                               
    print("Using model "+sys.argv[2])
else:
    modeleval = load_model(Const.MODELFILE+".hdf5") 

xcoords=[]
ycoords=[]
starttime = time.time()
b = chess.Board()
for file in glob.glob(Const.VALIDATIONDATADIR+"/*.pickle"):
    numsamples -= 1
    if numsamples<=0: break
    (epd, X, Y1, Y2) = pickle.load(open(file, "rb"))
    b.set_epd(epd)
    val = modeleval.predict(np.array([extract_features(b)]), batch_size=1)[0][0]
    ycoords.append(val)
    if Y1[0] < -Const.INFINITECP:
        xcoords.append(-Const.INFINITECP)
    elif Y1[0] > Const.INFINITECP:
        xcoords.append(Const.INFINITECP)
    else:
        xcoords.append(Y1[0])
    print(file+" "+epd+" sf:"+str(Y1[0])+" nn:"+str(val))
elapsed=time.time()-starttime
print(str(int(sys.argv[1]))+" samples in "+str(elapsed)+" seconds = "+str(int(sys.argv[1])/elapsed)+" nodes/sec")
    
plt.autoscale = False
plt.axis([-Const.INFINITECP, Const.INFINITECP, -Const.INFINITECP, Const.INFINITECP])
plt.plot(xcoords,ycoords,"ro")
plt.grid(True)
plt.axes().set_aspect('equal')
plt.ylabel('ann model eval')
plt.xlabel('std engine eval')
plt.title('"Confusion plot" of model evaluations')

plt.show()

