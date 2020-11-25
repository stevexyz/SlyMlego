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
import chess.engine

#if len(sys.argv)<2:
#    print("Plot the confusion graph using validation samples")
#    print("Usage: "+sys.argv[0]+" [<numsamples> [<modelfile>]]")
#    exit(1)

if len(sys.argv)>=2:
    numsamples = int(sys.argv[1])
else:
    numsamples = 400

if len(sys.argv)>=3: 
    lastmodel = sys.argv[2]                               
else:
    allmodels = glob.glob(Const.MODELFILE+"-v*.hdf5")
    if len(allmodels)!=0:
        # load the last version
        allmodels.sort(reverse=True)
        lastmodel = allmodels[0]
    else:
        print("No model found")
        exit(1)
modeleval = load_model(lastmodel) 
print("Using model "+lastmodel)
    
xcoords=[]
ycoords=[]
starttime = time.time()
b = chess.Board()
for file in glob.glob(Const.VALIDATIONDATADIR+"/*.pickle"):
    numsamples -= 1
    if numsamples<=0: break
    (epd, X, Y1, Y2) = pickle.load(open(file, "rb"))

    b.set_epd(epd)
    if b.turn != chess.WHITE:
        b.apply_mirror()

    ym = modeleval.predict(np.array([extract_features(b)]), batch_size=1)

    val = ym[0][0][0] * Const.INFINITECP
    if val < -Const.INFINITECP:
        ycoords.append(-Const.INFINITECP)
    elif val > Const.INFINITECP:
        ycoords.append(Const.INFINITECP)
    else:
        ycoords.append(val)

    if Y1[0] < -Const.INFINITECP:
        xcoords.append(-Const.INFINITECP)
    elif Y1[0] > Const.INFINITECP:
        xcoords.append(Const.INFINITECP)
    else:
        xcoords.append(Y1[0])

    print("---")
    print(file+":  "+epd)
    print("Evaluation   SF: %-5.2f   NN: %-5.2f"%(Y1[0],val))

    l = []
    for m in b.generate_legal_moves():
        a = str(m).upper()
        l.append((m, Y2[ord(a[0])-65, ord(a[1])-49, ord(a[2])-65, ord(a[3])-49]))
    print("sorted moves SF: ", end="")
    for m in sorted(l, key=lambda e: e[1], reverse=True):
        print(str(m[0]), " ", end="")
    print("")

    l = []
    for m in b.generate_legal_moves():
        a = str(m).upper()
        l.append((m, ym[1][0][ord(a[0])-65, ord(a[1])-49, ord(a[2])-65, ord(a[3])-49]))
    print("sorted moves NN: ", end="")
    for m in sorted(l, key=lambda e: e[1], reverse=True):
        print(str(m[0]), " ", end="")
    print("")

elapsed=time.time()-starttime
print("---")
print("---")
print(str(numsamples)+" samples in "+str(elapsed)+" seconds = "+str(numsamples/elapsed)+" nodes/sec")
    
plt.autoscale = False
plt.axis([-Const.INFINITECP-1, Const.INFINITECP+1, -Const.INFINITECP-1, Const.INFINITECP+1])
plt.plot(xcoords,ycoords,"ro")
plt.grid(True)
plt.axes().set_aspect('equal')
plt.ylabel('ann model eval')
plt.xlabel('std engine eval')
plt.title('"Confusion plot" of model evaluations '+lastmodel)
print("Used model "+lastmodel)

plt.show()