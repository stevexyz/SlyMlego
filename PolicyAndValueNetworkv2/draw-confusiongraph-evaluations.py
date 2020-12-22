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

if sys.argv[0]=="./draw-confusiongraph-evaluations.py":
    graphtype = 1
    numsamples = 200 # default
    print("Drawing POSITION EVALUATIONS")
elif sys.argv[0]=="./draw-confusiongraph-moves.py":
    graphtype = 2
    numsamples = 20 # default
    print("Drawing MOVE POLICY FITTING")
elif sys.argv[0]=="./draw-confusiongraph-policy.py":
    graphtype = 3
    numsamples = 20 # default
    print("Drawing MOVE POLICY VALUES")
else:
    raise ValueError("Program name "+sys.argv[0]+" not recognized!")

if len(sys.argv)>=2:
    numsamples = int(sys.argv[1])

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
    
xcoords = []
ycoords = []
starttime = time.time()
b = chess.Board()
totsamples = float(numsamples)
for file in glob.glob(Const.VALIDATIONDATADIR+"/*.pickle"):
    numsamples -= 1
    if numsamples<=0: break
    (epd, X, Y1, Y2) = pickle.load(open(file, "rb"))

    b.set_epd(epd)
    if b.turn != chess.WHITE:
        b.apply_mirror()

    ym = modeleval.predict(np.array([extract_features(b)]), batch_size=1)

    sfval = Y1[0]
    if sfval < -Const.INFINITECP:
        sfval = -Const.INFINITECP
    elif sfval > Const.INFINITECP:
        sfval = Const.INFINITECP
    nnval = ym[0][0][0] * Const.INFINITECP
    if nnval < -Const.INFINITECP:
        nnval = -Const.INFINITECP
    elif nnval > Const.INFINITECP:
        nnval = Const.INFINITECP

    if graphtype==1:
        xcoords.append(sfval)
        ycoords.append(nnval)

    print("---")
    print(file+":  "+epd)
    print("Evaluation   SF: %-5.2f   NN: %-5.2f"%(sfval,nnval))

    lsf = []
    lnn = []
    for m in b.generate_legal_moves():
        a = str(m).upper()
        lsf.append((m, Y2[ord(a[0])-65, ord(a[1])-49, ord(a[2])-65, ord(a[3])-49]))
        lnn.append((m, ym[1][0][ord(a[0])-65, ord(a[1])-49, ord(a[2])-65, ord(a[3])-49]))
    lsf = sorted(lsf, key=lambda e: e[1], reverse=True)
    lnn = sorted(lnn, key=lambda e: e[1], reverse=True)
    print("sorted moves SF: ", end="")
    for m in lsf:
        print(str(m[0]), " ", end="")
    print("")
    print("sorted moves NN: ", end="")
    for m in lnn:
        print(str(m[0]), " ", end="")
    print("")

    if graphtype==2:
        for i in range(len(lsf)):
            xcoords.append(float(i)+(numsamples/(totsamples+1)))
            for j in range(len(lnn)): # slowish search buk ok
                if lsf[i][0]==lnn[j][0]:
                    ycoords.append(j+(numsamples/(totsamples+1)))
    if graphtype==3:
        for i in range(len(lsf)):
            xcoords.append(lsf[i][1])
            for j in range(len(lnn)): # slowish search buk ok
                if lsf[i][0]==lnn[j][0]:
                    ycoords.append(lnn[i][1])

elapsed=time.time()-starttime
print("---")
print("---")
print(str(numsamples)+" samples in "+str(elapsed)+" seconds = "+str(numsamples/elapsed)+" nodes/sec")
    
plt.autoscale = False
if graphtype==1:
    plt.axis([-Const.INFINITECP-1, Const.INFINITECP+1, -Const.INFINITECP-1, Const.INFINITECP+1])
    plt.xlabel('std engine eval')
    plt.ylabel('ann model eval')
elif graphtype==2:
    plt.axis([-1, 50, -1, 50])
    plt.xlabel('std engine move index')
    plt.ylabel('ann model move index')
elif graphtype==3:
    plt.axis([-1, 2, -1, 2])
    plt.xlabel('std engine policy')
    plt.ylabel('ann model policy')
else:
    raise ValueError("Graphtype not recognized!")
plt.scatter(xcoords,ycoords, color='red')
plt.grid(True)
plt.axes().set_aspect('equal')

plt.title('"Confusion plot" of model evaluations '+lastmodel)
print("Used model "+lastmodel)

plt.show()