#!/usr/bin/python3

import Const
import matplotlib.pyplot as plt
from Eval import Eval 

import numpy as np
import glob, shutil
import pickle
import os
import sys

if len(sys.argv)<2:
    print("Plot the confusion graph using validation samples")
    print("Usage: "+sys.argv[0]+" [<numsamples> [<modelfile>]]")
    exit(1)

if len(sys.argv)>=2:
    numsamples = int(sys.argv[1])
else:
    numsamples = 999999999

if len(sys.argv)>=3: 
    eval = Eval(sys.argv[2])
    print("Using model "+sys.argv[2])
else:
    eval = Eval()

xcoords=[]
ycoords=[]
for file in glob.glob(Const.VALIDATIONDATADIR+"/*.pickle"):
    numsamples -= 1
    if numsamples<=0: break
    (epd, X, Y) = pickle.load(open(file, "rb"))
    val = eval.EvaluatePosition(epd)[0] # model evaluation
    ycoords.append(val)
    if Y[0] < -Const.INFINITECP:
        xcoords.append(-Const.INFINITECP)
    elif Y[0] > Const.INFINITECP:
        xcoords.append(Const.INFINITECP)
    else:
        xcoords.append(Y[0])
    print(file+" "+epd+" sf:"+str(Y[0])+" nn:"+str(val))
    
#plt.plot([-10000,10000],[0,0])
#plt.plot([0,0],[-10000,10000])

plt.autoscale = False
plt.axis([-15000, 15000, -15000, 15000])

plt.plot(xcoords,ycoords,"ro")

plt.grid(True)
plt.axes().set_aspect('equal')

plt.ylabel('ann model eval')
plt.xlabel('std engine eval')
plt.title('"Confusion plot" of model evaluations')

plt.show()

