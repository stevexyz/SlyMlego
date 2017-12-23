#!/usr/bin/python3

import Const
import matplotlib.pyplot as plt
from Eval import Eval 

import numpy as np
import glob, shutil
import pickle
import os
import sys

#if len(sys.argv)<2:
#    print("Print the confusion graph using validation samples")
#    print("Usage: "+sys.argv[0]+" [<modelfile>]")
#    exit(1)

if len(sys.argv)>=2: 
    eval = Eval(sys.argv[1])
    print("Using model "+sys.argv[1])
else:
    eval = Eval()

xcoords=[]
ycoords=[]
for file in glob.glob(Const.VALIDATIONDATADIR+"/*.pickle"):
    (epd, X, Y) = pickle.load(open(file, "rb"))
    val = eval.EvaluatePosition(epd)[0] # model evaluation
    xcoords.append(Y[0])
    ycoords.append(val)
    print(epd+": "+str(Y[0])+" | "+str(val))
    
#plt.plot([-10000,10000],[0,0])
#plt.plot([0,0],[-10000,10000])

plt.autoscale = False
plt.axis([-15000, 15000, -15000, 15000])

plt.plot(xcoords,ycoords,"ro")

plt.grid(True)
plt.axes().set_aspect('equal')

plt.ylabel('model eval')
plt.xlabel('true eval')
plt.title('"Confusion graph" of model evaluations')

plt.show()

