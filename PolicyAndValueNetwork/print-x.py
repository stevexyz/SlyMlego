#!/usr/bin/python3

import Const

import numpy as np
import glob, shutil
import pickle
import os
import sys

if len(sys.argv)<2:
    print("Print position, evaluation and feature matrixes extracted.")
    print("Usage: "+sys.argv[0]+" <picklefile>")
    exit(1)

picklefile=sys.argv[1] 

if not os.path.isfile(picklefile):
    print("Error: "+picklefile+" is not loadable")
    exit(1)

(epd, X, Y1, Y2) = pickle.load(open(picklefile, "rb"))

print(epd+": "+str(Y1[0]))

for i1 in range(8):
    for i2 in range(8):
        for i3 in range(8):
            for i4 in range(8):
                if Y2[i1,i2,i3,i4]>0:
                    print("move ",i1,i2,i3,i4," = ",Y2[i1,i2,i3,i4])


for f in range(Const.NUMFEATURES):
    print("")
    print("- "+str(f)+" ---------------")
    for i in range(8):
        for j in range(8):
            print(str(X[j,7-i,f]), end=' ')
        print("")


