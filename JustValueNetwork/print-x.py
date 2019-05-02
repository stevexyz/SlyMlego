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

(epd, X, Y) = pickle.load(open(picklefile, "rb"))

print(epd+": "+str(Y[0]))

for f in range(Const.NUMFEATURES):
    print("")
    print("- "+str(f)+" ---------------")
    for i in range(8):
        for j in range(8):
            print(str(X[j,7-i,f]), end=' ')
        print("")


