#!/usr/bin/python3

import Const
import glob, shutil
import os
import sys
import subprocess

for file in glob.glob(argv[1]):
    print("Copying "+str(file)+" to "+argv[2])
    shutil.copy(file, argv[2])

