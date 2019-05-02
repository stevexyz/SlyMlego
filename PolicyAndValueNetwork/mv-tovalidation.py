#!/usr/bin/python3

import Const
import glob, shutil
import os
import sys
import subprocess

if not os.path.exists(Const.VALIDATIONDATADIR):
    os.makedirs(Const.VALIDATIONDATADIR)

if len(sys.argv) >= 2:
    print(sys.argv[1])
    totalfiles = int(sys.argv[1])
    print(totalfiles)
else:
    totalfiles = 999999999

print("Start moving "+Const.TOBEPROCESSEDDIR+"/*.pickle")

for file in glob.glob(Const.TOBEPROCESSEDDIR+"/*.pickle"):
    print("Moving "+file)
    shutil.move(file, Const.VALIDATIONDATADIR)
    totalfiles -= 1
    if totalfiles==0: exit(0)

