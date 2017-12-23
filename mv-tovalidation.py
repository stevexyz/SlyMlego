#!/usr/bin/python3

import Const
import glob, shutil
import os
import sys
import subprocess

if not os.path.exists(Const.VALIDATIONDATADIR)
    os.makedir(Const.VALIDATIONDATADIR)

if len(sys.argv) >= 2:
    totalfiles = int(sys.argv[1])
else:
    totalfiles = 999999999

for file in glob.glob(Const.ALREADYPROCESSEDDIR+"/*.pickle"):
    shutil.move(file, Const.VALIDATIONDATADIR)
    totalfiles -= 1
    if totalfiles==0: exit(0)

