#!/usr/bin/python3

import Const
import glob, shutil
import os
import sys
import subprocess

for file in glob.glob(Const.ALREADYPROCESSEDDIR+"/*.pickle"):
    shutil.move(file, Const.TOBEPROCESSEDDIR)

