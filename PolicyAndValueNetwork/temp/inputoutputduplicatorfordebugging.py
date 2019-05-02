#!/usr/bin/python3

# This program stays in the middle between GUI and engine and output all the communication
# to a file ".communication.debug.txt"

# Refs:
# https://stackoverflow.com/questions/375427/non-blocking-read-on-a-subprocess-pipe-in-python
# https://stackoverflow.com/questions/36476841/python-how-to-read-stdout-of-subprocess-in-a-nonblocking-way
# https://github.com/mdoege/PyTuroChamp/blob/master/xboard-host.py
# https://repolinux.wordpress.com/2012/10/09/non-blocking-read-from-stdin-in-python/

import os, sys, select, time
from subprocess import Popen, PIPE


if len(sys.argv)<=1:
    print("Please specify an executable file name.")
    exit(1)


executable = sys.argv[1]

# array copy...
proc = Popen([executable], stdin=PIPE, stdout=PIPE, stderr=None)
ppoll = select.poll()
ppoll.register(proc.stdout, select.POLLIN)
file = open("./__"+os.path.basename(executable)+".communication.debug.txt","w")

while True:

    #https://stackoverflow.com/questions/8475290/how-do-i-write-to-a-python-subprocess-stdin
    #p = Popen(['myapp'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    #stdout_data = p.communicate(input='data_to_write')[0]

    while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:

        # process the input to the executable
        line = sys.stdin.readline()
        if line:
            proc.stdin.write(line.encode("utf8"))
            file.write("i: "+str(line))
        else: # an empty line means stdin has been closed
            file.close()
            exit(0)

    else:

        # process the output from the executable
        if ppoll.poll(1): # stdout
              line = proc.stdout.readline()
              print(line)
              file.write("o: "+str(line)+"\n")

    time.sleep(0.1) # no need for tight looping
