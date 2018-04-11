#!/usr/bin/python3

#23456789012345678901234567890123456789012345678901234567890123456789012
#import pdb; pdb.set_trace() #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


import Const

#from keras import backend as K

import sys
import math
import numpy as np
import chess.uci
import pickle
import pickledb
import os

import logging, sys
#logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
#logging.debug('A debug message!')
#logging.info('We processed %d records', len(processed_records))

import FeaturesExtraction as fe


# MAIN

if len(sys.argv) < 2:
    print("Usage:", 
          sys.argv[0], 
          "<nomefile.epd> [<initial-line> [<number-of-position-to-process>]]")
    exit(1)

engine1 = chess.uci.popen_engine(Const.ENGINE1)
engine1.setoption({"MultiPV": 50})
info_handler1 = chess.uci.InfoHandler()
engine1.info_handlers.append(info_handler1)
board = chess.Board()

lines = [line.rstrip('\n') for line in open(sys.argv[1])]
if len(sys.argv) >= 3:
    initialline = int(sys.argv[2])
    lines = lines[initialline:]
else:
    initialline = 0

if not os.path.exists(Const.TOBEPROCESSEDDIR):
    os.mkdir(Const.TOBEPROCESSEDDIR)

#dbeval = pickledb.load(Const.TOBEPROCESSEDDIR+'__dbeval.pickledb', True)

for line in range(int(sys.argv[3]) if len(sys.argv)>=4 else len(lines)):

    epdposition = " ".join(lines[line].split()[0:4])
    #print(str(line)+"/"+str(len(lines)))
    #print(epdposition)

    # always switch in order now to move is black and responses
    # evaluations are on white side
    board.set_epd(epdposition)
    if board.turn != chess.BLACK:
        epdposition = fe.fen_invert_position(epdposition)
        board.set_epd(epdposition)
        #print("inverted: "+epdposition)

    #Y[0]=dbeval.get()
    #if Y[0]==None:
    #else
    #    dbeval.set('key', 'value')

    engine1.position(board)
    engine1.go(movetime=Const.MOVETIME)
    print(" ".join(epdposition.split()[0:4])+": "+str(len(info_handler1.info["pv"])))
    #print(board)
    for i in info_handler1.info["pv"]:

        # y output values array
        Y = np.zeros((1))
        if info_handler1.info["score"][i].mate is None:
            # minus because evaluation was black side perspective
            Y[0] = -info_handler1.info["score"][i].cp
        else:
            # adjust return e.g. mate in -2
            Y[0] = -math.copysign(Const.INFINITECP,
                                  info_handler1.info["score"][i].mate)
        #print("i: "+str(i)+"; eval="+str(Y[0]))

        # x input features arrays
        board.push(chess.Move.from_uci(str(info_handler1.info["pv"][i][0])))
        newepd = board.fen()
        X = fe.extract_features(board)
        board.pop()

        pickle.dump(
            (newepd, X, Y),
            open(Const.TOBEPROCESSEDDIR + "/" +
                 (sys.argv[1]).replace("/", "_") + "-" +
                 str(line + initialline) + "-" + str(i) + ".pickle", "wb"))

exit(0)
