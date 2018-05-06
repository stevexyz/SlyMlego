#!/usr/bin/python3

import Const

import sys
import math
import numpy as np
import chess.uci
import pickle
import pickledb
import os

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

for line in range(int(sys.argv[3]) if len(sys.argv)>=4 else len(lines)):

    epdposition = " ".join(lines[line].split()[0:4])

    # always switch in order now to move is black and responses evaluations are on white side
    board.set_epd(epdposition)
    if board.turn != chess.BLACK:
        epdposition = fe.fen_invert_position(epdposition)
        board.set_epd(epdposition)

    engine1.position(board)
    engine1.go(movetime=Const.MOVETIME)

    print(str(initialline+line)+": "+" ".join(epdposition.split()[0:4])+" x"+str(len(info_handler1.info["pv"])))

    yv = []
    ym = []
    for i in info_handler1.info["pv"]:
        ym.add( str(info_handler1.info["pv"][i][0]) )
        ... convert move in algebraic format if not already...
        if info_handler1.info["score"][i].mate is None:
            yv.add( info_handler1.info["score"][i].cp )
        else:
            # adjust return e.g. mate in -2
            yv.add( math.copysign(Const.INFINITECP, info_handler1.info["score"][i].mate) )

    X = fe.extract_features(board)

    # Y1 is the position value
    # Y2 is the policy tensor (from 8x8 x to 8x8): softmax of the value of the moves available, 0 the others
    Y1 = np.zeros((1))
    Y2 = np.zeros((8,8,8,8))

    Y1[0] = yv[0]

    ys = np.exp(yv) / np.sum(np.exp(yv), axis=0) # softmax of move values
    for i in ym:
        Y2[ ym[i].asc-65, ym[i].asc-48, ym[i].asc-65, ym[i].asc-48 ] = ys[i]

    pickle.dump(
        (newepd, X, Y1, Y2),
         open(Const.TOBEPROCESSEDDIR + "/" +
             (sys.argv[1]).replace("/", "_") + "-" +
              str(line + initialline) + "-" + str(i) + ".pickle", "wb"))

exit(0)
