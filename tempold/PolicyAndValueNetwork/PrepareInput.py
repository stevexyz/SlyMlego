#!/usr/bin/python3

import Const

import sys
import math
import numpy as np
import chess
import chess.engine
import pickle
import pickledb
import os

from pathlib import Path
#data_folder = "source_data/text_files/")
#file_to_open = data_folder / "raw_data.txt"

import FeaturesExtraction as fe


SOFTMAX_CURVE = 10 # the higher the flatter (adapt move value in centipawn)


# MAIN

if len(sys.argv) < 2:
    print("Usage:",
          sys.argv[0],
          "<nomefile.epd> [<initial-line> [<number-of-position-to-process>]]")
    exit(1)

print("Using engine", Const.ENGINE1)
engine1 = chess.engine.popen_uci(Const.ENGINE1)
engine1.setoption({"MultiPV": 50})
engine1.setoption({"Threads": Const.ENGINETHREADS})  
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

    epdposition = " ".join(lines[line].split()[0:6])

    # always switch in order now to move is black and responses evaluations are on white side
    board.set_epd(epdposition)
    if board.turn != chess.WHITE:
        epdposition = fe.fen_invert_position(epdposition)
        board.set_epd(epdposition)

    engine1.position(board)
    engine1.go(movetime=Const.MOVETIME)

    print(str(initialline+line)+": "+" ".join(epdposition.split()[0:4])+" x"+str(len(info_handler1.info["pv"])))

    yv = []
    ym = []
    for i in range(1, len(info_handler1.info["pv"])+1):
        ym.append( str(info_handler1.info["pv"][i][0]).upper() )
        if info_handler1.info["score"][i].mate is None:
            yv.append( info_handler1.info["score"][i].cp )
        else:
            # adjust return e.g. mate in -2
            yv.append( math.copysign(Const.INFINITECP, info_handler1.info["score"][i].mate) )

    if len(yv)>0:

        X = fe.extract_features(board)

        # Y1 is the position value
        # Y2 is the policy tensor (from 8x8 x to 8x8): softmax of the value of the moves available, 0 the others
        Y1 = np.zeros((1))
        Y2 = np.zeros((8,8,8,8))

        Y1[0] = yv[0]

        ys = (np.exp(yv)/SOFTMAX_CURVE) / np.sum((np.exp(yv)/SOFTMAX_CURVE), axis=0) # softmax of move values
        for i in range(len(ym)):
            if not math.isnan(ys[i]):
                Y2[ ord(ym[i][0])-65, ord(ym[i][1])-49, ord(ym[i][2])-65, ord(ym[i][3])-49 ] = ys[i]

        pickle.dump(
            (epdposition, X, Y1, Y2),
             open(Path(Const.TOBEPROCESSEDDIR + "/" +
                       (Path(sys.argv[1])).name + "-" +
                        str(line + initialline) + "-" + str(i) + ".pickle"),
                  "wb"))

exit(0)
