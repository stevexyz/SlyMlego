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

import FeaturesExtraction as fe

# MAIN

DEBUG = False
if DEBUG:
    sys.argv.append("../EpdFiles/stsall.epd")
    sys.argv.append("2")
    sys.argv.append("10")

if len(sys.argv) < 2:
    print("Usage:",
          sys.argv[0],
          "<nomefile.epd> [<initial-line> [<number-of-position-to-process>]]")
    exit(1)

print("Using engine", Const.ENGINE1)
engine1 = chess.engine.SimpleEngine.popen_uci(Const.ENGINE1)
engine1.configure({"Threads": Const.ENGINETHREADS})
engine1.configure({"Hash": Const.HASHSIZE})

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

    info = engine1.analyse(board, chess.engine.Limit(time=Const.MOVETIME), multipv=99)

    print(str(initialline+line)+": "+" ".join(epdposition.split()[0:4])+" x"+str(len(info)))

    maxscore = -9999
    ym = [] # moves
    yv = [] # score of the moves
    for pos in info:
        score = pos["score"]
        score = str(score.pov(score.turn))
        if score[0]=="#":
            if score[1]=='-': score = - Const.INFINITECP
            else: score = Const.INFINITECP
        else: score = int(score)
        if score > maxscore: maxscore = score
        ym.append( str(pos["pv"][0]).upper() )
        yv.append( score / 100.0 ) # value in pawns (and not centipawns) prevent softmax to explode

    if len(yv)>0:

        X = fe.extract_features(board)

        # Y1 is the position value
        # Y2 is the policy tensor value (from 8x8 x to 8x8): 4096 position containing softmax of the score value for the legal moves, 0 the others
        Y1 = np.zeros((1))
        Y2 = np.zeros((8,8,8,8))

        Y1[0] = maxscore

        ys = np.exp(yv)
        ys /= np.sum(ys, axis=0) # softmax of move values
        # for i in range(len(ym)): print(ym[i], yv[i], ys[i])

        for i in range(len(ym)):
            if not math.isnan(ys[i]):
                Y2[ ord(ym[i][0])-65, ord(ym[i][1])-49, ord(ym[i][2])-65, ord(ym[i][3])-49 ] = ys[i]
            else:
                raise ValueError("Softmax returned a NaN!")

        pickle.dump(
            (epdposition, X, Y1, Y2),
             open(Path(Const.TOBEPROCESSEDDIR + "/" +
                       (Path(sys.argv[1])).name + "-" +
                        str(line + initialline) + "-" + str(i) + ".pickle"),
                  "wb"))

engine1.quit()

exit(0)
