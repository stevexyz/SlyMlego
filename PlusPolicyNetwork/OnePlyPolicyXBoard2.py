#!/usr/bin/python3

import Const
import chess.uci
import sys
from FeaturesExtraction import extract_features
import numpy as np
from keras.models import load_model

modeleval=None
board=None

if len(sys.argv) > 2:
    print('Xboard engine string: "python3 ', sys.argv[0], '[<modelfile>]"')
    exit(1)

forcemove = False
while True:

    try: line = sys.stdin.readline()
    except KeyboardInterrupt: pass # avoid control-c breaks
    line = line.strip(' \t\n'+chr(3))
    parts = line.split(' ')

    if parts[0]=='xboard':
        print('tellics say 1-ply experimental chess engine "policy" version')
        print('tellics say based on SlyMlego deep learning platform')
        print('tellics say by Stefano Marago\' 2018')
        print('tellics say https://github.com/stevexyz/SlyMlego')

    elif parts[0]=='protover' and parts[1]=='2':
        print('feature done=0')
        sys.stdout.flush() # ensure xboard wait to activate network
        print('feature debug=1')
        if len(sys.argv)<=1:
            modeleval = load_model(Const.MODELFILE+".hdf5")
        else:
            modeleval = load_model(sys.argv[1])
        board=chess.Board()
        modeleval.predict(np.array([extract_features(board)]), batch_size=1, verbose=0) # just to startup engine
        print('feature myname="1ply-policy-v0.1"')
        print('feature variants="normal"')
        print('feature setboard=1')
        print('feature ping=1')
        print('feature usermove=1')
        print('feature analyze=0')
        print('feature pause=0')
        print('feature nps=0')
        print('feature memory=0')
        print('feature sigint=0')
        print('feature done=1')

    elif parts[0]=='quit':
        break

    elif parts[0]=='new':
        board = chess.Board()

    elif parts[0]=='setboard':
        board.set_epd(parts[1])

    elif parts[0]=='ping':
        print('pong '+parts[1])

    elif parts[0]=='undo':
        board.pop()

    elif parts[0]=='force':
        forcemove = True

    elif parts[0]=='usermove':
        board.push_uci(parts[1])

    if parts[0]=='go' or (parts[0]=='usermove' and not forcemove): # usermove already processed
        forcemove = False
        bestmove = None
        bestval = -float('inf')
        if board.turn == chess.WHITE:
            b = board
        else:
            b = board.mirror()
        y = modeleval.predict(np.array([extract_features(b)]), batch_size=1, verbose=0)
        for m in board.legal_moves: # single PLY evaluation!
            a = str(m).upper()
            val = y[1][0][ord(a[0])-65, ord(a[1])-49, ord(a[2])-65, ord(a[3])-49]
            print("# currmove "+str(m)+" score "+str(val))
            if val>bestval:
                bestmove = m
                bestval = val
        print('move %s' % bestmove)
        board.push(bestmove)

    sys.stdout.flush()

exit(0)
