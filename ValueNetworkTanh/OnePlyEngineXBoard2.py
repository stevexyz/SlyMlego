#!/usr/bin/python3

import Const
from Eval import Eval
import chess.uci
import sys

forcemove = False
while True:

    line = sys.stdin.readline()
    if line == None: break
    line = line.strip(' \t\n')
    parts = line.split(' ')

    if parts[0]=='xboard':
        print('tellics say "1 ply" experimental chess engine')
        print('tellics say based on SlyMlego deep learning platform')
        print('tellics say by Stefano Maragò 2018')
        print('tellics say https://github.com/stevexyz/SlyMlego')
        board = chess.Board()

    elif parts[0]=='quit':
        break

    elif parts[0]=='protover':
        print('feature done=0')
        print('feature debug=1')
        eval = Eval()
        eval.EvaluatePositionB(board)[0] # just to startup engine
        print('feature myname="oneply"')
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
        bestval = Const.INFINITECP
        for m in board.legal_moves: # single PLY evaluation!
            board.push(m)
            if board.is_checkmate():
                val = -999999
            else:
                val = eval.EvaluatePositionB(board)[0] * ( -1 if board.turn==chess.BLACK else 1 )
            print("# currmove "+str(m)+" score "+str(-val))
            if val<bestval: # look for minimum value for adversary
                bestmove = m
                bestval = val
            board.pop()
        print('move %s' % bestmove)
        board.push(bestmove)

    sys.stdout.flush()

exit(0)
