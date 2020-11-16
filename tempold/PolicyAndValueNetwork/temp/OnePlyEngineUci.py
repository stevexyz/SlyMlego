#!/usr/bin/python3

import Const
from Eval import Eval
import chess.uci
import os
import sys

if len(sys.argv) > 2:
    print('Xboard engine string: "python3 ', sys.argv[0], '[<modelfile>]"')
    exit(1)

while True:

    line = sys.stdin.readline()
    if line == None: break
    line = line.strip(' \t\n')
    parts = line.split(' ')

    if line == None: break
    line = line.rstrip('\n')
    #if len(line) == 0: continue
    parts = line.split(' ')

    if parts[0] == 'uci':
        print('id name 1-ply experimental chess engine')
        print('id author Stefano Maragò')
        #print('based on SlyMlego deep learning platform')
        #print('https://github.com/stevexyz/SlyMlego')
        print('uciok')

    elif parts[0] == 'isready':
        board = chess.Board()
        if len(sys.argv)<=1: eval = Eval()
        else: eval = Eval(modelfile=sys.argv[1])
        eval.EvaluatePositionB(board) # just to startup engine
        print('readyok')

    elif parts[0] == 'ucinewgame':
        board = chess.Board()

    elif parts[0] == 'position':
        is_moves = False
        nr = 1
        while nr < len(parts):
            if is_moves:
                board.push_uci(parts[nr])
            elif parts[nr] ==  'fen':
                board = Board(' '.join(parts[nr + 1:]))
                break
            elif parts[nr] == 'startpos':
                board = Board()
            elif parts[nr] == 'moves':
                is_moves = True
            else:
                error('unknown: %s' % parts[nr])
            nr += 1

    elif parts[0] == 'go':
        bestmove = None
        bestval = Const.INFINITECP
        for m in board.legal_moves: # single PLY evaluation!
            board.push(m)
            if board.is_checkmate():
                val = -999999 # ensure to follow up
            else:
                val = eval.EvaluatePositionB(board)[0] * ( -1 if board.turn==chess.BLACK else 1 )
            print("info currmove "+str(m)+" score "+str(-val))
            if val<bestval: # minimum value for adversary
                bestmove = m
                bestval = val
            board.pop()
        print('bestmove %s' % bestmove)
        board.push(bestmove)

    elif parts[0] == 'quit':
        break

    sys.stdout.flush()

exit(0)
