#!/usr/bin/python3

import Const
from Eval import Eval
import chess.uci
import sys

eval = Eval()
board = chess.Board()

print('tellics say  "1 ply" experimental chess engine')
print('tellics say  by Stefano Maragò - 2018')

forcemove = False
while True:

    line = sys.stdin.readline()
    if line == None:
        break
    line = line.rstrip('\n')
    if len(line) == 0:
        continue
    parts = line.split(' ')

    if parts[0] == 'new':
        board = chess.Board()

    elif parts[0] == 'force':
        forcemove = True
        #try: board.push_uci(parts[1])
        #except: pass

    elif parts[0] == 'undo':
        board.pop()

    elif parts[0] == 'go':
        forcemove = False
        bestmove = None
        bestval = Const.INFINITECP
        for m in board.legal_moves: # single PLY evaluation!
            board.push(m)
            if board.is_checkmate():
                val = -999999 # ensure to follow up
            else:
                val = eval.EvaluatePositionB(board)[0]
            print("# currmove "+str(m)+" score "+str(-val))
            if val<bestval: # minimum value for adversary
                bestmove = m
                bestval = val
            board.pop()
        print('move %s' % bestmove)
        board.push(bestmove)

    elif forcemove:
        if(len(parts[0])<=5):
            try: board.push_uci(parts[0])
            except: pass

    sys.stdout.flush()

exit(0)
