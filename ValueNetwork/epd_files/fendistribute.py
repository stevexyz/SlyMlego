#!/usr/bin/python3

import sys
import chess

import logging, sys
#logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
#logging.debug('A debug message!')
#logging.info('We processed %d records', len(processed_records))

# MAIN

if len(sys.argv) < 2:
    print("Usage:", 
          sys.argv[0], 
          "<filename_with_games_fen_positions_each_game_starting_with_underscore>")
    exit(1)

filename = sys.argv[1]
fileopenings = open(filename+".openings.epd","w")
filemiddlegames = open(filename+".middlegames.epd","w")
fileendgames = open(filename+".endgames.epd","w")

board = chess.Board()
lines = [line.rstrip('\n') for line in open(sys.argv[1])]
for line in lines:
    if line=="_":
        currentmove = 0
    else:
        currentmove = currentmove+1
        epdposition = " ".join(line.split()[0:4])
        if currentmove<10:
            fileopenings.write(epdposition+"\n")
        else:
            board.set_epd(epdposition)
            if chess.popcount(board.occupied)<10:
                fileendgames.write(epdposition+"\n")
            else:
                filemiddlegames.write(epdposition+"\n")

exit(0)
