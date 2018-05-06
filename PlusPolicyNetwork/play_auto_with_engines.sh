#!/bin/bash

xboard -fcp "python3 OnePlyEngineXBoard2.py" -scp fairymax -tc 0:05 -inc 1 -mg 1 -debug -sgf __tests.pgn
mv xboard.debug xboard1.debug
xboard -scp "python3 OnePlyEngineXBoard2.py" -fcp fairymax -tc 0:05 -inc 1 -mg 1 -debug -sgf __tests.pgn
mv xboard.debug xboard2.debug

xboard -fcp "python3 OnePlyEngineXBoard2.py" -scp "gnuchess -x" -tc 0:05 -inc 1 -mg 1 -debug -sgf __tests.pgn
mv xboard.debug xboard1.debug
xboard -scp "python3 OnePlyEngineXBoard2.py" -fcp "gnuchess -x" -tc 0:05 -inc 1 -mg 1 -debug -sgf __tests.pgn
mv xboard.debug xboard2.debug

xboard -fcp "python3 OnePlyEngineXBoard2.py" -scp "python /usr/lib/python2.7/dist-packages/pychess/Players/PyChess.py" -tc 0:05 -inc 1 -mg 1 -debug -sgf __tests.pgn
mv xboard.debug xboard1.debug
xboard -scp "python3 OnePlyEngineXBoard2.py" -fcp "python /usr/lib/python2.7/dist-packages/pychess/Players/PyChess.py" -tc 0:05 -inc 1 -mg 1 -debug -sgf __tests.pgn
mv xboard.debug xboard2.debug

xboard -fcp "python3 OnePlyEngineXBoard2.py" -scp shamax -tc 0:05 -inc 1 -mg 1 -debug -sgf __tests.pgn
mv xboard.debug xboard1.debug
xboard -scp "python3 OnePlyEngineXBoard2.py" -fcp shamax -tc 0:05 -inc 1 -mg 1 -debug -sgf __tests.pgn
mv xboard.debug xboard2.debug

xboard -fcp "python3 OnePlyEngineXBoard2.py" -scp stockfish -sUCI -tc 0:05 -inc 1 -mg 1 -debug -sgf __tests.pgn
mv xboard.debug xboard1.debug
xboard -scp "python3 OnePlyEngineXBoard2.py" -fcp stockfish -fUCI -tc 0:05 -inc 1 -mg 1 -debug -sgf __tests.pgn
mv xboard.debug xboard2.debug

