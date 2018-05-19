#!/bin/bash
 xboard -fcp "python3 ./MlegoMctsXboard2.py __model.hdf5 50" -scp "fairymax" -tc 0:5 -inc 3 -mg 1 -debug -sgf __tests.pgn
