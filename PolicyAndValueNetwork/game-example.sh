#!/bin/bash

xboard -fcp "python3 ./MlegoMctsXboard2.py __model.hdf5 50" -scp "fairymax" -tc 0:5 -inc 3 -mg 1 -debug -sgf __tests.pgn

xboard -scp "python3 ./OnePlyValueXBoard2.py __model-v000009.hdf5" -fcp "python3 ./OnePlyPolicyXBoard2.py __model-v000009.hdf5" -tc 0:5 -inc 3 -mg 1 -debug -sgf __tests.pgn

