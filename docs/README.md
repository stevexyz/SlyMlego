
# Sly Mlego: A Deep Learning Chess Platform

## Context

Sly Mlego has been developed to familiarize hands-on with machine learning concepts and libraries.

While deep learning has already proved in a lot of fields as the best way to accomplish “intelligent” tasks, still the best available chess engines are based on imperative approaches, with hand made heuristics developed and improved over a long time (note: this was true when I originally wrote it, while for few days seems that there is a new kid on the block, namely Alpha Zero, that seems to have surpassed the current best chess programs...).


## Approach

Since premature optimization is the root of all evil in this initial phase the goal is to find what features and models can provide the best results, leaving optimized implementations to later phases. For this reason I have chosen Python and Keras as the current best options (even if I graduated in a math faculty and know very well C and assembler :))

In this moment the focus is to identify "value network" (features and models), so final output should be a “simple” real number identifying winning probability for white (-inf/+inf measured in centipawns as a standard chess engine). "Policy networks" to directly identify the best moves will be experimented later on (there is a vague idea to use GAN paradigm for them).

Actual training is supervised, based on expert chess engines position (states in standard operation research terminology) labelling - e.g. Stockfish, Gull, etc. - or also expert human input (e.g. labelled Strategic Test Suite) for policy network. Later on reinforcement learning can be a nice way to improve strenght.

Note that the chess engine used to label positions is exploring a full tree of them to calculate the position value, while this network is calculating the value just looking at the current position! Naturally, when inserted in an engine also this network will be used to explore a tree, but if it will work properly it will explore far much less positions to have the same strength results.


## Tasks

Tasks completed:
- Creation of the framework to prepare input data (extract and save features together with labels)
- Implementation of the extraction of a number of features (30 total up to now)
- Implementation of the training process for the models with fit "generator" and checkpoints model saving
- Creation of the value function for the engine implementation
- Creation of some help scripts to be used in the training process

Tasks to be done:
- Preparation: extract even more features (details down) and use more engines for labelling
- Training: experiment with different models / verify long training sessions results
- Testing: Insert evaluation model inside a chess engine (e.g. Sunfish)

Nice things that can be done in later phases:
- Implementation of "policy network"
- Improve training with reinforcement learning
- Autodiscovery of optimal model and hyperparameters (and maybe optimal input features)
- Input features minimization / model optimization
- Chess engine optimized implementation

PS: I'm not (still?) a "pythonist": this is my second project on it and the first was some 15 years ago: so if you are an expert don't be afraid to suggest more pythonic ways to manage things!


## Audience

Everyone knowing chess rules and a bit of python should be able to easily add new input features and create or modify models, and if you are a Deep Learning expert hope you'll find this framework very comfortable to experiment with! And please, share experiments and results! :)


## Prerequisites

1. Install all dependencies: e.g. python-chess, tensorflow, keras, numpy, matplotlib, stockfish (also interpreter for .sh scripts in case of non Unix platforms, still not tested anyway)
2. Obtain/create more fen/epd positions for training (e.g. download the 5 million position giraffe set)


## HOW-TO

**Train the model**:
1. Get fen/epd positions for training
2. Prepare training input -> e.g. `PrepareInput.py file.epd 1 100000`
3. Put some of the position prepared from `__inputstobeprocessed` into `__validationdata` -> e.g. `mv-tovalidation.py 1000`
4. Train model -> e.g. `TrainModel.py` (you can run tensorboard to check results)
5. Play with the engine -> Eval function is ready: you can try it with `print-confusiongraph.py` or let me know if you'll insert it in an engine! :)

**Alternatively train the model without keeping files**:
1. Get fen/epd positions for training
2. Prepare validation data -> e.g. `PrepareInput.py file.epd 1 1000 && mv-tovalidation.py 1000`
3. Train model -> e.g. `TrainModel.py file.epd 1001`

**Reset model / learning**:
1. Remove saved model file -> `clean-model.sh`
2. Put back all training files in inputfiletoprocess directory -> `mv-backtoprocess.py`

**Add training samples** (and train):
1. Simply prepare input files from new epd(s) -> `PrepareInput.py newfile.epd`
2. Continue training model -> `TrainModel.py`

**Backup the trained model**:
1. Move or copy all `__model.*` files in a new directory

**Restore the trained model** (and train more):
1. Copy all the previously saved `__model.*` back to main directory
Note: features should be the same of the model saved else input layer will not match

**Add/modify features**:
1. Remove all training files from all directories -> `clean-model-and-data.sh`
2. Modify `FeatureExtraction.py` program and add/modify features (in `Const.py` there is the list of features)
3. To see features extracted use `print_x.py` (note that is adapting dimension order and coordinates for pretty printing)
4. Prepare and train again

**Evaluate a single chess position**:
1. Copy the trained model to be used (`__model.hdf5`) in the current directory
2. Call to `EvaluatePosition(board)` function (in `Eval.py`) with the fen string of the position to be evaluated return the evaluation score in centipawns


## TROUBLESHOOTING

The app seems "waiting" at the beginning or during epoch execution</br>
    -> Do you have samples in the inputs to be processed directory? If not you can either `PrepareInputs.py` or run `TrainModel.py` with optional epd input file as parameter

The app seems "waiting" at the end of the epoch</br>
    -> Do you have samples in the validation directory?


## Features and Models

Current features implemented:
- Pieces position 1-hot encoding: (6piecesx2colors)x8x8 with 1/0 values
- Pieces mobility 1-hot encoding "added" for multiple pieces moveable on same square: (6piecesx2colors)x8x8 with n/0 values
- Note: for now things like castling and en-passant are embedded in piece mobility and not considered separately 
- Direct square control ("added" if more pieces controlling): 8x8x2colors with n/0 values
- King crown separated: 8x8 x2colors with 1/0 values

Example features still possible to implement to substitute or extend current ones:
- Piece position "joined" (8x8x6pieces with 1/0/-1 values)
- Direct square control "joined" (8x8x1 with +n/0/-n values)
- King mobility / crown "joined" (8x8 with 1/0/-1 values) – castling embedded
- Double move pieces mobility (6piecesx2colors)x8x8 = n/0
- Double move square control (8x8x1 with +n/0/-n values)
- "Protection" trees
- Additional board features (e.g. realsidetomove + 40movescount + enpassantsquare + castlingflags)


## Files

Python file | Description
---- | ----
Const.py | Common constant values used in other files
Eval.py | Contains evaluation function (to be inserted in chess engine)
FeaturesExtraction.py | Extract features vector "X" from a chess position
PrepareInput.py | Script to extract and save features from .fen/.epd files in order to be used in training model <br/> Usage: `PrepareInput.py fenOrEpdFile [startingPosition [numberOfPositionToProcess]]`
TrainModel.py | Model creation and training <br/> Usage: `TrainModel.py [fenOrEpdFile [startingPosition]]` <br/> With no parameters uses the extracted features present in the "to be processed" directory (done with PrepareInput.py) and move them in the "already processed" directory while using them. If epd file is specified the features are extracted just temporarily and deleted after being used. 

Auxiliary shell script | Description
---- | ----
clean-model.sh | Clean all model files in order to start from scratch with a new model
clean-model-and-data.sh | Clean all model files but also extracts data (to be used when features extraction changed)
monitor-log.sh | When training process in ongoing shows main advancements
monitor-tensorboard.sh | Launch tensorboard for training process analysis (and/or TF model exploration)
mv-backtoprocess.py | Move the "already processed" files back in the "to be processed" directory (a script was required since command line is not able to manage very large number of files as it can happen to have)
mv-tovalidation.py | Move the files in the "to be processed" directory to the directory used to validate model
print-confusiongraph.py | Print a "confusion graph" showing actual evaluations of validation samples compared to the ones computed by engine (optional input different model file than the one under training)
print-x.py | Print the extracted feature vectors of the already processed position given in input (pickle file)


## Actual tests

Up to now I've just tested with CPU, running just some models one night each. From the chart seems the system is learning, but with a very slow rate for now. Hope to find better models or new features to improve the situation. As said if someone would try it with a better hw or more time or different models please share the experience (e.g. with a pull request or opening an issue)!

<img src="https://github.com/stevexyz/SlyMlego/blob/master/docs/ex01.png" alt="TF scalars" style="width: 600px;"/>

<img src="https://github.com/stevexyz/SlyMlego/blob/master/docs/ex02.png" alt="Confusion graph" style="width: 400px;"/>


## Acknowledgements

Thanks to:
- Francois Hollet and Niklas Fellas for making simple abstraction libraries
- Mihai Dobre and Massimo Natale for their initial counseling


## LICENSE

Sly Mlego is licensed under the Affero GPL v3 (or any later version at your option). Check out LICENSE file for the full text.

