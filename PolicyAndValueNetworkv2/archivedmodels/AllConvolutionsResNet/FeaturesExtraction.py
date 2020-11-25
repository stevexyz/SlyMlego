#!/usr/bin/python3

import Const
from tensorflow.keras import backend as K
import sys
import math
import numpy as np
import chess
import pickle
import logging, sys
import os

# Input matrixes IDs (all 8x8)
#@featuresbegin

NUMFEATURES = 30
#
# one-hot encoding of position of the pieces
X_white_pawns = 0
X_white_rooks = 1
X_white_knights = 2
X_white_bishops = 3
X_white_queens = 4
X_white_king = 5
X_black_pawns = 6
X_black_rooks = 7
X_black_knights = 8
X_black_bishops = 9
X_black_queens = 10
X_black_king = 11
#
# number of attackers for each side on each square
# (note: pinned pieces considered attacking for now)
X_white_attackers = 12
X_black_attackers = 13
#
# one-hot encoding of pinned pieces
X_white_is_pinned = 14
X_black_is_pinned = 15
#
# squares on which pieces can move
# (note1: just legal moves, so pinned pieces cannot move for now)
# (note2: number add if different pieces of same type can move to same square)
X_white_pawns_moves = 16
X_white_rooks_moves = 17
X_white_knights_moves = 18
X_white_bishops_moves = 19
X_white_queens_moves = 20
X_white_king_moves = 21
X_black_pawns_moves = 22
X_black_rooks_moves = 23
X_black_knights_moves = 24
X_black_bishops_moves = 25
X_black_queens_moves = 26
X_black_king_moves = 27
#
# area around the kings
X_white_king_crown= 28
X_black_king_crown= 29

#@featuresend

def addCrown(X, board, i, j, f):
    for x in (-1, 0, +1):
        for y in (-1, 0, +1):
            if 0 <= (i + x) <= 7 and 0 <= (j + y) <= 7:
                X[i + x, j + y, f] = 1

def bitCount(int_type): # TODO: optimize with chess.popcount
    count = 0
    while (int_type):
        int_type &= int_type - 1
        count += 1
    return (count)

def extract_features(board):

    if board.turn != chess.WHITE:
        raise ValueError("Features should be extracted from a board with white perspective")

    if K.image_data_format() == "channels_last": # K.image_dim_ordering()=="tf"
        X = np.zeros((8, 8, NUMFEATURES))  # Channel last!
    elif K.image_data_format() == "channels_firs": # K.image_dim_ordering()=="th"
        raise ValueError("Theano dimension ordering not implemented") # np.zeros((NUMFEATURES, 8, 8))
        exit(1)

    # piece positions
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(i, j))
            # position and moves
            if piece == chess.Piece(chess.PAWN, chess.WHITE):
                X[i, j, X_white_pawns] = 1
            elif piece == chess.Piece(chess.KNIGHT, chess.WHITE):
                X[i, j, X_white_knights] = 1
            elif piece == chess.Piece(chess.BISHOP, chess.WHITE):
                X[i, j, X_white_bishops] = 1
            elif piece == chess.Piece(chess.ROOK, chess.WHITE):
                X[i, j, X_white_rooks] = 1
            elif piece == chess.Piece(chess.QUEEN, chess.WHITE):
                X[i, j, X_white_queens] = 1
            elif piece == chess.Piece(chess.KING, chess.WHITE):
                X[i, j, X_white_king] = 1
                addCrown(X, board, i, j, X_white_king_crown)
            elif piece == chess.Piece(chess.PAWN, chess.BLACK):
                X[i, j, X_black_pawns] = 1
            elif piece == chess.Piece(chess.KNIGHT, chess.BLACK):
                X[i, j, X_black_knights] = 1
            elif piece == chess.Piece(chess.BISHOP, chess.BLACK):
                X[i, j, X_black_bishops] = 1
            elif piece == chess.Piece(chess.ROOK, chess.BLACK):
                X[i, j, X_black_rooks] = 1
            elif piece == chess.Piece(chess.QUEEN, chess.BLACK):
                X[i, j, X_black_queens] = 1
            elif piece == chess.Piece(chess.KING, chess.BLACK):
                X[i, j, X_black_king] = 1
                addCrown(X, board, i, j, X_black_king_crown)
            # attackers
            X[i, j, X_white_attackers] =\
                bitCount(int(board.attackers(chess.WHITE, chess.square(i, j))))
            X[i, j, X_black_attackers] =\
                bitCount(int(board.attackers(chess.BLACK, chess.square(i, j))))
            # pins
            if board.is_pinned(chess.WHITE, chess.square(i, j)):
                X[i, j, X_white_is_pinned] = 1
            if board.is_pinned(chess.BLACK, chess.square(i, j)):
                X[i, j, X_black_is_pinned] = 1

    # white moves
    for m in board.generate_legal_moves():
        piece = board.piece_at(m.from_square)
        if piece == chess.Piece(chess.PAWN, chess.WHITE):
            X[m.to_square % 8, int(m.to_square / 8), X_white_pawns_moves] += 1
        elif piece == chess.Piece(chess.KNIGHT, chess.WHITE):
            X[m.to_square % 8, int(m.to_square / 8), X_white_knights_moves] += 1
        elif piece == chess.Piece(chess.BISHOP, chess.WHITE):
            X[m.to_square % 8, int(m.to_square / 8), X_white_bishops_moves] += 1
        elif piece == chess.Piece(chess.ROOK, chess.WHITE):
            X[m.to_square % 8, int(m.to_square / 8), X_white_rooks_moves] += 1
        elif piece == chess.Piece(chess.QUEEN, chess.WHITE):
            X[m.to_square % 8, int(m.to_square / 8), X_white_queens_moves] += 1
        elif piece == chess.Piece(chess.KING, chess.WHITE):
            X[m.to_square % 8, int(m.to_square / 8), X_white_king_moves] += 1
        else:
            raise ValueError("White move not recognized!")

    # black moves
    board.turn = chess.BLACK # temporarily to generate black moves
    for m in board.generate_legal_moves():
        piece = board.piece_at(m.from_square)
        if piece == chess.Piece(chess.PAWN, chess.BLACK):
            X[m.to_square % 8, int(m.to_square / 8), X_black_pawns_moves] += 1
        elif piece == chess.Piece(chess.KNIGHT, chess.BLACK):
            X[m.to_square % 8, int(m.to_square / 8), X_black_knights_moves] += 1
        elif piece == chess.Piece(chess.BISHOP, chess.BLACK):
            X[m.to_square % 8, int(m.to_square / 8), X_black_bishops_moves] += 1
        elif piece == chess.Piece(chess.ROOK, chess.BLACK):
            X[m.to_square % 8, int(m.to_square / 8), X_black_rooks_moves] += 1
        elif piece == chess.Piece(chess.QUEEN, chess.BLACK):
            X[m.to_square % 8, int(m.to_square / 8), X_black_queens_moves] += 1
        elif piece == chess.Piece(chess.KING, chess.BLACK):
            X[m.to_square % 8, int(m.to_square / 8), X_black_king_moves] += 1
        else:
            raise ValueError("Black move not recognized!")
    board.turn = chess.WHITE

    return X
