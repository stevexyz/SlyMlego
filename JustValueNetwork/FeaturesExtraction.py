#!/usr/bin/python3

import Const

from keras import backend as K

import sys
import math
import numpy as np
import chess.uci
import pickle

import logging, sys

import os


def fen_invert_position(fenstring):
    v = fenstring.swapcase().split()
    return "/".join((v[0].split("/"))[::-1])+" " \
           + ("b" if v[1].upper()=="W" else "w")+" " \
           + ("-" if v[2]=="-" \
                  else ''.join(c for c in v[2] if c in "QK")+ \
                       ''.join(c for c in v[2] if c in "qk"))+" " \
           + ("-" if v[3]=="-" \
                  else v[3][0].lower()+str(9-int(v[3][1])))


def bitCount(int_type): # TODO: optimize with chess.popcount
    count = 0
    while (int_type):
        int_type &= int_type - 1
        count += 1
    return (count)


def addMoves(X, board, i, j, f, blackpiece=False):
    if blackpiece: board.turn = chess.BLACK
    for m in board.generate_legal_moves():
        if chess.square(i, j) == m.from_square:
            X[m.to_square % 8, int(m.to_square / 8), f] += 1
    if blackpiece: board.turn = chess.WHITE


def addCrown(X, board, i, j, f):
    for x in (-1, 0, +1):
        for y in (-1, 0, +1):
            if 0 <= (i + x) <= 7 and 0 <= (j + y) <= 7:
                X[i + x, j + y, f] = 1


def extract_features(board):

    if board.turn != chess.WHITE:
        raise ValueError("Features should be extracted from " +
                         "a board with white perspective")

    if K.image_dim_ordering() == "th":
        X = np.zeros((Const.NUMFEATURES, 8, 8))  # Channel first!
    elif K.image_dim_ordering() == "tf":
        X = np.zeros((8, 8, Const.NUMFEATURES))  # Channel last!
    else:
        raise ValueError("Dim ordering not understood (nor tf nor th)!")

    # 8x8 N times for now avoid to add non square related features
    # e.g.: also castling right for the AI will be seen as a possible
    #       king move exactly on the square it is happening
    for i in range(8):
        for j in range(8):

            piece = board.piece_at(chess.square(i, j))

            if K.image_dim_ordering() == "tf":

                # position and moves
                if piece == chess.Piece(chess.PAWN, chess.WHITE):
                    X[i, j, Const.X_white_pawns] = 1
                    addMoves(X, board, i, j, Const.X_white_pawns_moves)
                elif piece == chess.Piece(chess.KNIGHT, chess.WHITE):
                    X[i, j, Const.X_white_knights] = 1
                    addMoves(X, board, i, j, Const.X_white_knights_moves)
                elif piece == chess.Piece(chess.BISHOP, chess.WHITE):
                    X[i, j, Const.X_white_bishops] = 1
                    addMoves(X, board, i, j, Const.X_white_bishops_moves)
                elif piece == chess.Piece(chess.ROOK, chess.WHITE):
                    X[i, j, Const.X_white_rooks] = 1
                    addMoves(X, board, i, j, Const.X_white_rooks_moves)
                elif piece == chess.Piece(chess.QUEEN, chess.WHITE):
                    X[i, j, Const.X_white_queens] = 1
                    addMoves(X, board, i, j, Const.X_white_queens_moves)
                elif piece == chess.Piece(chess.KING, chess.WHITE):
                    X[i, j, Const.X_white_king] = 1
                    addMoves(X, board, i, j, Const.X_white_king_moves)
                    addCrown(X, board, i, j, Const.X_white_king_crown)
                elif piece == chess.Piece(chess.PAWN, chess.BLACK):
                    X[i, j, Const.X_black_pawns] = 1
                    addMoves(X, board, i, j, Const.X_black_pawns_moves, True)
                elif piece == chess.Piece(chess.KNIGHT, chess.BLACK):
                    X[i, j, Const.X_black_knights] = 1
                    addMoves(X, board, i, j, Const.X_black_knights_moves, True)
                elif piece == chess.Piece(chess.BISHOP, chess.BLACK):
                    X[i, j, Const.X_black_bishops] = 1
                    addMoves(X, board, i, j, Const.X_black_bishops_moves, True)
                elif piece == chess.Piece(chess.ROOK, chess.BLACK):
                    X[i, j, Const.X_black_rooks] = 1
                    addMoves(X, board, i, j, Const.X_black_rooks_moves, True)
                elif piece == chess.Piece(chess.QUEEN, chess.BLACK):
                    X[i, j, Const.X_black_queens] = 1
                    addMoves(X, board, i, j, Const.X_black_queens_moves, True)
                elif piece == chess.Piece(chess.KING, chess.BLACK):
                    X[i, j, Const.X_black_king] = 1
                    addMoves(X, board, i, j, Const.X_black_king_moves, True)
                    addCrown(X, board, i, j, Const.X_black_king_crown)

                # attackers
                X[i, j, Const.X_white_attackers] =\
                    bitCount(int(board.attackers(chess.WHITE, chess.square(i, j))))
                X[i, j, Const.X_black_attackers] =\
                    bitCount(int(board.attackers(chess.BLACK, chess.square(i, j))))

                # pins
                if board.is_pinned(chess.WHITE, chess.square(i, j)):
                    X[i, j, Const.X_white_is_pinned] = 1
                if board.is_pinned(chess.BLACK, chess.square(i, j)):
                    X[i, j, Const.X_black_is_pinned] = 1

            else:  #th
                raise ValueError("Theano dimension ordering to be updated")
                # STOP!
                exit(1)
                if piece == chess.Piece(chess.PAWN, chess.WHITE):
                    X[Const.X_white_pawns, i, j] = 1
                elif piece == chess.Piece(chess.KNIGHT, chess.WHITE):
                    X[Const.X_white_knights, i, j] = 1
                elif piece == chess.Piece(chess.BISHOP, chess.WHITE):
                    X[Const.X_white_bishops, i, j] = 1
                elif piece == chess.Piece(chess.ROOK, chess.WHITE):
                    X[Const.X_white_rooks, i, j] = 1
                elif piece == chess.Piece(chess.QUEEN, chess.WHITE):
                    X[Const.X_white_queens, i, j] = 1
                elif piece == chess.Piece(chess.KING, chess.WHITE):
                    X[Const.X_white_king, i, j] = 1
                elif piece == chess.Piece(chess.PAWN, chess.BLACK):
                    X[Const.X_black_pawns, i, j] = 1
                elif piece == chess.Piece(chess.KNIGHT, chess.BLACK):
                    X[Const.X_black_knights, i, j] = 1
                elif piece == chess.Piece(chess.BISHOP, chess.BLACK):
                    X[Const.X_black_bishops, i, j] = 1
                elif piece == chess.Piece(chess.ROOK, chess.BLACK):
                    X[Const.X_black_rooks, i, j] = 1
                elif piece == chess.Piece(chess.QUEEN, chess.BLACK):
                    X[Const.X_black_queens, i, j] = 1
                elif piece == chess.Piece(chess.KING, chess.BLACK):
                    X[Const.X_black_king, i, j] = 1
                #else:

    return X
