#!/usr/bin/python3

# references:
# - http://ccg.doc.gold.ac.uk/ccg_old/teaching/ludic_computing/ludic16.pdf
# - https://int8.io/monte-carlo-tree-search-beginners-guide/

# To be verified: "eval" recalculation in backpropagation

balance_constant = 50 # exploration vs exploitation balance
move_timeframe = 20  # number of seconds for evaluation
modeleval = None # model will be loaded in protover call


import Const
from queue import *
import chess.uci
import sys
from math import (sqrt, log)
from FeaturesExtraction import extract_features
import numpy as np
from copy import deepcopy


class MctsNode(object):

    def __init__(self, board, parent=None):
        #h#global evaltable, hitsevaltable
        self.board = board
        self.parent = parent
        self.children = []
        self.movetochild = []
        self.visitcount = 1
        #h#try: 
        #h#    self.anneval, self.movestoexpand = evaltable[" ".join(self.board.fen().split()[0:4])]
        #h#    print("# evaltable ", self.board.fen(), " = ", self.anneval)
        #h#    hitsevaltable += 1
        #h#except KeyError:
        if True: #h#
            self.movestoexpand = Queue()
            if self.board.is_checkmate():
                self.anneval = -999999
            else:
                if board.turn == chess.WHITE:
                    b = self.board
                else:
                    b = self.board.mirror()
                y = modeleval.predict(np.array([extract_features(b)]), batch_size=1, verbose=0) 
                self.anneval = y[0][0] #* (1 if self.board.turn==chess.WHITE else -1)
                print("# evaluated ", self.board.fen(), " = ", self.anneval)
                l = []
                self.terminalnode = True
                for m in self.board.generate_legal_moves():
                    a = str(m).upper()
                    l.append((m, y[1][0][ord(a[0])-65, ord(a[1])-49, ord(a[2])-65, ord(a[3])-49]))
                    self.terminalnode = False
                print("# sorted moves ", end="")
                for m in sorted(l, key=lambda e: e[1], reverse=True): 
                    self.movestoexpand.put(m[0]) # moves ordered via policy values
                    print(str(m[0]), " ", end="")
                print("")
        #h#        evaltable[" ".join(self.board.fen().split()[0:4])] = (self.anneval, deepcopy(self.movestoexpand))
        self.value = self.anneval
        self.isroot = False # root will be moved during game progress


    def boardcopy(self):
        return self.board.copy()

    def setroot(self):
        self.isroot = True


def traverse(parent, mainline=""):

    parent.visitcount += 1
    print("# visit ", parent.visitcount, " of ", mainline if mainline!="" else "root")

    if parent.terminalnode:
        print("# terminal node")
        return parent

    if not parent.movestoexpand.empty():
        # pick_univisted_children and create node for it
        move = parent.movestoexpand.get()
        parent.movetochild.append(move)
        board = parent.boardcopy()
        board.push(move)
        node = MctsNode(board, parent=parent)
        parent.children.append(node)
        print("# picked unvisited ", move)
        return node

    # traverse child with best upper confidence bound
    max = -float('inf')
    bestchild = None
    for i in range(len(parent.children)):
        exploit = -parent.children[i].value
        explore = balance_constant * sqrt( log( parent.visitcount ) / parent.children[i].visitcount )
        e = exploit + explore
        print("# deciding on move ", parent.movetochild[i], \
              " exploit ", exploit, " explore ", explore)
        if e > max:
            bestchild = i
            max = e
    print("# picked move ", parent.movetochild[bestchild], \
          " visited already ", parent.children[bestchild].visitcount, " times")

    return traverse(parent.children[bestchild], mainline+str(parent.movetochild[bestchild])+",")


def backpropagate(node, eval):

    # recalculate max (to be optimized)
    if len(node.children)>1:
        print("# backpropagated value ", eval)
        mx = eval
        for n in node.children:
            mx = max( mx, -n.value) # minus because is adversary
        eval = mx

    if node.movestoexpand.empty() or not node.value:
        node.value = eval
        print("# backpropagated assignment: old node value ", node.value, " new node value ", eval)
        if not node.isroot:
            print("# continue backpropagation")
            backpropagate(node.parent, -eval)
    else:
        print("# backpropagation stops here")

    return


#=========================
root = MctsNode(chess.Board())
root.setroot() # stop backpropagation here
while true:
	leaf = traverse(root) # to pick unvisited/best node
	eval = leaf.value # simulation
	backpropagate(leaf.parent, -eval)
