#!/usr/bin/python3

# To be verified: "eval" recalculation in backpropagation


import time
import Const
from queue import *
import Eval
import chess.uci
import sys
from math import (sqrt, log)

balance_constant = 3 # exploration vs exploitation balance
move_timeframe = 3  # number of seconds for a move
modeleval = {} # model will be loaded in protover call


# references:
# - http://ccg.doc.gold.ac.uk/ccg_old/teaching/ludic_computing/ludic16.pdf
# - https://int8.io/monte-carlo-tree-search-beginners-guide/


class MctsNode(object):

    def __init__(self, board, parent=None):
        self.parent = parent
        self.children = []
        self.movetochild = []
        self.visitcount = 1
        self.anneval = None
        self.value = None
        self.board = board
        self.movetoexpand = Queue()
        self.terminalnode = True
        for m in self.board.generate_legal_moves():
            self.movetoexpand.put(m)
            self.terminalnode = False
        self.isroot = False # root will be moved during game progress

    def valuepos(self):
        if not self.value:
            self.anneval = modeleval.EvaluatePositionB(self.board)[0] \
                           * (-1 if self.board.turn==chess.WHITE else 1)
            self.value = self.anneval
        return self.value

    def boardcopy(self):
        return self.board.copy()

    def setroot(self):
        self.isroot = True

    def is_empty_unvisited_children(self):
        return self.movetoexpand.empty()


def traverse(parent):

    parent.visitcount += 1

    if parent.terminalnode:
        print("# terminal")
        return parent

    if not parent.is_empty_unvisited_children():
        # pick_univisted_children and create node for it
        move = parent.movetoexpand.get()
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
        e = parent.children[i].valuepos() + \
            balance_constant * sqrt( log( parent.children[i].visitcount ) / parent.visitcount )
        if e > max:
            bestchild = i
            max = e
    print("# picked move ", parent.movetochild[bestchild], \
          " visited already ", parent.children[bestchild].visitcount, " times")

    return traverse(parent.children[bestchild])


def backpropagate(node, eval):

    # recalculate max (to be optimized)
    if not node.terminalnode:
        if len(node.children)>1:
            mx = eval
            for n in node.children:
                mx = max( mx, -n.valuepos()) # minus because is adversary
            eval = mx
            print("# backpropagated recalculation value ", mx)

    print("# old node value ", node.value, " new node value ", eval)
    node.value = eval

    if not node.isroot:
        print("# continue backpropagation")
        backpropagate(node.parent, -eval)

    return


#=========================
# main MlegoMctsXboard2.py

if len(sys.argv) > 3:
    print('Xboard engine string: "python3 ', sys.argv[0], '[[<modelfile>] <bestlowerconfidenceboundselection>]"')
    exit(1)

if len(sys.argv) > 2:
    selection_mode = "best lower confidence bound selection"
else:
    selection_mode = "highest number of visits"

forcemove = False

while True:

    try: line = sys.stdin.readline()
    except KeyboardInterrupt: pass # avoid control-c breaks
    line = line.strip(' \t\n'+chr(3))
    parts = line.split(' ')

    if parts[0]=='xboard':
        print('tellics say Monte Carlo Tree Search experimental chess engine')
        print('tellics say based on SlyMlego deep learning platform')
        print('tellics say by Stefano Marago\' 2018')
        print('tellics say https://github.com/stevexyz/SlyMlego')

    elif parts[0]=='protover' and parts[1]=='2':
        print('feature done=0')
        sys.stdout.flush() # ensure xboard wait to activate network
        print('feature debug=1')
        from Eval import Eval
        if len(sys.argv)<=1:
            modeleval = Eval(quiet=True)
        else:
            modeleval = Eval(modelfile=sys.argv[1], quiet=True)
        modeleval.EvaluatePositionB(chess.Board()) # just to startup engine
        print('feature myname="mcts-mlego-v0.1"')
        print('feature variants="normal"')
        print('feature setboard=0')
        print('feature ping=1')
        print('feature usermove=1')
        print('feature analyze=0')
        print('feature pause=0')
        print('feature nps=0')
        print('feature memory=0')
        print('feature sigint=0')
        print('feature done=1')

    elif parts[0]=='quit':
        break

    elif parts[0]=='new':
        # create root node
        root = MctsNode(chess.Board())

    elif parts[0]=='ping':
        print('pong '+parts[1])

    elif parts[0]=='undo':
        root.isroot = False
        root = root.parent

    elif parts[0]=='force':
        forcemove = True

    elif parts[0]=='usermove':

        # expand unvisited children in case move is there
        while not root.is_empty_unvisited_children():
            move = root.movetoexpand.get()
            root.movetochild.append(move)
            board = root.boardcopy()
            board.push(move)
            node = MctsNode(board, parent=root)
            root.children.append(node)
            print("# created unvisited child ", move)

        for i in range(len(root.children)):
            if root.movetochild[i].uci()==parts[1]:
                root = root.children[i]
                break

    if parts[0]=='go' or (parts[0]=='usermove' and not forcemove): # usermove already processed

        forcemove = False

        # until there is time expand tree with mc ucb selection approach
        start_time = time.time()
        root.setroot() # stop backpropagation here
        while time.time()-start_time < move_timeframe:
            leaf = traverse(root) # to pick unvisited/best node
            eval = leaf.valuepos() # simulation
            backpropagate(leaf.parent, -eval)

        # pick child with highest number of visits
        # or alternatively best lower confidence bound
        # and update root
        bestvalue = -float('inf')
        bestchild = None
        for i in range(len(root.children)):
            if selection_mode=="best lower confidence bound selection":
                # pick child with best lower confidence bound 
                e = root.children[i].valuepos() - \
                    balance_constant * sqrt( log( root.children[i].visitcount ) / root.visitcount )
                if e > bestvalue:
                    bestchild = i
                    bestvalue = e
            elif selection_mode=="highest number of visits":
                # pick child with highest number of visits
                if root.children[i].visitcount > bestvalue:
                    bestchild = i
                    bestvalue = root.children[i].visitcount
            else:
                raise ValueError('Selection mode "%s" not recognized' % selection_mode)

        print('move %s' % root.movetochild[bestchild])
        root = root.children[bestchild]

    sys.stdout.flush()

exit(0)
