#!/usr/bin/python3

# To be verified: "eval" recalculation in backpropagation


import time
import Const
from queue import *
import Eval
import chess.uci
import sys
from math import (sqrt, log)

balance_constant = 50 # exploration vs exploitation balance
move_timeframe = 2  # number of seconds for a move
modeleval = None # model will be loaded in protover call
evaltable = {} # hash store for evaluated positions
hitsevaltable = 0


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
        self.movestoexpand = Queue()
        self.terminalnode = True
        for m in self.board.generate_legal_moves():
            self.movestoexpand.put(m)
            self.terminalnode = False
        self.isroot = False # root will be moved during game progress

    def valuepos(self):
        global hitsevaltable
        if not self.value:
            try: 
                self.anneval = evaltable[" ".join(self.board.fen().split()[0:4])]
                print("# evaltable ", self.board.fen(), " = ", self.anneval)
                hitsevaltable += 1
            except KeyError:
                if self.board.is_checkmate():
                    self.anneval = -999999
                else:
                    self.anneval = modeleval.EvaluatePositionB(self.board)[0] \
                                   * (1 if self.board.turn==chess.WHITE else -1)
                evaltable[" ".join(self.board.fen().split()[0:4])] = self.anneval
                print("# evaluated ", self.board.fen(), " = ", self.anneval)
            self.value = self.anneval
        return self.value

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
        exploit = -parent.children[i].valuepos()
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
            mx = max( mx, -n.valuepos()) # minus because is adversary
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
# main MlegoMctsXboard2.py

if len(sys.argv) > 4:
    print('Xboard engine string: "python3 ', sys.argv[0], '[<modelfile> [<exploitvsexploreconstant> [<bestlowerconfidenceboundselection>]]]"')
    exit(1)

if len(sys.argv) > 3:
    selection_mode = "best lower confidence bound selection"
else:
    selection_mode = "highest number of visits"

if len(sys.argv) > 2:
    balance_constant = float(sys.argv[2])

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

    elif parts[0]=='level':
        # e.g. "level 0 1:05 3"
        move_timeframe = ( int((parts[2].split(":"))[0])*60 +  \
                           int((parts[2].split(":"))[1]) )/60 + int(parts[3])*0.95
        print("# new timeframe ", move_timeframe)

    elif parts[0]=='ping':
        print('pong '+parts[1])

    elif parts[0]=='undo':
        root.isroot = False
        root = root.parent

    elif parts[0]=='force':
        forcemove = True

    elif parts[0]=='usermove':

        # expand unvisited children in case move is there
        while not root.movestoexpand.empty():
            move = root.movestoexpand.get()
            root.movetochild.append(move)
            board = root.boardcopy()
            board.push(move)
            node = MctsNode(board, parent=root)
            root.children.append(node)
            print("# created unvisited child ", move)

        for i in range(len(root.children)):
            if root.movetochild[i].uci()==parts[1]:
                print('# new root is %s' % root.movetochild[i])
                root = root.children[i]
                break

    if parts[0]=='go' or (parts[0]=='usermove' and not forcemove): # usermove already processed

        forcemove = False

        # until there is time expand tree with mc ucb selection approach
        start_time = time.time()
        root.setroot() # stop backpropagation here
        calculated_positions = 0
        while time.time()-start_time < move_timeframe:
            leaf = traverse(root) # to pick unvisited/best node
            eval = leaf.valuepos() # simulation
            backpropagate(leaf.parent, -eval)
            calculated_positions += 1
        print('# ', calculated_positions, ' positions evaluated with ', hitsevaltable, ' evaltable hits')

        # pick best child and update root
        bestvalue = -float('inf')
        bestchild = None
        for i in range(len(root.children)):
            if selection_mode=="best lower confidence bound selection":
                # pick child with best lower confidence bound
                if not bestchild: print('# pick child with best lower confidence bound')
                e = -root.children[i].valuepos() - \
                    balance_constant * sqrt( log( root.visitcount ) / root.children[i].visitcount )
                if e > bestvalue:
                    bestchild = i
                    bestvalue = e
            elif selection_mode=="highest number of visits":
                # pick child with the highest number of visits
                if not bestchild: print('# pick child with the highest number of visits')
                if root.children[i].visitcount > bestvalue:
                    bestchild = i
                    bestvalue = root.children[i].visitcount
            else:
                raise ValueError('Selection mode "%s" not recognized' % selection_mode)

        print('move %s' % root.movetochild[bestchild])
        print('# new root is %s' % root.movetochild[bestchild])
        root = root.children[bestchild]

    sys.stdout.flush()

exit(0)
