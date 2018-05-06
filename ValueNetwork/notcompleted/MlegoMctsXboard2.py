#!/usr/bin/python3

import time
import Const
import Eval
import chess.uci
import sys

balance_constant = 2 # exploration vs exploitation balance
move_timeframe = 10  # number of seconds for a move

# references:
# - https://int8.io/monte-carlo-tree-search-beginners-guide/
# - http://ccg.doc.gold.ac.uk/ccg_old/teaching/ludic_computing/ludic16.pdf


class MctsNode(object):

    def __init__(self, board, parent=None):
        self.parent = parent
        self.children = []
        self.movetochild = []
        self.visitcount = 0
        self.evalann = None
        self.eval = None
        self.board = board
        self.movetoexpand = queue()
        self.terminalnode = True
        for m in self.board.generate_legal_moves(): 
            self.movetoexpand.put(m)
            self.terminalnode = False
        self.isroot = False # moving root during game progress

    def eval(self):
        if not self.eval:
            evalann = Eval(board)
            eval = evalann
        return self.eval

    def boardcopy(self):
        return self.board.copy()

    def setroot(self):
        self.isroot = True

    def pick_univisted_children(self):
        return self.movetoexpand.get()

    def is_empty_univisted_children(self):
        return self.movetoexpand.empty()

    def is_terminal_node(self):
        return self.terminalnode

    def is_root(self):
        return self.isroot

    #def parent(self): #-> cambiare anche isterminal and isroot se funziona??!!!
    #    return self.parent


def best_child_mcts(root, timeframe=10, pickbestlowerconfidencebound=False):

    start_time = time.time()
    while time.time()-start_time < timeframe:
        leaf = traverse(root) # to pick unvisited/best node 
        eval = leaf.eval() # simulation
        backpropagate(leaf.parent, -eval)

    # pick child with highest number of visits
        or alternatively best lower confidence bound
    return best


def traverse(node):

    if node.terminalnode:
        return node
   
    if not node.isempty_unvisited_children():
        # create node
        return MctsNode(node.boardcopy().push(node.pick_univisted_children()), parent=node)

    # traverse child with best upper confidence bound
    max = -infinite
    tn = None
    for n in node.children():
        e = n.eval() + balance_constant * sqrt( log( n.visitcount ) / node.visitcount )
        if e > max:
            tn = n
            max = e 

    return traverse(tn)
 
 
def backpropagate(node, eval):

    node.visitcount += 1

    # recalculate max (to be optimized)
    if not node.terminalnode:
        max = -infinite
        for n in node.children:
            if max < -n.eval:
                max = -n.eval # minus because is adversary
        node.eval = max

    if not node.is_root():
        backpropagate(node.parent, -eval)

    return


#=========================
# main MlegoMctsXboard2.py

if len(sys.argv) > 2:
    print('Xboard engine string: "python3 ', sys.argv[0], '[<modelfile>]"')
    exit(1)

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
        if len(sys.argv)<=1: eval = Eval(quiet=True)
        else: eval = Eval(modelfile=sys.argv[1], quiet=True)
        eval.EvaluatePositionB(chess.Board()) # just to startup engine
        print('feature myname="mcts-mlego-v0.1"')
        print('feature variants="normal"')
        print('feature setboard=1')
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
        board = chess.Board()
        # create root node
        root = MctsNode(chess.Board())
        root.setroot()

    elif parts[0]=='setboard':
        board.set_epd(parts[1])

    elif parts[0]=='ping':
        print('pong '+parts[1])

    elif parts[0]=='undo':
        board.pop()

    elif parts[0]=='force':
        forcemove = True

    elif parts[0]=='usermove':
        board.push_uci(parts[1])

    if parts[0]=='go' or (parts[0]=='usermove' and not forcemove): # usermove already processed
        forcemove = False
        bestmove = None
        bestval = Const.INFINITECP
        for m in board.legal_moves: # single PLY evaluation!
            board.push(m)
            if board.is_checkmate():
                val = -999999
            else:
                val = eval.EvaluatePositionB(board)[0] * (-1 if board.turn==chess.BLACK else 1)
            print("# currmove "+str(m)+" score "+str(-val))
            if val<bestval: # look for minimum value for adversary
                bestmove = m
                bestval = val
            board.pop()

        start_time = time.time()
        while time.time()-start_time < timeframe:
            leaf = traverse(root) # to pick unvisited/best node
            eval = leaf.eval() # simulation
            backpropagate(leaf.parent, -eval)

        # pick child with highest number of visits
        # or alternatively best lower confidence bound
        ...

        root = ...

        print('move %s' % bestmove)
        board.push(bestmove)

    sys.stdout.flush()

exit(0)
