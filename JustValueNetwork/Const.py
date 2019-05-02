
# directories and files
TOBEPROCESSEDDIR = "__inputstobeprocessed"
ALREADYPROCESSEDDIR = "__inputsalreadyprocessed"
VALIDATIONDATADIR = "__validationdata"
MODELFILE = "__model"

# engine configuration for input preparation
MOVETIME = 5000 # increase when model starts to be good
ENGINE1 = "stockfish"
OPENINGBOOK = "__book.bin"
INFINITECP = 2000 # 20 centipawn...

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
