import sys

# directories and files
TOBEPROCESSEDDIR = "__inputstobeprocessed"
ALREADYPROCESSEDDIR = "__inputsalreadyprocessed"
VALIDATIONDATADIR = "__validationdata"
MODELFILE = "__model"

#print("Platform:", sys.platform)
if sys.platform == "Windows" or sys.platform == "win32":
    ENGINE1 = "c:\\Portable Programs\\stockfish-10-win\\Windows\\stockfish_10_x64_popcnt.exe"
    from pathlib import Path
    enginefile = Path(ENGINE1)
    if not enginefile.is_file():
        error("Correct the path for the engine in the file Const.py") 
else:
    ENGINE1 = "stockfish"

# engine configuration for input preparation
MOVETIME = 2 # seconds, increase when model starts to be good...
ENGINETHREADS = 10 # to be edited depending on system...
HASHSIZE = 8000 # to be edited depending on system...

#OPENINGBOOK = "__book.bin"
INFINITECP = 2000 # centipawn
