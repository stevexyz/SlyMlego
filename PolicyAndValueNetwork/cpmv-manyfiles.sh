#!/bin/bash

if [ "$1" != "" ]; then
    find $2 -name "*.pickle" -exec $1 {} $3 \;
else
    echo "Usage: " $0 " <command_mv_or_cp> <path_from> <path_to>"
    echo "Move or copy pickle files from one directory to another"
    echo "Useful when there are more files than shell expansion can manage"
fi

