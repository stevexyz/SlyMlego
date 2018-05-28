#!/bin/bash

if [ $# -eq 0 ] ; then
    echo Usage: $0 "epdfile [numberofprocesses [initialline [totallinetoprocess]]]" 
    exit 1
fi

epdfile=$1

if [ -z "$2" ] ; then
    numberofprocesses=8
else
    numberofprocesses=$2
fi

if [ -z "$3" ] ; then
    initialline=0
else
    initialline=$3
fi

if [ -z "$4" ] ; then
    endline=`cat $epdfile | wc -l`
else
    linestoprocess=$4
    endline=$((initialline + linestoprocess))
fi

slice=$((((endline - initialline) / numberofprocesses) +1))

pwd=`pwd`

echo $slice

for ((i=$initialline; i<$endline; i+=$slice)); do
    echo RUN $i $slice \( $initialline $endline \)
    xterm -hold -e "/bin/bash -c 'cd $pwd && /usr/bin/python3 PrepareInput.py $epdfile $i $slice'" &
    sleep 1
done
exit 0

