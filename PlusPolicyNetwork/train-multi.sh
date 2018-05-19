#!/bin/bash

if [ $# -eq 0 ] ; then
    echo Usage: $0 "epdfile [numberofprocesses [initialline [endline]]]" 
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
    endline=$4
fi

slice=$((((endline - initialline) / numberofprocesses) +1))

echo $slice

for ((i=$initialline; i<$endline; i+=$slice)); do
    xterm +hold -e /usr/bin/python3 ./Trainmodel.py $i $slice &
done

exit 0

