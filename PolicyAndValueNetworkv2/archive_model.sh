#!/bin/bash

if [ "$1" != "" ] && [ "$2" != "" ]; then
    mkdir -p "__archivedmodels/$1"
    mv "$2"/__model* "__archivedmodels/$1/"
    mv "$2"/__logs "__archivedmodels/$1/"
    cp "$2"/FeaturesExtraction.py "__archivedmodels/$1/"
    echo "model archived to \"__archivedmodels/$1\""
else
    echo ""
    echo "Usage: " $0 " <new_model_name> <from_directory>"
    echo "Example: " $0 "test_model_1 ."
    echo "Example: " $0 "test_model_2 /ramdisk0/PolicyAndValueNetworkv2"
    echo "Models files are stored in the \"__archivemodels\" directory."
    echo ""
    echo "Current saved models:"
    ls -1 -d __archivedmodels/*
    echo ""
fi

