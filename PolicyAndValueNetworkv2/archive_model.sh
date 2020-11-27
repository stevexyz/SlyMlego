#!/bin/bash

if [ "$1" != "" ]; then
    mkdir -p "__archivedmodels/$1"
    mv __model*.* "__archivedmodels/$1/"
    mv __logs "__archivedmodels/$1/"
    cp FeaturesExtraction.py "__archivedmodels/$1/"
    echo "model archived to \"__archivedmodels/$1\""
else
    echo ""
    echo "Usage: " $0 " <new_model_name>"
    echo "Models files are stored in the \"__archivemodels\" directory."
    echo ""
    echo "Current saved models:"
    ls -1 -d __archivedmodels/*
    echo ""
fi

