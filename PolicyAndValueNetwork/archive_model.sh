#!/bin/bash

if [ "$1" != "" ]; then
    mkdir "__archivemodel_$1"
    mv __model*.* "__archivemodel_$1/"
    mv __logs "__archivemodel_$1/"
    echo "model archived to \"__archivemodel_$1\""
else
    echo ""
    echo "Usage: " $0 " <new_model_name>"
    echo "Note: \"__archivemodel_\" is being added automatically to the name) and directory should not already exists."
    echo ""
    echo "Current saved models:"
    ls -1 -d __archivemodel_*
    echo ""
fi

