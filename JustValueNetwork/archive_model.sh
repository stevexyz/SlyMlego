#!/bin/bash

if [ "$1" != "" ]; then
    mkdir "__model_$1"
    mv __model*.* "__model_$1/"
    mv __logs "__model_$1/"
else
    echo "Usage: " $0 " <new_model_name>"
    echo "Model name directory should not already exist."
    echo "Current saved models:"
    ls __model_*
    echo "(__model_ is being added automatically to the name)"
fi

