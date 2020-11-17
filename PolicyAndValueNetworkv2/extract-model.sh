#!/bin/bash
echo "" > __model.descr
echo "FEATURES ------------------------------" >> __model.descr
cat FeaturesExtraction.py | sed '1,/\@featuresbegin/d;/\@featuresend/,$d' >> __model.descr
echo "" >> __model.descr
echo "MODEL ------------------------------" >> __model.descr
cat $1 | sed '1,/\@modelbegin/d;/\@modelend/,$d' >> __model.descr
