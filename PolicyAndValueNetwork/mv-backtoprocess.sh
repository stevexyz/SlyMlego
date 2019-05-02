#!/bin/bash

find __inputsalreadyprocessed/ -name "*.pickle" -exec mv {} __inputstobeprocessed/ \;

