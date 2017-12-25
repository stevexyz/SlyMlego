#!/bin/bash
tensorboard --logdir=__logs/ &
sleep 2
#chromium-browser "http://localhost:6006"
firefox "http://localhost:6006"
kill %1
