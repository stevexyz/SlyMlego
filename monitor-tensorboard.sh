#!/bin/bash
tensorboard --logdir=__logs/ &
sleep 5
#chromium-browser "http://localhost:6006"
firefox "http://localhost:6006"
wait $!

