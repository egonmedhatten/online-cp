#!/bin/bash

rsync -rlptzv --progress --exclude=.git . johan@192.168.1.130:/home/johan/Documents/Phd/online-cp