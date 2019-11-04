#!/bin/bash

dating=`date`

git status    # Git show the status
git add .   # Git add the changes into the workspace
git commit -m "$dating $1"   # commit the changes
git push origin master   # push commit into the Github in the back station and kill the output.
git log    # show the log
# taqini say 0x29a , No 0x29a , I am so weak :)