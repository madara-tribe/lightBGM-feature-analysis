#!/bin/sh
find . -name '.DS_Store' -type f -ls -delete
rm -rf pandas/results DNN-analysis/results lightBGM/*.png 
rm -rf DNN-analysis/__pycache__ lightBGM/__pycache__
