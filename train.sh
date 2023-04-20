#!/usr/bin/env bash

for i in {1..4}
do
    python src/train.py --batch-size 1 --page 1 --start-pixels 64 --fold $i
done