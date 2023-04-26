#!/usr/bin/env bash

for i in {0..4}
do
    python src/train.py --batch-size 1 --page 1 --start-pixels 48 --fold $i
done