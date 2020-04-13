#! /bin/bash

scp -i ~/.ssh/simgridvm-key -p preprocess.py centos@206.167.182.108:/mnt/bigdata/project/preprocess.py
scp -i ~/.ssh/simgridvm-key -p train_test.py centos@206.167.182.108:/mnt/bigdata/project/train_test.py
scp -i ~/.ssh/simgridvm-key -p pipeline.py centos@206.167.182.108:/mnt/bigdata/project/pipeline.py
scp -i ~/.ssh/simgridvm-key -p benchmark.sh centos@206.167.182.108:/mnt/bigdata/project/benchmark.sh