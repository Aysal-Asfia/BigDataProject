#! /bin/bash

scp -i ~/.ssh/simgridvm-key -p centos@206.167.182.108:/mnt/bigdata/project/atop.log log/
scp -i ~/.ssh/simgridvm-key -p centos@206.167.182.108:/mnt/bigdata/project/collectl-simgrid-vm-*.dsk log/
scp -i ~/.ssh/simgridvm-key -p centos@206.167.182.108:/mnt/bigdata/project/timestamps.json log/