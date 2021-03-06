#! /bin/bash

rm -f timestamps.json
rm -f collectl-simgrid-vm*.*
rm -f atop.log

sudo echo 3 | sudo tee /proc/sys/vm/drop_caches

collectl -sCDnfM -omT --dskopts z --cpuopts z -i 1 --sep , -P -f collectl --procfilt P p  &
atop -P MEM 1 400 > atop.log &

python3 pipeline.py 1 2

