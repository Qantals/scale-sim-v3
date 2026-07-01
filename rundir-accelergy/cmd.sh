#!/usr/bin/bash
mkdir -p output
./run_all.sh -c ../configs/scale.cfg -t ../topologies/conv_nets/test.csv -p ../test_runs/ -o ./output > ./output/run.log 2>&1
