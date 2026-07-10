#!/usr/bin/bash
OUTPUT_DIR=output/yolo_tiny
mkdir -p "$OUTPUT_DIR"
./run_all.sh -c ../configs/scale.cfg -t ../topologies/conv_nets/yolo_tiny.csv -p ../test_runs/ -o "$OUTPUT_DIR" -f sparse_npy > "${OUTPUT_DIR}/run.log" 2>&1
