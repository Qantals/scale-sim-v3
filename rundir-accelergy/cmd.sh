#!/usr/bin/bash
OUTPUT_DIR=output_yolo_tiny_test
mkdir -p "$OUTPUT_DIR"
./run_all.sh -c ../configs/scale.cfg -t ../topologies/conv_nets/yolo_tiny.csv -p ../test_runs/ -o "$OUTPUT_DIR" > "${OUTPUT_DIR}/run.log" 2>&1
