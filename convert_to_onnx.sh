#!/bin/bash

search_dir=networks

mkdir -p prepared_networks

for entry in "$search_dir"/*
do
    network_out="${entry/".pb"/".onnx"}"
    lc0/lc0 leela2onnx --input=$entry --output=$network_out
done