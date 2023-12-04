#!/bin/bash

GPU=0
    
CUDA_VISIBLE_DEVICES=0 \
python val.py \
	--test_1 /home/zdz/str/evaluation/IIIT5k_3000 \
	--LR True \
	--cuda \
	--n_bm 5 \
	--MODEL "your model path"
   