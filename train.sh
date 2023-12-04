#!/bin/bash
    
CUDA_VISIBLE_DEVICES=0 \
python main.py \
	--train_1 /ABInet_dataset/MJ/MJ_train \
	--train_2 /ABInet_dataset/ST \
	--train_3 /ABInet_dataset/SA \
	--test_1 /ABInet_dataset/real_data \
	--LR True \
	--batchSize 128 \
	--niter 6 \
	--lr 1 \
	--cuda \
	--displayInterval 1000 \
	--valInterval 5000 \
	--n_bm 5 \
	--val_start_epoch 1.0\
       # --MODEL "your model path"
    
