#!/bin/bash
VER="v3"
echo "Chosen version ${VER}"
mlx_lm.lora \
	--model ./models/base/gemma-3-text-27b-it-4bit/ \
	--train \
	--adapter-path ./models/adapters/adapter-${VER}_gemma-3-text-27b-it-4bit \
	--data data/for_train/${VER} \
	--iters 60 \
	--num-layers 32