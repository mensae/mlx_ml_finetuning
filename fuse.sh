#!/bin/bash
VER="v3"
echo "Chosen version ${VER}"
mlx_lm.fuse \
	--model ./models/base/gemma-3-text-27b-it-4bit \
	--adapter-path ./models/adapters/adapter-${VER}_gemma-3-text-27b-it-4bit \
	--save-path ./models/fused/fused-${VER}_gemma-3-text-27b-it-4bit \