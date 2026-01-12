#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
../miniconda3/envs/nonsense/bin/python main.py --mode peak  --kwargs '{"overwrite_embeddings":True, "overwrite_contexts": True}' &

export CUDA_VISIBLE_DEVICES=1
../miniconda3/envs/nonsense/bin/python main.py --mode valley  --kwargs '{"overwrite_embeddings":True, "overwrite_contexts": True}' &

wait
echo "All extractions complete."