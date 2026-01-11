#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
../miniconda3/envs/nonsense/bin/python main.py --mode random  --kwargs '{"overwrite_embeddings":True}' &

export CUDA_VISIBLE_DEVICES=1
../miniconda3/envs/nonsense/bin/python main.py --mode baseline  --kwargs '{"overwrite_embeddings":True}' &

wait
echo "All extractions complete."