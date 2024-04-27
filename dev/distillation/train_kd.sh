#!/usr/bin/env bash

# script to run distillation, arguments are under config dir
# input config filename as an argument

TRANSFORMERS_OFFLINE=0 CUDA_VISIBLE_DEVICES=0 python main.py wow.p_qc099_step300