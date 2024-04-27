#!/usr/bin/env bash

# script to run distillation, arguments are in config.py

TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=6 python main.py wow.p_p_qc099_q_qc099_r_top15_len_step50