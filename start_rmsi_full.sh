#!/bin/bash
set -e

CODE_DIR=xxx
EXP_DIR=xxx
CHECKPOINT=xxx

cd "${CODE_DIR}"
mkdir -p "${EXP_DIR}"/{checkpoints,logs,wandb,hydra}

CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=online \
WANDB_DIR="${EXP_DIR}/wandb" \
LD_LIBRARY_PATH= \
accelerate launch --num_processes 1 --mixed_precision fp16 train_stereo.py \
  --config-name train_sceneflow_single4090 \
  checkpoint="${CHECKPOINT}" \
  accelerator.mixed_precision=fp16 \
  model.instance.cfg.use_legacy_structure_prompt=False \
  model.instance.cfg.msca.enabled=True \
  model.instance.cfg.msca.gamma=0.05 \
  model.instance.cfg.cvum.enabled=True \
  model.instance.cfg.cvum.temperature=1.0 \
  model.instance.cfg.cvum.tau_d=0.5 \
  model.instance.cfg.cvum.mode=entropy_peak \
  rapc.enabled=True \
  rapc.lambda_p=0.01 \
  rapc.iteration_gamma=0.9 \
  rapc.mode=support \
  save_path="${EXP_DIR}/checkpoints" \
  tracker.init_kwargs.wandb.name=rmsi_stereo_full \
  +tracker.init_kwargs.wandb.group=rmsi_stereo \
  hydra.run.dir="${EXP_DIR}/hydra" \
  2>&1 | tee "${EXP_DIR}/logs/train.log"
