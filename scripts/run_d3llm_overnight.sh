#!/bin/bash
# Overnight d3LLM trajectory training to convergence on 500 examples.
set -uo pipefail
cd ~/Documents/GitHub/Open-dLLM
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=FASTEST_FIRST PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=/tmp/d3llm_overnight.log
PY=.venv/bin/python
TRAJ=/home/johndpope/ds_offload/trajectories/qwen3-1.7b-500/trajectories.jsonl
ANCH=/home/johndpope/ds_offload/anchors/qwen3-1.7b-500

echo "[orch $(date)] waiting for trajectory generation (PID $(cat /tmp/traj_gen_500.pid 2>/dev/null))..." >> $LOG
while kill -0 $(cat /tmp/traj_gen_500.pid 2>/dev/null) 2>/dev/null; do sleep 30; done
echo "[orch $(date)] trajectories: $(wc -l < $TRAJ 2>/dev/null) lines" >> $LOG
if [ ! -s "$TRAJ" ]; then echo "[orch] ERROR: no trajectories, aborting" >> $LOG; exit 1; fi

echo "[orch $(date)] dumping anchors (500 ex, layers 7,14,21,28, seq 2048)..." >> $LOG
$PY scripts/precompute_anchor.py --model_path Qwen/Qwen3-1.7B \
  --data_path /home/johndpope/ds_offload/data_overfit_500.jsonl \
  --output_dir $ANCH --layers "7,14,21,28" --max_seq_len 2048 --max_examples 500 >> $LOG 2>&1
echo "[orch $(date)] anchors: $(find $ANCH -name "*.safetensors" 2>/dev/null | wc -l) chunks" >> $LOG

echo "[orch $(date)] launching training to convergence (50 epochs)..." >> $LOG
$PY tasks/train_torch.py configs/pretrain/d3llm_1_7b_overnight.yaml >> $LOG 2>&1
echo "[orch $(date)] TRAINING DONE (exit $?)" >> $LOG
