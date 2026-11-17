#!/bin/bash
# Download model weights from HuggingFace into project storage, as a Slurm job.
#
# Isambard's file-transfer guide and FAQ: login nodes are limited to 1 CPU core and
# 4 GiB of memory, and large downloads there are likely to be killed, so run them as
# jobs. Isambard-AI jobs need at least 1 GPU, so this is charged at that minimum
# (0.25 NHR per hour) for as long as it runs, not for the full --time.
#
# Prerequisites: the model's licence accepted on HuggingFace, `hf auth login` done on a
# login node, and .venv built.
#
# Submit from the repo root:
#   sbatch scripts/isambard_ai/download_model.sbatch
# Another model:
#   HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct sbatch scripts/isambard_ai/download_model.sbatch
#
#SBATCH --job-name=ubuntuguard-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

set -eo pipefail

HF_MODEL_ID="${HF_MODEL_ID:-meta-llama/Llama-3.3-70B-Instruct}"
MODEL_DIR="${MODEL_DIR:-${PROJECTDIR:?PROJECTDIR is not set}/${USER}/models/${HF_MODEL_ID#*/}}"

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
set -u

# Keep the downloader's chunk cache off the 100 GiB $HOME quota. The token written by
# `hf auth login` stays in $HOME, where this job reads it.
export HF_XET_CACHE="${PROJECTDIR}/${USER}/xet_cache"
mkdir -p "${MODEL_DIR}"

echo "Downloading ${HF_MODEL_ID} to ${MODEL_DIR}"
# original/ holds Meta's own checkpoint format (another ~140 GB for 70B); vLLM only
# needs the HuggingFace-format safetensors files.
hf download "${HF_MODEL_ID}" --local-dir "${MODEL_DIR}" --exclude "original/*"

echo "safetensors files: $(ls "${MODEL_DIR}" | grep -c safetensors)"
du -sh "${MODEL_DIR}"
echo "Download complete."
