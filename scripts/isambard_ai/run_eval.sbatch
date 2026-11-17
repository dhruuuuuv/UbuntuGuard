#!/bin/bash
# Evaluate one model on UbuntuGuard data files on Isambard-AI (by default, all three
# full splits).
#
# Submit from the repo root, pointing at a local weights folder:
#   MODEL_NAME=$PROJECTDIR/$USER/models/Llama-3.3-70B-Instruct sbatch scripts/isambard_ai/run_eval.sbatch
#
# Defaults suit Llama-3.3-70B-Instruct (~140 GB of weights), which needs two
# 96 GB GH200s. For a model that fits on one GPU, override both together:
#   MODEL_NAME=... TP_SIZE=1 sbatch --gpus=1 scripts/isambard_ai/run_eval.sbatch
#
# Evaluate other data files. Single quotes stop your shell expanding the pattern
# before the job sees it:
#   MODEL_NAME=... DATA_GLOB='data/kenya_health/*.jsonl' sbatch scripts/isambard_ai/run_eval.sbatch
#
# Answer-only baseline (no chain-of-thought):
#   MODEL_NAME=... EXTRA_ARGS=--no-cot sbatch scripts/isambard_ai/run_eval.sbatch
#
#SBATCH --job-name=ubuntuguard-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -eo pipefail

# Required: an explicit model avoids silently downloading ~140 GB by default
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME to a local weights folder or a HuggingFace model ID}"
MODEL_TYPE="${MODEL_TYPE:-llama}"
TP_SIZE="${TP_SIZE:-2}"
DATA_GLOB="${DATA_GLOB:-data/Ubuntu_guard_test_*.jsonl}"

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

# Multi-GPU settings from Isambard's vLLM tutorial: Isambard's NCCL build (the library
# GPUs use to talk to each other under tensor parallelism) and vLLM's symmetric-memory
# communication disabled. Loaded before `set -u`, as module scripts may use unset vars.
type module >/dev/null 2>&1 || source /etc/profile
module load brics/nccl
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_NCCL_USE_SYMM_MEM=0
set -u

if [[ -d "${MODEL_NAME}" ]]; then
    # Local weights: never contact HuggingFace, so no token or network is needed
    export HF_HUB_OFFLINE=1
    echo "Loading local weights from ${MODEL_NAME}"
else
    # Keep downloaded weights off the 100 GiB $HOME quota. Only the hub cache moves:
    # relocating HF_HOME would also move where `hf auth login` stored the token.
    export HF_HUB_CACHE="${HF_HUB_CACHE:-${PROJECTDIR:-$HOME}/hf_cache}"
    echo "Loading ${MODEL_NAME} via HuggingFace cache ${HF_HUB_CACHE}"
fi

nvidia-smi

# Fail fast rather than letting vLLM hang waiting for GPUs the job doesn't have
VISIBLE_GPUS=$(nvidia-smi -L | wc -l)
if (( TP_SIZE > VISIBLE_GPUS )); then
    echo "TP_SIZE=${TP_SIZE} but only ${VISIBLE_GPUS} GPU(s) allocated; pass --gpus=${TP_SIZE} to sbatch" >&2
    exit 1
fi

# One process per split rather than --data_dir: evaluate.py builds a new vLLM engine
# per split, and in a single process the previous engine's GPU memory (especially
# with tensor parallelism) is not reliably released, so later splits can OOM.
# A failed split is recorded and the remaining splits still run.
shopt -s nullglob
SPLITS=( ${DATA_GLOB} )
shopt -u nullglob
if (( ${#SPLITS[@]} == 0 )); then
    echo "No data files match DATA_GLOB=${DATA_GLOB}" >&2
    exit 1
fi
echo "Evaluating ${#SPLITS[@]} file(s): ${SPLITS[*]}"

FAILED=()
for split in "${SPLITS[@]}"; do
    echo "===== ${split} ====="
    # 8192 rather than the 4096 default: translated African-language prompts tokenise
    # far longer than English and one over-length prompt aborts the vLLM batch.
    if ! python evaluate.py \
        --data_path "${split}" \
        --model_name "${MODEL_NAME}" \
        --model_type "${MODEL_TYPE}" \
        --tensor_parallel_size "${TP_SIZE}" \
        --max_model_len 8192 \
        --output_dir results/ ${EXTRA_ARGS:-}; then
        FAILED+=("${split}")
    fi
done

if (( ${#FAILED[@]} > 0 )); then
    echo "FAILED splits: ${FAILED[*]}" >&2
    exit 1
fi
echo "All splits completed."
