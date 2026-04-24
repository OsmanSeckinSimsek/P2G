#!/bin/bash

# Strong scaling study: submit run_800.sh for 2, 4, 8, 16 GPUs.
# Assumes a maximum of 4 GPUs per node.
# sbatch CLI flags override the #SBATCH directives inside run_800.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/run_800.sh"

# GPU count -> "nodes ntasks_per_node gpus_per_node"
declare -A CFG
CFG[2]="1 2 2"
CFG[4]="1 4 4"
CFG[8]="2 4 4"
CFG[16]="4 4 4"

for NGPU in 2 4 8 16; do
    read -r NODES TASKS_PER_NODE GPUS_PER_NODE <<< "${CFG[$NGPU]}"

    echo "Submitting: ${NGPU} GPUs  (${NODES} node(s) x ${TASKS_PER_NODE} tasks/node)"

    sbatch \
        --nodes="${NODES}" \
        --ntasks-per-node="${TASKS_PER_NODE}" \
        --gres="gpu:${GPUS_PER_NODE}" \
        --output="scaling_${NGPU}gpu.out" \
        --export=ALL,NGPU="${NGPU}" \
        "${JOB_SCRIPT}"
done
