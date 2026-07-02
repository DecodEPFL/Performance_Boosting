#!/usr/bin/env bash
set -euo pipefail

: "${GASPAR:?Set GASPAR to your EPFL GASPAR username.}"
: "${PROJECT:?Set PROJECT to your Harbor project name.}"

IMAGE="${IMAGE:-performance-boosting}"
TAG="${TAG:-v1.0}"
REGISTRY="${REGISTRY:-registry.rcp.epfl.ch}"
EXPERIMENT="${EXPERIMENT:-gate}"
GPU="${GPU:-1}"
RUNAI_PROJECT="${RUNAI_PROJECT:-}"
SCRATCH_CLAIM="${SCRATCH_CLAIM:-sci-sti-gft-scratch}"
HOME_CLAIM="${HOME_CLAIM:-home}"
CODE_DIR="${CODE_DIR:-/home/${GASPAR}/Performance_Boosting}"

case "${EXPERIMENT}" in
  gate)
    EXPERIMENT_SCRIPT="experiments/contextual_pb_gate_ssm/Moving_gate_exp.py"
    ;;
  obstacles)
    EXPERIMENT_SCRIPT="experiments/contextual_pb_obstacles_ssm/Moving_obstacles_exp.py"
    ;;
  *)
    echo "Unknown EXPERIMENT='${EXPERIMENT}'. Use EXPERIMENT=gate or EXPERIMENT=obstacles." >&2
    exit 2
    ;;
esac

JOB_NAME="${JOB_NAME:-pb-${EXPERIMENT}-$(date +%Y%m%d-%H%M%S)}"
RUN_ID="${RUN_ID:-${JOB_NAME}}"
IMAGE_URI="${REGISTRY}/${PROJECT}/${IMAGE}:${TAG}"

project_args=()
if [[ -n "${RUNAI_PROJECT}" ]]; then
  project_args=(--project "${RUNAI_PROJECT}")
fi

gpu_args=()
if [[ "${GPU}" == "0" ]]; then
  gpu_args=()
elif [[ "${GPU}" =~ ^[1-9][0-9]*$ ]]; then
  gpu_args=(--gpu-devices-request "${GPU}")
else
  gpu_args=(--gpu-portion-request "${GPU}")
fi

python_cmd="$(printf "%q " python3 "${EXPERIMENT_SCRIPT}" --device cuda --run_id "${RUN_ID}" --no_show_plots "$@")"
remote_cmd="cd $(printf "%q" "${CODE_DIR}") && ${python_cmd}"

echo "Submitting Run:AI job:"
echo "  Job:        ${JOB_NAME}"
echo "  Image:      ${IMAGE_URI}"
echo "  GPU:        ${GPU}"
echo "  Run:AI:     ${RUNAI_PROJECT:-current CLI project}"
echo "  Code dir:   ${CODE_DIR}"
echo "  Run ID:     ${RUN_ID}"
echo "  Experiment: ${EXPERIMENT}"
echo "  Command:    ${remote_cmd}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

runai training standard submit "${JOB_NAME}" \
  "${project_args[@]}" \
  --image "${IMAGE_URI}" \
  "${gpu_args[@]}" \
  --existing-pvc "claimname=${SCRATCH_CLAIM},path=/scratch" \
  --existing-pvc "claimname=${HOME_CLAIM},path=/home/${GASPAR}" \
  --command \
  -- bash -lc "${remote_cmd}"
