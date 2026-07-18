#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Priority: first CLI arg > MODEL_DIR env var > empty.
MODEL_DIR="${1:-${MODEL_DIR:-""}}"
echo "MODEL PATH: $MODEL_DIR"
# Priority: second CLI arg > HOST_PORT env var > 5555.
HOST_PORT="${2:-${HOST_PORT:-5555}}"
echo "HOST PORT: $HOST_PORT"
# Priority: third CLI arg > DEVICE_TYPE env var > CPU.
DEVICE_TYPE="${3:-${DEVICE_TYPE:-CPU}}"
echo "DEVICE TYPE: $DEVICE_TYPE"

DOCKER_IMAGE="${DOCKER_IMAGE:-physicalai_server:latest}"
echo "DOCKER IMAGE: $DOCKER_IMAGE"

if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    echo "Docker image not found, building: $DOCKER_IMAGE"
    docker build -t "$DOCKER_IMAGE" -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"
fi

DOCKER_ARGS=(
    --rm -it
    -p "${HOST_PORT}:5555"
    -e "DEVICE_TYPE=${DEVICE_TYPE}"
    --mount "type=bind,source=${MODEL_DIR},target=/workspace/weights,readonly"
)

if [[ -e /dev/accel/accel0 ]]; then
    DOCKER_ARGS+=(--device=/dev/accel/accel0)
else
    echo "Warning: /dev/accel/accel0 not found; skipping NPU device passthrough." >&2
fi

if [[ -e /dev/dri ]]; then
    DOCKER_ARGS+=(--device=/dev/dri)
else
    echo "Warning: /dev/dri not found; skipping GPU device passthrough." >&2
fi

if [[ -e /dev/dri/renderD128 ]]; then
    DOCKER_ARGS+=(--group-add="$(stat -c '%g' /dev/dri/renderD128)")
else
    echo "Warning: /dev/dri/renderD128 not found; skipping --group-add for render group." >&2
fi

docker run "${DOCKER_ARGS[@]}" "$DOCKER_IMAGE"
