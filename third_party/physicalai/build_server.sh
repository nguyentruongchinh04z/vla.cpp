#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 --model-dir MODEL_DIR [--host-port HOST_PORT] [--device-type DEVICE_TYPE] [--docker-image DOCKER_IMAGE] [--force-build]"
}

MODEL_DIR=""
HOST_PORT="5555"
DEVICE_TYPE="CPU"
DOCKER_IMAGE="${DOCKER_IMAGE:-physicalai_server:latest}"
FORCE_BUILD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)
            MODEL_DIR="${2:?Missing value for --model-dir}"
            shift 2
            ;;
        --host-port)
            HOST_PORT="${2:?Missing value for --host-port}"
            shift 2
            ;;
        --device-type)
            DEVICE_TYPE="${2:?Missing value for --device-type}"
            shift 2
            ;;
        --force-build)
            FORCE_BUILD=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$MODEL_DIR" ]]; then
    echo "Missing required argument: --model-dir" >&2
    usage >&2
    exit 2
fi

echo "MODEL PATH: $MODEL_DIR"
echo "HOST PORT: $HOST_PORT"
echo "DEVICE TYPE: $DEVICE_TYPE"
echo "DOCKER IMAGE: $DOCKER_IMAGE"
echo "FORCE BUILD: $FORCE_BUILD"

if [[ "$FORCE_BUILD" == true ]] || ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    echo "Building Docker image: $DOCKER_IMAGE"
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
