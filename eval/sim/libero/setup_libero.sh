#!/usr/bin/env bash
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euxo pipefail

# egl-probe and friends request cmake_minimum_required < 3.5, which CMake >= 4.0
# refuses; this lets them configure anyway (env var honored since CMake 3.31).
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set paths relative to script location
LIBERO_REPO="$SCRIPT_DIR/LIBERO"
LIBERO_UV_ENV="$SCRIPT_DIR/libero_uv"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GITMODULES_PATH="$REPO_ROOT/.gitmodules"
LIBERO_GIT_URL="$(git config -f "$GITMODULES_PATH" --get submodule.external_dependencies/LIBERO.url 2>/dev/null || true)"

if [ -z "$LIBERO_GIT_URL" ]; then
	LIBERO_GIT_URL="https://github.com/Lifelong-Robot-Learning/LIBERO.git"
fi

# Fallback for branches where LIBERO is listed in .gitmodules but not tracked as a gitlink.
mkdir -p "$(dirname "$LIBERO_REPO")"
if [ -d "$LIBERO_REPO/.git" ]; then
	echo "LIBERO repo already exists at $LIBERO_REPO, reusing existing checkout."
elif [ -d "$LIBERO_REPO" ] && [ -n "$(ls -A "$LIBERO_REPO" 2>/dev/null)" ]; then
	echo "Directory $LIBERO_REPO already exists and is not empty, skipping clone."
else
	git clone "$LIBERO_GIT_URL" "$LIBERO_REPO"
fi

rm -rf "$LIBERO_UV_ENV"
mkdir -p "$LIBERO_UV_ENV"
# Keep uv cache local to avoid failures on broken/unwritable ~/.cache mounts.
uv venv "$LIBERO_UV_ENV/.venv" --python 3.10
source "$LIBERO_UV_ENV/.venv/bin/activate"
uv pip install -e "$LIBERO_REPO" --config-settings editable_mode=compat
uv pip install \
	hydra-core==1.2.0 \
	numpy==1.26.4 \
	wandb==0.13.1 \
	easydict==1.9 \
	transformers==4.51.3 \
	opencv-python==4.6.0.66 \
	robomimic==0.2.0 \
	einops==0.8.1 \
	thop==0.1.1-2209072238 \
	robosuite==1.4.0 \
	bddl==1.0.1 \
	future==0.18.2 \
	matplotlib==3.5.3 \
	cloudpickle==2.1.0 \
	gym==0.25.2 \
	torch==2.5.1 \
	torchvision==0.20.1 \
	pydantic \
	av \
	tianshou==0.5.1 \
	tyro \
	pandas==2.0.3 \
	dm_tree \
	albumentations==1.4.18 \
	zmq \
	msgpack==1.1.0 \
	msgpack-numpy==0.4.8 \
	gymnasium==0.29.1 \
	pyarrow==12.0.1 \
	diffusers==0.30.1
# robosuite 1.4.0 calls the mujoco 2.3 mj_fullM(m, dst, M) signature; mujoco 3.x
# changed it, so a transitive dep pulling 3.x breaks the OSC controller at reset.
uv pip install mujoco==2.3.2

# LIBERO prompts for a dataset path on first import if ~/.libero/config.yaml is
# missing, which hangs a headless eval; clear stale config and seed defaults (N).
rm -rf "$HOME/.libero"
echo "N" | MUJOCO_GL=egl python -c "import libero.libero" || true
