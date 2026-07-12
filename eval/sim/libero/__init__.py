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

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# robosuite's EGL path parses CUDA_VISIBLE_DEVICES as comma-separated integers.
# In containers this is often set to "all", which crashes on int("all").
if os.environ.get("MUJOCO_EGL_DEVICE_ID") is None:
	cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
	if cuda_visible and not all(part.strip().isdigit() for part in cuda_visible.split(",")):
		os.environ["CUDA_VISIBLE_DEVICES"] = "0"
		os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.libero.libero_env import register_libero_envs
register_libero_envs()
