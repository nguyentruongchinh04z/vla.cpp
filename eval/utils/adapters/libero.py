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

import math
from typing import Any
import numpy as np
import torch
import einops
from tree import map_structure

from utils.adapters.base import (
    BasePipelineAdapter, 
    convert_nested_dict
)


class LeRobotPipelineAdapter(BasePipelineAdapter):
    def __init__(self, client: Any = None):
        super().__init__(client)

    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        parsed_obs = map_structure(
            lambda x: x[None] if isinstance(x, np.ndarray) else x, obs
        )
        parsed_obs = self._preprocess_observation(parsed_obs)
        for key in list(parsed_obs.keys()):
            if key.startswith(f"observation.images."):
                img = parsed_obs[key]
                # Flip both H and W
                img = torch.flip(img, dims=[2, 3])
                parsed_obs[key] = img
        # Process robot_state into a flat state vector
        observation_robot_state_str = "observation." + "robot_state"
        if observation_robot_state_str in parsed_obs:
            robot_state = parsed_obs.pop(observation_robot_state_str)

            # Extract components
            eef_pos = robot_state["eef"]["pos"]  # (B, 3,)
            eef_quat = robot_state["eef"]["quat"]  # (B, 4,)
            gripper_qpos = robot_state["gripper"]["qpos"]  # (B, 2,)

            # Convert quaternion to axis-angle
            eef_axisangle = self._quat2axisangle(eef_quat)  # (B, 3)
            # Concatenate into a single state vector
            state = torch.cat((eef_pos, eef_axisangle, gripper_qpos), dim=-1)

            # ensure float32
            state = state.float()
            if state.dim() == 1:
                state = state.unsqueeze(0)

            parsed_obs["observation.state"] = state
        
        parsed_obs = map_structure(
            lambda x: (x.numpy()[0] if isinstance(x, torch.Tensor) else x), parsed_obs
        )
        parsed_obs["task"] = obs.get("task_description", "")
        return parsed_obs

    def parse_action(self, action: np.ndarray) -> np.ndarray:
        return action
    
    @staticmethod
    def _preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        # map to expected inputs for the policy
        return_observations = {}
        if "pixels" in observations:
            if isinstance(observations["pixels"], dict):
                imgs = {f"observation.images.{key}": img for key, img in observations["pixels"].items()}
            else:
                imgs = {"observation.images": observations["pixels"]}

            for imgkey, img in imgs.items():
                # TODO(aliberts, rcadene): use transforms.ToTensor()?
                img_tensor = torch.from_numpy(img)

                # When preprocessing observations in a non-vectorized environment, we need to add a batch dimension.
                # This is the case for human-in-the-loop RL where there is only one environment.
                if img_tensor.ndim == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                # sanity check that images are channel last
                _, h, w, c = img_tensor.shape
                assert c < h and c < w, f"expect channel last images, but instead got {img_tensor.shape=}"

                # sanity check that images are uint8
                assert img_tensor.dtype == torch.uint8, f"expect torch.uint8, but instead {img_tensor.dtype=}"

                # convert to channel first of type float32 in range [0,1]
                img_tensor = einops.rearrange(img_tensor, "b h w c -> b c h w").contiguous()
                img_tensor = img_tensor.type(torch.float32)
                img_tensor /= 255

                return_observations[imgkey] = img_tensor

        if "environment_state" in observations:
            env_state = torch.from_numpy(observations["environment_state"]).float()
            if env_state.dim() == 1:
                env_state = env_state.unsqueeze(0)

            return_observations["observation.environment_state"] = env_state

        if "agent_pos" in observations:
            agent_pos = torch.from_numpy(observations["agent_pos"]).float()
            if agent_pos.dim() == 1:
                agent_pos = agent_pos.unsqueeze(0)
            return_observations["observation.state"] = agent_pos

        if "robot_state" in observations:
            return_observations[f"observation.robot_state"] = convert_nested_dict(observations["robot_state"])

        # Handle IsaacLab Arena format: observations have 'policy' and 'camera_obs' keys
        if "policy" in observations:
            return_observations[f"observation.policy"] = observations["policy"]

        if "camera_obs" in observations:
            return_observations[f"observation.camera_obs"] = observations["camera_obs"]

        return return_observations
    
    @staticmethod
    def _quat2axisangle(quat: torch.Tensor) -> torch.Tensor:
        if not isinstance(quat, torch.Tensor):
            raise TypeError(f"_quat2axisangle expected a torch.Tensor, got {type(quat)}")

        if quat.ndim != 2 or quat.shape[1] != 4:
            raise ValueError(f"_quat2axisangle expected shape (B, 4), got {tuple(quat.shape)}")
        
        quat = quat.to(dtype=torch.float32)
        device = quat.device
        batch_size = quat.shape[0]

        w = quat[:, 3].clamp(-1.0, 1.0)

        den = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))

        result = torch.zeros((batch_size, 3), device=device)

        mask = den > 1e-10

        if mask.any():
            angle = 2.0 * torch.acos(w[mask])  # (M,)
            axis = quat[mask, :3] / den[mask].unsqueeze(1)
            result[mask] = axis * angle.unsqueeze(1)

        return result



class Evo1PipelineAdapter(BasePipelineAdapter):
    def __init__(self, client: Any = None):
        super().__init__(client)

    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        front_img = np.ascontiguousarray(obs["pixels"]["image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["pixels"]["image2"][::-1, ::-1])

        return {
            "image": [front_img, wrist_img, np.zeros_like(front_img)],
            "state": np.concatenate((
                obs["robot_state"]["eef"]["pos"],
                self.quat2axisangle(obs["robot_state"]["eef"]["quat"]),
                obs["robot_state"]["gripper"]["qpos"],
            )),
            "prompt": obs["task_description"],
            "image_mask": [1, 1, 0],
            "action_mask": [1] * 7 + [0] * 17,
        }

    def parse_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action[:7], dtype=np.float32).copy()
        action[6] = -1.0 if action[6] > 0.5 else 1.0
        return action

    @staticmethod
    def encode_image_array(img_array: np.ndarray):
        return img_array.astype(np.uint8).tolist()

    @staticmethod
    def quat2axisangle(quat):
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0
        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            return np.zeros(3)
        return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class Gr00tPipelineAdapter(BasePipelineAdapter):

    def __init__(self, client: Any = None):
        super().__init__(client)

    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:

        front = np.ascontiguousarray(obs["pixels"]["image"][::-1, ::-1]).astype(np.uint8)
        wrist = np.ascontiguousarray(obs["pixels"]["image2"][::-1, ::-1]).astype(np.uint8)

        eef_pos = np.asarray(obs["robot_state"]["eef"]["pos"], dtype=np.float32)
        rpy = Evo1PipelineAdapter.quat2axisangle(
            np.array(obs["robot_state"]["eef"]["quat"], dtype=np.float64, copy=True)
        ).astype(np.float32)
        gripper = np.asarray(obs["robot_state"]["gripper"]["qpos"], dtype=np.float32)

        def _scalar_state(v: float) -> np.ndarray:

            return np.array([[[v]]], dtype=np.float32)

        return {
            "video.image":       front[None, None],
            "video.wrist_image": wrist[None, None],
            "state.x":       _scalar_state(eef_pos[0]),
            "state.y":       _scalar_state(eef_pos[1]),
            "state.z":       _scalar_state(eef_pos[2]),
            "state.roll":    _scalar_state(rpy[0]),
            "state.pitch":   _scalar_state(rpy[1]),
            "state.yaw":     _scalar_state(rpy[2]),
            "state.gripper": gripper.reshape(1, 1, -1).astype(np.float32),

            "task": (obs.get("task_description", ""),),
            "annotation.human.action.task_description": (obs.get("task_description", ""),),
        }

    def parse_action(self, action: np.ndarray) -> np.ndarray:

        action = np.asarray(action[:7], dtype=np.float32).copy()
        action[6] = -1.0 if action[6] > 0.5 else 1.0
        return action


class Gr00tN15PipelineAdapter(Gr00tPipelineAdapter):

    def parse_action(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action[:7], dtype=np.float32).copy()


class OpenVINOPipelineAdapter(LeRobotPipelineAdapter):
    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        parsed_obs = super().parse_observation(obs)
        return {
            "images": {
                "image": parsed_obs["observation.images.image"][None],
                "image2": parsed_obs["observation.images.image2"][None],
            },
            "state": parsed_obs["observation.state"][None],
            "task": [parsed_obs["task"]],
        }
