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

from typing import Any
import numpy as np

class BasePipelineAdapter:
    def __init__(self, client: Any = None):
        self._client = client

    def reset(self):
        return self._client.reset()

    def get_action(self, obs: dict[str, Any]) -> np.ndarray:
        parsed_obs = self.parse_observation(obs)
        action = self._client.get_action(parsed_obs)
        parsed_action = self.parse_action(action)
        return parsed_action

    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def parse_action(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError
