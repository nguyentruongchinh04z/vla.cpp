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

import io
from dataclasses import dataclass
from typing import Any, Callable, Dict

import zmq
import msgpack
import numpy as np
from collections import deque


class MsgSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        return msgpack.packb(
            data,
            default=MsgSerializer.encode_custom_classes,
            use_bin_type=True,
        )

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        return msgpack.unpackb(
            data,
            object_hook=MsgSerializer.decode_custom_classes,
            raw=False,
        )

    @staticmethod
    def decode_custom_classes(obj):
        if "__ndarray_class__" in obj:
            obj = np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class OpenVINOInferenceClient:

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str = None,
        n_action_steps: int = 1,
    ):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

        self._action_queue = deque(maxlen=n_action_steps)
        self.n_action_steps = n_action_steps

    def _init_socket(self):

        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()
            return False

    def kill_server(self):

        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> dict:

        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        response = MsgSerializer.from_bytes(message)

        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def __del__(self):

        self.socket.close()
        self.context.term()
    
    def reset(self) -> None:
        self._action_queue.clear()

    def get_action_chunk(self, observations: Dict[str, Any]) -> np.ndarray:
        action_chunk = self.call_endpoint("get_action", observations)
        return action_chunk[:self.n_action_steps]

    def get_action(self, observations: Dict[str, Any]) -> np.ndarray:
        if not self._action_queue:
            action_chunk = self.call_endpoint("get_action", observations)
            self._action_queue.extend(action_chunk[:self.n_action_steps])
        return self._action_queue.popleft()
