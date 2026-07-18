import time
import numpy as np
from physicalai.inference import InferenceModel

import io
from dataclasses import dataclass

from typing import Any, Callable, Dict

import os
import zmq
import msgpack
import numpy as np
import openvino as ov


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


class OpenVINOInferenceServer:

    def __init__(
        self,
        policy: InferenceModel,
        host: str = "*",
        port: int = 5555,
        api_token: str = None
    ):
        self.policy = policy
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}
        self.api_token = api_token

        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)

        self.register_endpoint("get_action", self._get_action)

    def _kill_server(self):

        self.running = False

    def _handle_ping(self) -> dict:

        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):

        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def _validate_token(self, request: dict) -> bool:

        if self.api_token is None:
            return True
        return request.get("api_token") == self.api_token

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")
        while self.running:
            try:
                message = self.socket.recv()
                request = MsgSerializer.from_bytes(message)

                if not self._validate_token(request):
                    self.socket.send(
                        MsgSerializer.to_bytes({"error": "Unauthorized: Invalid API token"})
                    )
                    continue

                endpoint = request.get("endpoint", "get_action")

                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]
                result = (
                    handler.handler(request.get("data", {}))
                    if handler.requires_input
                    else handler.handler()
                )
                self.socket.send(MsgSerializer.to_bytes(result))
            except Exception as e:
                print(f"Error in server: {e}")
                import traceback

                print(traceback.format_exc())
                self.socket.send(MsgSerializer.to_bytes({"error": str(e)}))

    @staticmethod
    def start_server(policy: InferenceModel, host: str = "*", port: int = 5555, api_token: str = None):
        server = OpenVINOInferenceServer(
            policy=policy,
            host=host,
            port=port,
            api_token=api_token
        )
        server.run()

    def _get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        t0 = time.time()
        action_chunk = self.policy.predict_action_chunk(observation)
        t1 = time.time()
        print(f"[get_action] - action_shape: {action_chunk.shape}, inference_time: {round(1000 * (t1 - t0))}ms")
        return action_chunk


if __name__ == "__main__":
    core = ov.Core()
    print("Available devices:", core.available_devices)
    for d in core.available_devices:
        try:
            print(d, "=>", core.get_property(d, "FULL_DEVICE_NAME"))
        except Exception as e:
            print(d, "=>", e)

    device_type = os.environ.get("DEVICE_TYPE", "CPU").strip().upper()
    valid_device_types = {"CPU", "GPU", "HETERO:GPU,CPU"}
    assert device_type in valid_device_types, (
        f"Invalid DEVICE_TYPE: {device_type}. Supported values: CPU, GPU, HETERO:GPU,CPU"
    )
    device = "HETERO:GPU,CPU" if device_type == "GPU" else device_type

    print("Starting OpenVINO Inference Server...")
    t0 = time.time()
    policy = InferenceModel("./weights", device=device)
    t1 = time.time()
    print(f"Model Compile time: {t1 - t0:.2f} seconds")
    print(f"Backend: {policy.backend}")
    print(f"OpenVINO target device: {policy.device}")
    OpenVINOInferenceServer.start_server(
        policy=policy, host="*", port=5555
    )
