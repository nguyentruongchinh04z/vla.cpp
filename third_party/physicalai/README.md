# PhysicalAI OpenVINO Server

This directory provides a Dockerized OpenVINO inference server for running the
SmolVLA LIBERO policy through the PhysicalAI Python runtime.

This is separate from the native C++ `vla-server` path:

```text
LIBERO simulator
  -> eval/scripts/run_libero_client_openvino.py
  -> ZeroMQ + msgpack
  -> third_party/physicalai/src/run_server.py
  -> physicalai.inference.InferenceModel
  -> OpenVINO IR model
```

The PhysicalAI server uses ZeroMQ with msgpack serialization. It is not
protocol-compatible with the C++ `vla-server`, which uses ZeroMQ with protobuf.

## Contents

| Path | Purpose |
| --- | --- |
| `Dockerfile` | Ubuntu 24.04 image containing OpenVINO, PhysicalAI, and Intel CPU/GPU/NPU userspace drivers. |
| `build_server.sh` | Builds the image, mounts the model, passes through available Intel devices, and starts the server. |
| `src/run_server.py` | Loads the PhysicalAI model and serves inference requests on port 5555. |
| `inference_model_to_cpp_backend.drawio` | Diagram of the PhysicalAI-to-OpenVINO runtime flow. |

`src/convert_smolvla_lrb_to_ov.py` is currently empty and is not used by this
workflow. Use an already exported OpenVINO policy package.

## Supported configuration

- Linux `x86_64`/`amd64` host
- Docker with BuildKit support
- OpenVINO model package exported for PhysicalAI
- Supported `DEVICE_TYPE` values:
  - `CPU`
  - `GPU`
  - `HETERO:GPU,CPU`

When `GPU` is selected, `run_server.py` compiles the model as
`HETERO:GPU,CPU`, allowing OpenVINO to place unsupported operations on the CPU.

The Docker image installs an Intel NPU driver and the launcher passes
`/dev/accel/accel0` through when available. However, `run_server.py` does not
currently accept `NPU` as a `DEVICE_TYPE`.

## 1. Install prerequisites

Run the commands in this guide from the repository root.

### Docker

Install Docker Engine, the Docker CLI, and Buildx using the
[official Docker installation guide](https://docs.docker.com/engine/install/).

Verify the installation:

```bash
docker version
docker buildx version
docker run --rm hello-world
```

### uv

The repository uses `uv` to download the model and create the LIBERO Python
environment.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

The server-side Python packages and OpenVINO runtime are installed inside the
Docker image. You do not need to build `vla.cpp` or install PhysicalAI on the
host for this workflow.

## 2. Download the OpenVINO model

Download the PhysicalAI/OpenVINO SmolVLA LIBERO package:

```bash
mkdir -p weights

uvx --from huggingface-hub hf download \
    OpenVINO/smolvla-libero-fp16-ov \
    --local-dir weights/smolvla-libero-fp16-ov
```

Verify the required files:

```bash
test -f weights/smolvla-libero-fp16-ov/manifest.json
test -f weights/smolvla-libero-fp16-ov/smolvla.xml
test -f weights/smolvla-libero-fp16-ov/smolvla.bin
test -f weights/smolvla-libero-fp16-ov/tokenizer.xml
test -f weights/smolvla-libero-fp16-ov/tokenizer.bin
```

The launcher mounts the model directory read-only at `/workspace/weights`
inside the container. Pass the directory itself, not `manifest.json` or
`smolvla.xml`.

## 3. Start the inference server

The launcher accepts:

```text
build_server.sh MODEL_DIR [HOST_PORT=5555] [DEVICE_TYPE=CPU]
```

### CPU

In terminal 1:

```bash
bash third_party/physicalai/build_server.sh \
    "$(realpath weights/smolvla-libero-fp16-ov)" \
    5555 \
    CPU
```

On the first run, the script builds `physicalai_server:latest`. Later runs
reuse the local image.

A successful startup prints the available OpenVINO devices, compile time,
backend, target device, and then:

```text
Server is ready and listening on tcp://0.0.0.0:5555
```

Stop the server with `Ctrl-C`. The container uses `--rm`, so Docker removes it
after exit.

### Intel GPU

The host must expose `/dev/dri`, and Docker must have permission to access the
render device.

```bash
bash third_party/physicalai/build_server.sh \
    "$(realpath weights/smolvla-libero-fp16-ov)" \
    5555 \
    GPU
```

Confirm that the startup output lists an OpenVINO `GPU` device before assuming
that inference is using the accelerator.

### Environment variables

The launcher also accepts environment variables. Positional arguments take
precedence.

```bash
MODEL_DIR="$(realpath weights/smolvla-libero-fp16-ov)" \
HOST_PORT=5555 \
DEVICE_TYPE=CPU \
DOCKER_IMAGE=physicalai_server:latest \
bash third_party/physicalai/build_server.sh
```

The server always listens on port 5555 inside the container. `HOST_PORT`
changes the published host port.

The launcher publishes the port on all host interfaces and does not configure
an API token. Do not expose it to an untrusted network.

## 4. Install LIBERO

In terminal 2, from the repository root:

```bash
bash eval/sim/libero/setup_libero.sh
```

The script:

- clones or reuses LIBERO under `eval/sim/libero/LIBERO/`;
- creates a Python 3.10 environment at
  `eval/sim/libero/libero_uv/.venv/`;
- installs the pinned simulator, PyTorch, Gymnasium, ZeroMQ, and msgpack
  dependencies; and
- initializes `~/.libero/config.yaml` non-interactively.

Important: rerunning the setup removes the existing
`eval/sim/libero/libero_uv/` environment and resets `~/.libero/`.

## 5. Run a LIBERO evaluation

Keep the inference server running in terminal 1. In terminal 2:

```bash
source eval/sim/libero/libero_uv/.venv/bin/activate

MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
python eval/scripts/run_libero_client_openvino.py \
    --host localhost \
    --port 5555 \
    --task libero_object \
    --task-id 0 \
    --n-episodes 1 \
    --n-action-steps 8 \
    --output-dir outputs/physicalai
```

`--n-action-steps` controls how many actions are replayed from a predicted
chunk before the client requests another chunk. The client default is 1. The
example uses 8 to reduce the number of server calls. The bundled model manifest
trims output to a 50-action chunk.

Results are written to:

```text
outputs/physicalai/libero_object/
```

The directory contains episode videos and `summary.txt`, including the success
rate, skipped episodes, and average inference time.

## 6. Benchmark report

### 6.1 System configuration

| Component | Configuration |
| --- | --- |
| CPU | Intel Core Ultra 9 H285 (Arrow Lake architecture) |
| GPU | NVIDIA RTX Pro 1000 Blackwell, 8 GB (Blackwell architecture) |
| Operating system | Ubuntu 24.04 |
| CUDA | 12.8 |
| NVIDIA driver | 580.159.03 |

### 6.2 Evaluation setup

The measurements below were collected with the following configuration:

- LIBERO suite: `libero_object`
- Task: `task_0`
- Episodes: 10
- Action replay: `n_action_steps=8`
- VRAM baseline: 13.44 MiB
- RAM baseline: 11,550 MiB

### 6.3 Results

| Backend | Server inference time (ms) | Client average inference time (ms) | Success rate | Peak VRAM (MiB) | Peak RAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native CPU | 2,050 (vision: 1,320; head: 730; other: 0.5) | 318.1 | 100% (10/10) | 1,276 | 13,500 |
| CUDA | 127 (vision: 74; head: 53; other: 0.7) | 46 | 80% (8/10) | 13.44 | 14,000 |
| OpenVINO CPU | 1,600 | 230 | 100% (10/10) | 13.44 | 15,500 |
| OpenVINO iGPU | 300 | 72 | 100% (10/10) | 13.44 | 16,500 |

These values are the observed results for this specific machine, task, episode
count, and action-replay configuration. They should not be treated as general
performance guarantees for other hardware or LIBERO tasks.

### Useful client options

| Option | Description |
| --- | --- |
| `--task` | LIBERO suite: `libero_10`, `libero_spatial`, `libero_object`, `libero_goal`, or `libero_90`. |
| `--task-id` | Task variation within the selected suite. |
| `--n-episodes` | Number of episodes to evaluate. |
| `--view-mode` | `multi-view` or `single-view` output video. |
| `--host` | Inference-server hostname or IP address. |
| `--port` | Published server port. |
| `--n-action-steps` | Number of actions consumed before requesting another chunk. |

For a remote server, replace `--host localhost` with the server hostname or IP
and allow the selected `HOST_PORT` through the firewall. The current transport
does not provide encryption or authentication.

## Optional server ping

With the LIBERO environment installed and the server running:

```bash
timeout 10s env PYTHONPATH=eval \
eval/sim/libero/libero_uv/.venv/bin/python - <<'PY'
from utils.clients.openvino import OpenVINOInferenceClient

client = OpenVINOInferenceClient(host="localhost", port=5555)
print("server ready:", client.ping())
PY
```

Expected output:

```text
server ready: True
```

## Rebuild the Docker image

`build_server.sh` only builds the image when `DOCKER_IMAGE` does not already
exist. After changing the Dockerfile or server source, rebuild explicitly:

```bash
docker build \
    -t physicalai_server:latest \
    -f third_party/physicalai/Dockerfile \
    third_party/physicalai
```

Alternatively, build and run a separate development image:

```bash
DOCKER_IMAGE=physicalai_server:dev \
bash third_party/physicalai/build_server.sh \
    "$(realpath weights/smolvla-libero-fp16-ov)" \
    5555 \
    CPU
```

## Troubleshooting

### Invalid or empty model bind mount

Confirm that `MODEL_DIR` is an existing absolute directory:

```bash
MODEL_DIR="$(realpath weights/smolvla-libero-fp16-ov)"
test -d "$MODEL_DIR"
test -f "$MODEL_DIR/manifest.json"
```

If the shell runs inside a development container, VM, or remote workspace while
Docker uses a host-side daemon, the daemon may not see the shell's absolute
path. Pass the corresponding host-visible path as `MODEL_DIR`; otherwise Docker
can mount a different empty directory.

### Model loading fails

The container expects the complete PhysicalAI policy package at
`/workspace/weights`. Check that every artifact referenced by `manifest.json`
exists in the mounted directory.

### GPU is not available

Check the host devices:

```bash
ls -l /dev/dri
ls -l /dev/dri/renderD128
```

If `/dev/dri` is absent, the launcher cannot pass the GPU to Docker. If the
device exists but access is denied, ensure Docker and the current user can
access the render device. The launcher adds the numeric group that owns
`/dev/dri/renderD128` when it exists.

### Invalid `DEVICE_TYPE`

Use `CPU`, `GPU`, or `HETERO:GPU,CPU`. Unsupported values such as `CUDA`,
`AUTO`, and `NPU` are rejected by the current server.

### Client connects to the wrong server

Do not run the native C++ `vla-server` and this PhysicalAI server on the same
host port. `run_libero_client_openvino.py` must connect to `run_server.py`; the
two servers use different message formats.

### Headless LIBERO rendering fails

Use `MUJOCO_GL=egl` and a valid `CUDA_VISIBLE_DEVICES` index as shown above. An
empty `CUDA_VISIBLE_DEVICES` can break robosuite's EGL renderer.

## Implementation notes

- `run_server.py` uses a synchronous ZeroMQ `REP` socket and processes one
  request at a time.
- NumPy arrays are serialized as in-memory `.npy` data inside msgpack.
- The LIBERO adapter converts channel-last `uint8` camera images to
  channel-first `float32` values in `[0, 1]`.
- Robot end-effector quaternion state is converted to axis-angle before being
  sent to the server.
- PhysicalAI reads `manifest.json`, detects the OpenVINO artifact, compiles the
  model on the selected device, and applies the manifest's preprocessing and
  postprocessing pipeline.
- The mounted model directory is read-only. Videos and result summaries are
  generated by the host-side LIBERO client.

## References

- [PhysicalAI runtime](https://github.com/openvinotoolkit/physicalai)
- [OpenVINO SmolVLA LIBERO model](https://huggingface.co/OpenVINO/smolvla-libero-fp16-ov)
- [OpenVINO documentation](https://docs.openvino.ai/)
