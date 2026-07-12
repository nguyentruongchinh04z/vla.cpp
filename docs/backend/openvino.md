# OpenVINO Backend for vla.cpp

OpenVINO is an open-source toolkit for optimizing and deploying high-performance AI inference, specifically designed for Intel hardware, including CPUs, GPUs, and NPUs, in the cloud, on-premises, and on the edge. OpenVINO backend for vla.cpp enables hardware-accelerated inference on Intel® CPUs, GPUs, and NPUs while remaining compatible with the existing GGUF model ecosystem. The backend translates GGML compute graphs into OpenVINO graphs and leverages graph compilation, kernel fusion, and device-specific optimizations to improve inference performance on supported Intel hardware.

## Supported Devices
OpenVINO backend supports the following hardware:
- Intel CPUs
- Intel GPUs (integrated and discrete)
- Intel NPUs

## Prerequisites

- Linux system (22.04 or 24.04) with Intel hardware (CPU, GPU, or NPU)
- For Intel GPU or NPU Usage: Install the appropriate hardware drivers for your Intel GPU or NPU. For detailed instructions, see: [Additional Configurations for Hardware Acceleration](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md)

## Install OpenVINO Runtime
### For Manual Installation
- For manual installation, follow the guide to install OpenVINO Runtime from an archive file: [Install OpenVINO Runtime from an Archive File](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-archive-linux.html)
- For automatic installation, run the following command:
    ```bash
    bash backend/openvino/install_runtime.sh
    ```
- Verify OpenVINO is initialized properly: 
    ```bash
    echo $OpenVINO_DIR
    ```

## Build

Initialize the OpenVINO environment, configure the OpenVINO GGML backend, and
build the server:

```bash
source /opt/intel/openvino/setupvars.sh
cmake -B build/ReleaseOV -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENVINO=ON
cmake --build build/ReleaseOV --parallel
```

## Run

Set `GGML_OPENVINO_DEVICE` to the actual target name. Do not enter the
documentation placeholder `<DEVICE_TYPE>` literally: in a shell, angle brackets
are interpreted as input redirection.

```bash
GGML_OPENVINO_DEVICE=GPU \
GGML_OPENVINO_STATEFUL_EXECUTION=1 \
./build/ReleaseOV/vla-server ./weights/smolvla-libero.gguf
```

Use `CPU`, `GPU`, or `NPU` according to the devices exposed by the installed
OpenVINO runtime. At startup, two diagnostics identify the selection:

```text
OpenVINO: using device GPU
vla: backend = OPENVINO (requested device GPU)
```

The first line is authoritative for the resolved OpenVINO device. If the
requested device is unavailable, the OpenVINO backend emits a warning, falls
back to CPU, and reports `OpenVINO: using device CPU`. The `vla:` line separately
confirms that GGML is using its OpenVINO backend rather than its native CPU
backend.


