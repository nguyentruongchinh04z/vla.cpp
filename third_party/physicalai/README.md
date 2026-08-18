# PhysicalAI OpenVINO Runtime

This directory contains the OpenVINO runtime for PhysicalAI. It provides a Python interface to interact with the OpenVINO inference engine, allowing users to perform inference on models optimized for Intel hardware.

## Prerequisites
- Ubuntu 22.04 / 24.04
- Docker
- Intel CPU or GPU accelerated hardware

## Build Server
To build the OpenVINO runtime, run the following command in the vla.cpp directory:

```bash
bash ./third_party/physicalai/server/build_server.sh \
    --model-dir <path_to_model_directory> \
    --device-type <CPU|GPU> \
    --host-port 5555
```
- For the first time, it will take a while to build the Docker image and compile the model. Subsequent runs will be faster as the compiled model will be cached.
- When the server is ready, you should see the following message in the terminal:
    ```
    Server is ready and listening on tcp://0.0.0.0:5555
    ```

## Usage
To use the OpenVINO runtime, you can import the `OpenVINOClient` class from the `eval.utils.clients.openvino` module and create an instance of it. You can then call the `get_action` method to perform inference on your observations.
- Setup the virtual environment for the client:
    ```bash
    uv venv --python=3.12

    source .venv/bin/activate

    uv pip install -e .[openvino]
    ```

- Then, you can use the following code snippet to create an instance of the `OpenVINOInferenceClient` and perform inference:
    ```python
    from eval.utils.clients.openvino import OpenVINOInferenceClient

    client = OpenVINOInferenceClient(
        host="0.0.0.0", 
        port=5555, 
        n_action_steps=1,
        api_token=None, 
    )

    client.get_action({
        "images": {
            "<image_key1>": np.array(...),
            "<image_key2>": np.array(...),
        },
        "state": np.array(...),
        "task": ["<task_name>"],
    })
    ```

## Benchmarking
We provide a benchmarking script to evaluate the performance of the OpenVINO runtime on LIBERO simulation, and open-loop benchmark. To run the benchmark, execute the following instructions:

Before running the benchmark, make sure to start the OpenVINO server as described in the [Build Server](#build-server) section above.

### LIBERO Simulation Benchmark
- Setup the environment for the simulation:
    ```bash
    bash eval/sim/libero/setup_libero.sh
    ```
- Then, run the following script to benchmark the OpenVINO runtime on LIBERO simulation:
    ```bash
    ./eval/.venvs/libero/bin/python ./eval/scripts/run_libero_client_openvino.py \
        --task libero_object \
        --task-id 0 \
        --n-episodes 10 \
        --host 0.0.0.0 \
        --port 5555 \
        --n-action-steps 1
    ```

### Open-loop Benchmark
- Before benchmarking, make sure your dataset is prepared as LeRobot v3 format dataset.
- Activate the virtual environment for the client:
    ```bash
    source .venv/bin/activate
    ```
- Then, run the following script to benchmark the OpenVINO runtime on open-loop benchmark:
    ```bash
    python ./eval/scripts/open_loop_benchmark_openvino.py \
        --dataset-root <path_to_dataset_directory> \
        --host 0.0.0.0 \
        --port 5555 \
        --episode-ids "0,1,2" \
        --n-obs-steps 1 \
        --n-action-steps 1
    ```
