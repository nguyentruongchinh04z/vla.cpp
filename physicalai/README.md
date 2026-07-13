# Running VLA with OpenVINO Backend through PhysicalAI server

# Build Server with Docker
```bash
MODEL_DIR=<path_to_model_directory> DEVICE_TYPE=<device_type> bash ./physicalai/build_server.sh
```

# Run Evaluation
```bash
source eval/sim/libero/libero_uv/.venv/bin/activate

python eval/client/run_libero_eval_openvino.py \
    --task libero_object --task-id 0 --n-episodes 1 \
    --output-dir ./outputs/libero_outputs
```