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

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_ROOT))


import numpy as np
import zmq

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except Exception as exc:
    raise RuntimeError(
        "Failed to import LeRobotDataset. Install the LeRobot dataset extras in "
        "the Python environment used for this script, for example:\n"
        "  uv pip install 'lerobot[dataset]'\n"
        "or install the missing runtime packages such as datasets, pyarrow, "
        f"pandas, and av. Original import error: {type(exc).__name__}: {exc}"
    ) from exc

from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, ACTION  # noqa: E402

from utils.clients.openvino import OpenVINOInferenceClient  # noqa: E402


###########################################################################
######################## INITIALIZATION HELPERS ###########################
###########################################################################
def _resolve_existing_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
    return parsed


def _parse_episode_ids(spec: str | None) -> list[int] | None:
    if spec is None:
        return None
    spec = spec.strip()
    if not spec or spec.lower() in {"all", "none"}:
        return None

    ids: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, stop = int(left), int(right)
            if start < 0 or stop < 0 or start > stop:
                raise argparse.ArgumentTypeError(
                    f"invalid episode range {part!r}; expected START-END with 0 <= START <= END"
                )
            ids.update(range(start, stop + 1))
        else:
            ep_id = int(part)
            if ep_id < 0:
                raise argparse.ArgumentTypeError(
                    f"invalid episode id {ep_id}; expected a non-negative integer"
                )
            ids.add(ep_id)

    if not ids:
        raise argparse.ArgumentTypeError(f"no episode ids parsed from {spec!r}")
    return sorted(ids)


def _load_dataset_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"LeRobot dataset info not found: {info_path}")
    return json.loads(info_path.read_text())


def _resolve_delta_timestamp(
    dataset_info: dict[str, Any], n_obs_steps: int, n_action_steps: int
) -> float | None:
    """Resolve the delta timestamp for the dataset based on the number of observation and action steps."""
    if n_obs_steps <= 0 or n_action_steps <= 0:
        raise ValueError(
            f"n_obs_steps and n_action_steps must be positive integers, got {n_obs_steps} and {n_action_steps}"
        )
    fps = dataset_info.get("fps")
    if fps is None or fps <= 0:
        return None

    features = dataset_info.get("features", {})
    assert features, "LeRobot dataset info missing 'features' key"

    image_timestamps = {
        key: [i / fps for i in range(0, -n_obs_steps, -1)] 
        for key in features.keys() if key.startswith(OBS_IMAGES)
    }

    assert OBS_STATE in features, f"LeRobot dataset info missing '{OBS_STATE}' feature"
    state_timestamps = [i / fps for i in range(0, -n_obs_steps, -1)]

    assert ACTION in features, f"LeRobot dataset info missing '{ACTION}' feature"
    action_timestamps = [i / fps for i in range(0, n_action_steps)]

    return {
        **image_timestamps,
        OBS_STATE: state_timestamps,
        ACTION: action_timestamps,
    }

########################################################
############## FORMAT HELPERS ##########################
########################################################
def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy") and callable(value.numpy):
        return value.numpy()
    return np.asarray(value)


def _scalar(value: Any) -> Any:
    arr = _to_numpy(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return value


def _image_to_chw_float(value: Any, key: str) -> np.ndarray:
    arr = _to_numpy(value)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"{key}: expected a 3D image tensor/array, got shape {arr.shape}")

    if arr.shape[0] in (1, 3, 4):
        chw = arr[:3]
    elif arr.shape[-1] in (1, 3, 4):
        chw = np.transpose(arr[..., :3], (2, 0, 1))
    else:
        raise ValueError(
            f"{key}: expected CHW or HWC RGB image, got shape {arr.shape}"
        )

    if chw.shape[0] == 1:
        chw = np.repeat(chw, 3, axis=0)

    chw = np.ascontiguousarray(chw, dtype=np.float32)
    if chw.size and float(np.nanmax(chw)) > 1.5:
        chw = chw / 255.0
    return chw


def _vector_float(value: Any, key: str) -> np.ndarray:
    arr = _to_numpy(value)
    if arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr.reshape(-1)
    else:
        arr = arr.reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{key}: expected a non-empty vector")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _task_to_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _task_to_str(value[0])
    if hasattr(value, "item") and callable(value.item):
        try:
            return str(value.item())
        except Exception:
            pass
    return str(value)


def _dataset_item_to_openvino_observation(
    item: dict[str, Any]
) -> dict[str, Any]:
    images = {}
    for  key, value in item.items():
        if key.startswith(OBS_IMAGES) and not key.endswith("_is_pad"):
            images[key] = _image_to_chw_float(value, key)[None]
    state = _vector_float(item[OBS_STATE], OBS_STATE)[None]
    task = _task_to_str(item.get("task", ""))

    return {
        "images": images,
        "state": state,
        "task": [task],
    }


def _percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    idx = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[idx]


def _summary_ms(xs: list[float]) -> dict[str, float | int | None]:
    return _summary_values(xs, digits=3)


def _summary_values(xs: list[float], *, digits: int) -> dict[str, float | int | None]:
    if not xs:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "n": len(xs),
        "mean": round(statistics.fmean(xs), digits),
        "median": round(statistics.median(xs), digits),
        "p95": round(_percentile(xs, 0.95), digits),
        "p99": round(_percentile(xs, 0.99), digits),
        "min": round(min(xs), digits),
        "max": round(max(xs), digits),
    }


def _safe_hz_from_ms(ms: float | None) -> float | None:
    if ms is None or ms <= 0.0:
        return None
    return round(1000.0 / ms, 3)


def _configure_client_timeouts(client: OpenVINOInferenceClient, timeout_ms: int) -> None:
    client.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    client.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    client.socket.setsockopt(zmq.LINGER, 0)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Open-loop benchmark for the PhysicalAI OpenVINO SmolVLA server on a "
            "LeRobot v3 LIBERO dataset. The script loads recorded frames with "
            "LeRobotDataset, sends observations to an already-running OpenVINO "
            "server, and compares predicted actions to dataset actions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--repo-id", type=str, default=None,
        help="LeRobot repo_id label used with the local --dataset-root."
    )
    ap.add_argument("--host", type=str, default="localhost")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--timeout-ms", type=_positive_int, default=120_000)
    ap.add_argument("--api-token", type=str, default=None)
    ap.add_argument("--skip-ping", action="store_true",
        help="Do not ping the server before the benchmark loop."
    )
    ap.add_argument(
        "--episode-ids", "--episodes", dest="episode_ids", type=_parse_episode_ids, default=None,
        help="Episode ids to evaluate. Accepts comma/range syntax, e.g. '0,3,7-9'. Default: all."
    )
    ap.add_argument(
        "--max-samples", type=_positive_int, default=None,
        help="Maximum measured samples after warmup. Default: all selected frames."
    )
    ap.add_argument(
        "--warmup-samples", type=_non_negative_int, default=5,
        help="Samples to run before collecting metrics."
    )
    ap.add_argument("--stride", type=_positive_int, default=1,
        help="Evaluate every Nth dataset frame. Chunk replay is reset across skipped frames.")
    ap.add_argument(
        "--n-obs-steps", type=_positive_int, default=1,
        help="How many observation steps to send to the server for each predicted chunk. Default: checkpoint metadata or 1."
    )
    ap.add_argument(
        "--n-action-steps", type=_positive_int, default=1,
        help="How many actions to replay from each predicted chunk. Default: checkpoint metadata or 1."
    )
    ap.add_argument(
        "--output-json", type=Path,
        default=Path("outputs/open_loop_benchmark_openvino.json"),
    )
    ap.add_argument(
        "--per-sample-csv", type=Path, default=None,
        help="Optional CSV path for one row per measured sample."
    )
    return ap


def main() -> int:
    args = _build_parser().parse_args()

    dataset_root = _resolve_existing_path(args.dataset_root)
    dataset_info = _load_dataset_info(dataset_root)

    total_episodes = int(dataset_info.get("total_episodes", 0))
    if args.episode_ids is not None:
        bad = [ep for ep in args.episode_ids if ep >= total_episodes]
        if bad:
            raise ValueError(
                f"episode id(s) out of range for {dataset_root}: {bad}; "
                f"valid range is 0..{total_episodes - 1}"
            )

    n_action_steps = args.n_action_steps

    print(f"loading LeRobot dataset: {dataset_root}", flush=True)
    if args.episode_ids is None:
        print("episode ids: all", flush=True)
    else:
        print(f"episode ids: {args.episode_ids}", flush=True)
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=dataset_root,
        episodes=args.episode_ids,
        download_videos=False,
        delta_timestamps=_resolve_delta_timestamp(
            dataset_info, args.n_obs_steps, args.n_action_steps
        )
    )

    if len(dataset) == 0:
        raise RuntimeError(f"selected dataset is empty: {dataset_root}")

    print(
        f"connecting to OpenVINO server tcp://{args.host}:{args.port} "
        f"(n_action_steps={n_action_steps})",
        flush=True,
    )
    client = OpenVINOInferenceClient(
        host=args.host,
        port=args.port,
        timeout_ms=args.timeout_ms,
        api_token=args.api_token,
        n_action_steps=n_action_steps,
    )
    _configure_client_timeouts(client, args.timeout_ms)
    if not args.skip_ping:
        if not client.ping():
            _configure_client_timeouts(client, args.timeout_ms)
            raise RuntimeError(
                f"OpenVINO server ping failed for tcp://{args.host}:{args.port}"
            )
        _configure_client_timeouts(client, args.timeout_ms)

    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    csv_file = None
    csv_writer = None
    if args.per_sample_csv is not None:
        per_sample_csv = Path(args.per_sample_csv).expanduser()
        per_sample_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = per_sample_csv.open("w", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "sample_id",
                "dataset_position",
                "dataset_index",
                "episode_index",
                "frame_index",
                "task_index",
                "latency_ms",
                "is_inference_call",
                "mse",
                "mae",
            ],
        )
        csv_writer.writeheader()

    positions = range(0, len(dataset), args.stride)
    warmup_remaining = args.warmup_samples
    n_measured = 0
    n_inference_calls = 0
    n_queue_hits = 0
    n_resets_for_episode_or_gap = 0
    n_action_dim_mismatches = 0

    latency_ms: list[float] = []
    inference_time_ms: list[float] = []
    per_sample_mse: list[float] = []
    per_sample_mae: list[float] = []
    sum_sq: np.ndarray | None = None
    sum_abs: np.ndarray | None = None
    action_dim = None

    last_episode = None
    last_dataset_index = None
    measured_start_s = None
    measured_end_s = None

    try:
        for dataset_position in positions:
            if args.max_samples is not None and n_measured >= args.max_samples:
                break

            item = dataset[dataset_position]
            episode_index = int(_scalar(item["episode_index"]))
            dataset_index = int(_scalar(item["index"]))

            discontinuity = (
                last_episode is None
                or episode_index != last_episode
                or dataset_index != last_dataset_index + 1
            )
            if discontinuity:
                client.reset()
                if last_episode is not None:
                    n_resets_for_episode_or_gap += 1

            observation = _dataset_item_to_openvino_observation(item)
            target = _vector_float(item[ACTION], ACTION)

            is_warmup = warmup_remaining > 0
            if not is_warmup and measured_start_s is None:
                measured_start_s = time.perf_counter()
            will_query = len(client._action_queue) == 0
            t0 = time.perf_counter()
            pred = _vector_float(client.get_action_chunk(observation), "predicted_action")
            elapsed_ms = 1000.0 * (time.perf_counter() - t0)

            if is_warmup:
                warmup_remaining -= 1
                if warmup_remaining == 0:
                    client.reset()
                    last_episode = None
                    last_dataset_index = None
                else:
                    last_episode = episode_index
                    last_dataset_index = dataset_index
                continue

            if pred.size != target.size:
                n_action_dim_mismatches += 1
            dim = min(pred.size, target.size)
            pred = pred[:dim]
            target = target[:dim]
            err = pred - target
            sq = err * err
            ab = np.abs(err)

            if sum_sq is None:
                action_dim = dim
                sum_sq = np.zeros(dim, dtype=np.float64)
                sum_abs = np.zeros(dim, dtype=np.float64)
            elif dim != action_dim:
                raise ValueError(
                    f"action dimension changed from {action_dim} to {dim} at dataset index {dataset_index}"
                )

            sum_sq += sq
            sum_abs += ab
            sample_mse = float(np.mean(sq))
            sample_mae = float(np.mean(ab))
            per_sample_mse.append(sample_mse)
            per_sample_mae.append(sample_mae)
            latency_ms.append(elapsed_ms)
            if will_query:
                n_inference_calls += 1
                inference_time_ms.append(elapsed_ms)
            else:
                n_queue_hits += 1

            if csv_writer is not None:
                csv_writer.writerow({
                    "sample_id": n_measured,
                    "dataset_position": dataset_position,
                    "dataset_index": dataset_index,
                    "episode_index": episode_index,
                    "frame_index": int(_scalar(item["frame_index"])),
                    "task_index": int(_scalar(item["task_index"])),
                    "latency_ms": round(elapsed_ms, 6),
                    "is_inference_call": int(will_query),
                    "mse": round(sample_mse, 9),
                    "mae": round(sample_mae, 9),
                })

            n_measured += 1
            measured_end_s = time.perf_counter()
            last_episode = episode_index
            last_dataset_index = dataset_index
    finally:
        if csv_file is not None:
            csv_file.close()

    if n_measured == 0 or sum_sq is None or sum_abs is None or action_dim is None:
        raise RuntimeError(
            "no measured samples were collected; reduce --warmup-samples, add valid "
            "--episode-ids, or increase --max-samples"
        )

    eval_wall_s = (
        max(0.0, measured_end_s - measured_start_s)
        if measured_start_s is not None and measured_end_s is not None
        else 0.0
    )
    per_dim_mse = sum_sq / n_measured
    per_dim_mae = sum_abs / n_measured
    latency_summary = _summary_ms(latency_ms)
    inference_summary = _summary_ms(inference_time_ms)
    mean_latency_ms = statistics.fmean(latency_ms) if latency_ms else None
    mean_inference_ms = (
        statistics.fmean(inference_time_ms) if inference_time_ms else None
    )

    stats = {
        "dataset": {
            "root": str(dataset_root),
            "repo_id": args.repo_id,
            "codebase_version": dataset_info.get("codebase_version"),
            "fps": dataset_info.get("fps"),
            "total_episodes": dataset_info.get("total_episodes"),
            "total_frames": dataset_info.get("total_frames"),
            "selected_episode_ids": args.episode_ids,
            "selected_frames_loaded": len(dataset),
            "stride": args.stride,
        },
        "server": {
            "host": args.host,
            "port": args.port,
            "timeout_ms": args.timeout_ms,
        },
        "counters": {
            "measured_samples": n_measured,
            "inference_calls": n_inference_calls,
            "queue_hits": n_queue_hits,
            "resets_for_episode_or_gap": n_resets_for_episode_or_gap,
            "action_dim_mismatches_truncated": n_action_dim_mismatches,
        },
        "error": {
            "action_dim": action_dim,
            "mse": round(float(np.mean(per_dim_mse)), 9),
            "mae": round(float(np.mean(per_dim_mae)), 9),
            "per_dim_mse": [round(float(v), 9) for v in per_dim_mse.tolist()],
            "per_dim_mae": [round(float(v), 9) for v in per_dim_mae.tolist()],
            "per_sample_mse": _summary_values(per_sample_mse, digits=9),
            "per_sample_mae": _summary_values(per_sample_mae, digits=9),
        },
        "timing": {
            "eval_wall_s": round(eval_wall_s, 6),
            "latency_ms": latency_summary,
            "client_observed_inference_time_ms": inference_summary,
            "frequency_hz": {
                "wall_sample_hz": round(n_measured / eval_wall_s, 3) if eval_wall_s > 0 else None,
                "latency_mean_hz": _safe_hz_from_ms(mean_latency_ms),
                "inference_mean_hz": _safe_hz_from_ms(mean_inference_ms),
            },
        },
    }

    output_json.write_text(json.dumps(stats, indent=2) + "\n")

    print("\n=== Open-loop OpenVINO benchmark ===")
    print(f"dataset         : {dataset_root}")
    print(f"episodes        : {args.episode_ids if args.episode_ids is not None else 'all'}")
    print(f"samples         : {n_measured} measured, {args.warmup_samples} warmup")
    print(f"action dim      : {action_dim}")
    print(f"MSE / MAE       : {stats['error']['mse']} / {stats['error']['mae']}")
    print(
        "latency ms     : "
        f"mean={latency_summary['mean']} median={latency_summary['median']} "
        f"p95={latency_summary['p95']}"
    )
    print(
        "inference ms   : "
        f"mean={inference_summary['mean']} median={inference_summary['median']} "
        f"p95={inference_summary['p95']} calls={n_inference_calls}"
    )
    print(f"frequency hz   : {stats['timing']['frequency_hz']}")
    print(f"wrote          : {output_json}")
    if args.per_sample_csv is not None:
        print(f"per-sample csv : {Path(args.per_sample_csv).expanduser()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
