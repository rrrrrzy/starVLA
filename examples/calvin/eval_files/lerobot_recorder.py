import json
import shutil
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import numpy as np

from examples.calvin.eval_files.video_utils import extract_rgb_frame


FPS = 10
CHUNKS_SIZE = 1000


def normalize_lang(text: str) -> str:
    return text.split("\n")[0].replace("\u2019", "'")


def build_task_index(val_annotations):
    if not hasattr(val_annotations, "keys"):
        return {}, {}
    subtasks = sorted(list(val_annotations.keys()))
    task_to_index = {subtask: i for i, subtask in enumerate(subtasks)}
    index_to_lang = {
        i: normalize_lang(val_annotations[subtask][0])
        for subtask, i in task_to_index.items()
    }
    return task_to_index, index_to_lang


def calvin_state_from_obs(obs: dict) -> np.ndarray:
    robot_obs = np.asarray(obs["robot_obs"], dtype=np.float32).reshape(-1)
    state = np.zeros(8, dtype=np.float32)
    state[: min(6, robot_obs.shape[0])] = robot_obs[:6]
    if robot_obs.shape[0] > 6:
        state[7] = robot_obs[6]
    return state


class CalvinLeRobotSequenceRecorder:
    def __init__(self, root: str, worker_id: int, enabled: bool = True):
        self.root = Path(root) if root else None
        self.worker_id = worker_id
        self.enabled = enabled and self.root is not None
        self.frames = []
        self.segments = []
        self.episode_index = None
        self.round_index = None
        self.initial_state = None
        self.eval_sequence = None
        self.global_offset = 0
        if self.enabled:
            (self.root / "meta").mkdir(parents=True, exist_ok=True)

    def start_sequence(self, episode_index: int, local_sequence_i: int, initial_state: dict, eval_sequence):
        if not self.enabled:
            return
        self.frames = []
        self.segments = []
        self.episode_index = episode_index
        self.round_index = episode_index + 1
        self.initial_state = initial_state
        self.eval_sequence = eval_sequence
        self.global_offset = episode_index * 5 * 360

    def start_segment(self, subtask_i: int, subtask: str, task_index: int, lang_annotation: str):
        if not self.enabled:
            return None
        segment = {
            "worker_id": self.worker_id,
            "episode_index": self.episode_index,
            "round": self.round_index,
            "task_no": subtask_i + 1,
            "subtask": subtask,
            "task_index": task_index,
            "lang_annotation": lang_annotation,
            "start_frame": len(self.frames),
            "end_frame": None,
            "length": 0,
            "status": "running",
            "success": False,
        }
        self.segments.append(segment)
        return segment

    def add_step(self, obs: dict, action: np.ndarray, task_index: int):
        if not self.enabled:
            return
        image = extract_rgb_frame(obs, "rgb_static")
        wrist_image = extract_rgb_frame(obs, "rgb_gripper")
        if image is None or wrist_image is None:
            raise RuntimeError("CALVIN obs is missing rgb_static or rgb_gripper frames.")
        self.frames.append(
            {
                "image": image,
                "wrist_image": wrist_image,
                "state": calvin_state_from_obs(obs),
                "actions": np.asarray(action, dtype=np.float32).reshape(7),
                "task_index": int(task_index),
            }
        )

    def finish_segment(self, segment: Optional[dict], success: bool, status: str):
        if not self.enabled or segment is None:
            return
        segment["end_frame"] = len(self.frames)
        segment["length"] = segment["end_frame"] - segment["start_frame"]
        segment["status"] = status
        segment["success"] = bool(success)

    def add_not_attempted_segment(self, subtask_i: int, subtask: str, task_index: int, lang_annotation: str):
        if not self.enabled:
            return
        start = len(self.frames)
        self.segments.append(
            {
                "worker_id": self.worker_id,
                "episode_index": self.episode_index,
                "round": self.round_index,
                "task_no": subtask_i + 1,
                "subtask": subtask,
                "task_index": task_index,
                "lang_annotation": lang_annotation,
                "start_frame": start,
                "end_frame": start,
                "length": 0,
                "status": "not_attempted",
                "success": False,
            }
        )

    def save_sequence(self, success_count: int):
        if not self.enabled or self.episode_index is None or not self.frames:
            return
        chunk = self.episode_index // CHUNKS_SIZE
        data_path = self.root / f"data/chunk-{chunk:03d}/episode_{self.episode_index:06d}.parquet"
        image_path = self.root / f"videos/chunk-{chunk:03d}/image/episode_{self.episode_index:06d}.mp4"
        wrist_path = self.root / f"videos/chunk-{chunk:03d}/wrist_image/episode_{self.episode_index:06d}.mp4"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        wrist_path.parent.mkdir(parents=True, exist_ok=True)

        self._write_parquet(data_path)
        imageio.mimsave(image_path, [f["image"] for f in self.frames], fps=FPS)
        imageio.mimsave(wrist_path, [f["wrist_image"] for f in self.frames], fps=FPS)

        self._append_jsonl(
            self.root / f"meta/episodes_worker_{self.worker_id}.jsonl",
            {
                "episode_index": self.episode_index,
                "tasks": [s["lang_annotation"] for s in self.segments if s["length"] > 0],
                "length": len(self.frames),
                "worker_id": self.worker_id,
                "round": self.round_index,
                "success_count": success_count,
                "completed": success_count == len(self.eval_sequence),
            },
        )
        for segment in self.segments:
            self._append_jsonl(self.root / f"meta/sequence_segments_worker_{self.worker_id}.jsonl", segment)
        for transition in self._build_transitions():
            self._append_jsonl(self.root / f"meta/transitions_worker_{self.worker_id}.jsonl", transition)
        self._append_tasks_fragment()
        self._write_dataset_stub()

    def _write_parquet(self, path: Path):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Saving Calvin test data requires pyarrow in the Calvin/GPU environment."
            ) from exc

        n = len(self.frames)
        table = pa.table(
            {
                "state": pa.array([f["state"].tolist() for f in self.frames], type=pa.list_(pa.float32(), 8)),
                "actions": pa.array([f["actions"].tolist() for f in self.frames], type=pa.list_(pa.float32(), 7)),
                "timestamp": pa.array(np.arange(n, dtype=np.float32) / FPS, type=pa.float32()),
                "frame_index": pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
                "episode_index": pa.array(np.full(n, self.episode_index, dtype=np.int64), type=pa.int64()),
                "index": pa.array(np.arange(self.global_offset, self.global_offset + n, dtype=np.int64), type=pa.int64()),
                "task_index": pa.array([f["task_index"] for f in self.frames], type=pa.int64()),
            }
        )
        pq.write_table(table, path)

    def _build_transitions(self):
        transitions = []
        attempted = [s for s in self.segments if s["status"] != "not_attempted"]
        for prev, nxt in zip(attempted, attempted[1:]):
            transitions.append(
                {
                    "worker_id": self.worker_id,
                    "episode_index": self.episode_index,
                    "round": self.round_index,
                    "from_task_no": prev["task_no"],
                    "to_task_no": nxt["task_no"],
                    "from_subtask": prev["subtask"],
                    "to_subtask": nxt["subtask"],
                    "from_success": prev["success"],
                    "to_success": nxt["success"],
                    "transition_status": "success" if prev["success"] and nxt["success"] else "fail",
                    "start_frame": prev["start_frame"],
                    "end_frame": nxt["end_frame"],
                }
            )
        return transitions

    def _append_tasks_fragment(self):
        seen = {}
        for segment in self.segments:
            seen[segment["task_index"]] = segment["lang_annotation"]
        for task_index, lang in sorted(seen.items()):
            self._append_jsonl(
                self.root / f"meta/tasks_worker_{self.worker_id}.jsonl",
                {"task_index": task_index, "task": lang},
            )

    def _write_dataset_stub(self):
        meta = self.root / "meta"
        modality_src = Path(__file__).resolve().parents[3] / "examples/calvin/train_files/modality.json"
        modality_dst = meta / "modality.json"
        if modality_src.exists() and not modality_dst.exists():
            shutil.copyfile(modality_src, modality_dst)
        info_stub = meta / "info.stub.json"
        if not info_stub.exists():
            info_stub.write_text(json.dumps(build_info(total_episodes=0, total_frames=0), indent=4), encoding="utf-8")

    @staticmethod
    def _append_jsonl(path: Path, record: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_info(total_episodes: int, total_frames: int, total_tasks: int = 0) -> dict:
    return {
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * 2,
        "total_chunks": (total_episodes + CHUNKS_SIZE - 1) // CHUNKS_SIZE if total_episodes else 0,
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "image": _video_feature(200, 200),
            "wrist_image": _video_feature(84, 84),
            "state": {
                "dtype": "float32",
                "shape": [8],
                "names": ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"],
            },
            "actions": {
                "dtype": "float32",
                "shape": [7],
                "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }


def _video_feature(height: int, width: int) -> dict:
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": FPS,
            "video.channels": 3,
            "has_audio": False,
        },
    }
