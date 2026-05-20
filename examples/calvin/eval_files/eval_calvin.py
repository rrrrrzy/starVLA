"""
Calvin Multi-Step Evaluation Script

Based on RoboFlamingo's evaluation protocol:
https://github.com/RoboFlamingo/RoboFlamingo/blob/main/robot_flamingo/eval/eval_utils.py

Evaluates a policy server on Calvin's long-horizon multi-task benchmark.
Measures success rate on chains of 1-5 consecutive tasks.

Usage:
    python examples/calvin/eval_calvin.py \
        --args.host 0.0.0.0 \
        --args.port 8000 \
        --args.dataset_path /path/to/calvin/task_D_D \
        --args.num_sequences 1000
"""

import copy
import dataclasses
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import tyro

# # Add Calvin to path
# CALVIN_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "calvin"
# sys.path.insert(0, str(CALVIN_ROOT))
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
    print_and_save,
)
from omegaconf import OmegaConf
from termcolor import colored
from tqdm import tqdm

from deployment.model_server.tools import image_tools
from examples.LIBERO.eval_files.model2libero_interface import ModelClient
from examples.calvin.eval_files.lerobot_recorder import (
    CalvinLeRobotSequenceRecorder,
    build_task_index,
    normalize_lang,
)
from examples.calvin.eval_files.video_utils import RolloutVideoRecorder

# from calvin_env.envs.play_table_env import get_env

# Set OpenGL platform for headless rendering
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MUJOCO_GL"] = "osmesa"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Video recording env vars
CALVIN_SAVE_VIDEO = os.environ.get("CALVIN_SAVE_VIDEO", "0") == "1"
CALVIN_VIDEO_CAMERA = os.environ.get("CALVIN_VIDEO_CAMERA", "rgb_static")
CALVIN_VIDEO_DIR = os.environ.get("CALVIN_VIDEO_DIR", "")
CALVIN_PROGRESS_EVERY = int(os.environ.get("CALVIN_PROGRESS_EVERY", "25"))

EP_LEN = 360  # Max steps per task


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    pretrained_path: str = ""
    unnorm_key: str = ""

    #################################################################################################################
    # Calvin environment-specific parameters
    #################################################################################################################
    dataset_path: str = "/path/to/calvin/task_D_D"  # Path to Calvin dataset
    calvin_config_path: str = "/path/to/calvin/calvin_models/conf"
    eval_sequences_path: str = "/path/to/calvin/eval_sequences.json"
    num_sequences: int = 1000  # Number of evaluation sequences
    num_workers: int = 1  # For future multi-process support
    seed: int = 0
    create_plan_tsne: bool = False

    #################################################################################################################
    # Evaluation settings
    #################################################################################################################
    debug: bool = False  # Save debug videos
    eval_log_dir: str = "tmp/calvin/eval_logs"  # Path to save evaluation logs and videos
    reset: bool = False  # If True, reset robot state between tasks (easier)
    diverse_inst: bool = False  # Use diverse instructions (zero-shot generalization)
    worker_id: int = -1  # Parallel Calvin worker id, used for action trace records
    sequence_offset: int = 0  # Global index offset of this worker's first eval sequence
    action_trace_path: str = ""  # JSONL path for per-sequence task-level action traces
    lerobot_data_dir: str = ""  # LeRobot v2.1 output root for sequential Calvin test rollouts


class CalvinPolicyClient:
    """Wrapper around websocket client with Calvin-specific preprocessing."""

    def __init__(
        self,
        host: str,
        port: int,
        resize_size: int = 224,
        replan_steps: int = 5,
        pretrained_path: str = "",
        unnorm_key: str = "",
    ):
        # New ModelClient API: checkpoint is loaded by the StarVLA policy server.
        # The client only connects to the websocket server and sends observations.
        self.client = ModelClient(
            unnorm_key=(unnorm_key or None),
            policy_setup="franka",
            host=host,
            port=port,
        )
        self.resize_size = resize_size
        self.replan_steps = replan_steps
        self.step_count = 0

    def reset(self):
        """Reset action plan buffer."""
        self.step_count = 0

    def step(self, obs: dict, lang_annotation: str) -> np.ndarray:
        """
        Query policy for action given observation and language instruction.

        Args:
            obs: Calvin observation dict with keys:
                - rgb_obs: dict with 'rgb_static' (200x200x3) and 'rgb_gripper' (84x84x3)
                - robot_obs: (15,) proprioceptive state [ee_pos(3), ee_ori(3), gripper(2), joint_pos(7)]
            lang_annotation: Natural language task description
            get_action: If True, query model for new action chunk

        Returns:
            action: (7,) array [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        # Preprocess images
        rgb_static = obs["rgb_obs"]["rgb_static"]  # (200, 200, 3) uint8
        rgb_gripper = obs["rgb_obs"]["rgb_gripper"]  # (84, 84, 3) uint8

        # Resize and pad images
        image = image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb_static, self.resize_size, self.resize_size))
        wrist_image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(rgb_gripper, self.resize_size, self.resize_size)
        )

        # Prepare input for policy server (aligned with eval_libero)
        example = {
            "image": [image, wrist_image],
            "lang": lang_annotation,
        }

        # Query model
        model_output = self.client.step(example=example, step=self.step_count)
        raw_action = model_output["raw_action"]
        world_vector = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
        rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
        open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)

        action = np.concatenate([world_vector, rotation_delta, open_gripper], axis=0).astype(np.float32)
        self.step_count += 1
        return action


def make_env(dataset_path: str):
    """Initialize Calvin environment without tactile sensor (to avoid OpenGL issues)."""
    val_folder = Path(dataset_path) / "validation"

    # Load config and disable tactile sensor to avoid pyrender/OpenGL conflicts
    from omegaconf import OmegaConf

    config_path = val_folder / ".hydra" / "merged_config.yaml"
    cfg = OmegaConf.load(config_path)

    # Remove tactile sensor from camera list if it exists
    if hasattr(cfg.env, "cameras") and "tactile" in cfg.env.cameras:
        # Create a new camera dict without tactile
        new_cameras = OmegaConf.create({k: v for k, v in cfg.env.cameras.items() if k != "tactile"})
        cfg.env.cameras = new_cameras

    # Initialize environment with modified config
    import hydra

    env = hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)

    return env


def load_lang_task(dataset_path: str) -> dict:
    """Load language annotations and task oracle for Calvin validation set."""
    conf_dir = Path(dataset_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    return val_annotations, task_oracle


def write_action_trace(action_trace_path: str, record: dict):
    if not action_trace_path:
        return
    path = Path(action_trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_not_attempted_task(subtask_i: int, subtask: str, val_annotations, sequence_i: int, diverse_inst: bool):
    if diverse_inst:
        lang_annotation = val_annotations[sequence_i][subtask_i]
    else:
        lang_annotation = val_annotations[subtask][0]
    lang_annotation = lang_annotation.split("\n")[0].replace("\u2019", "'")
    return {
        "task_no": subtask_i + 1,
        "subtask": subtask,
        "lang_annotation": lang_annotation,
        "status": "not_attempted",
        "success": False,
        "steps": 0,
        "elapsed_sec": 0.0,
    }


def get_lang_annotation(subtask_i: int, subtask: str, val_annotations, sequence_i: int, diverse_inst: bool):
    if diverse_inst:
        return normalize_lang(val_annotations[sequence_i][subtask_i])
    return normalize_lang(val_annotations[subtask][0])


def get_task_index(subtask_i: int, subtask: str, task_to_index: dict, sequence_i: int, diverse_inst: bool):
    if diverse_inst:
        return sequence_i * 5 + subtask_i
    return task_to_index[subtask]


def evaluate_policy_ddp(
    policy,
    env,
    epoch,
    calvin_conf_path,
    eval_sequences_path,
    num_sequences,
    eval_log_dir=None,
    debug=False,
    create_plan_tsne=False,
    reset=False,
    diverse_inst=False,
    worker_id=-1,
    sequence_offset=0,
    action_trace_path="",
    lerobot_data_dir="",
):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    # val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    if diverse_inst:
        with open("/mnt/bn/robotics/lxh/robot-flamingo/lang_annotation_cache.json", "r") as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    task_to_index, _ = build_task_index(val_annotations)

    eval_log_dir = get_log_dir(eval_log_dir)
    with open(eval_sequences_path, "r") as f:
        eval_sequences = json.load(f)
    # device_num = int(torch.distributed.get_world_size())
    # device_id = torch.distributed.get_rank()
    # assert num_sequences % device_num == 0
    # interval_len = int(num_sequences // device_num)
    # eval_sequences = eval_sequences[device_id*interval_len:min((device_id+1)*interval_len, num_sequences)]
    results = []
    plans = defaultdict(list)
    local_sequence_i = 0

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        sequence_i = sequence_offset + local_sequence_i
        sequence_recorder = CalvinLeRobotSequenceRecorder(
            lerobot_data_dir,
            worker_id=worker_id,
            enabled=bool(lerobot_data_dir),
        )
        sequence_recorder.start_sequence(sequence_i, local_sequence_i, initial_state, eval_sequence)
        result, task_records = evaluate_sequence(
            env,
            policy,
            task_oracle,
            initial_state,
            eval_sequence,
            val_annotations,
            task_to_index,
            plans,
            debug,
            eval_log_dir,
            sequence_i,
            reset=reset,
            diverse_inst=diverse_inst,
            sequence_recorder=sequence_recorder,
        )
        results.append(result)
        sequence_recorder.save_sequence(result)
        write_action_trace(
            action_trace_path,
            {
                "worker_id": worker_id,
                "round": sequence_i + 1,
                "sequence_index_global": sequence_i,
                "sequence_index_local": local_sequence_i,
                "initial_state": initial_state,
                "tasks": task_records,
                "success_count": result,
                "completed": result == len(eval_sequence),
            },
        )
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )
        local_sequence_i += 1

    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    def extract_iter_from_tqdm(tqdm_iter):
        return [_ for _ in tqdm_iter]

    # if create_plan_tsne:
    #     create_tsne(plans, eval_log_dir, epoch)

    eval_sequences = extract_iter_from_tqdm(eval_sequences)

    print_and_save(results, eval_sequences, eval_log_dir, epoch)

    return results


def evaluate_sequence(
    env,
    policy,
    task_checker,
    initial_state,
    eval_sequence,
    val_annotations,
    task_to_index,
    plans,
    debug,
    eval_log_dir="",
    sequence_i=-1,
    reset=False,
    diverse_inst=False,
    sequence_recorder=None,
):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    task_records = []
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        if reset:
            rollout_record = rollout(
                env,
                policy,
                task_checker,
                subtask,
                val_annotations,
                plans,
                debug,
                eval_log_dir,
                subtask_i,
                sequence_i,
                robot_obs=robot_obs,
                scene_obs=scene_obs,
                diverse_inst=diverse_inst,
                task_to_index=task_to_index,
                sequence_recorder=sequence_recorder,
            )
        else:
            rollout_record = rollout(
                env,
                policy,
                task_checker,
                subtask,
                val_annotations,
                plans,
                debug,
                eval_log_dir,
                subtask_i,
                sequence_i,
                diverse_inst=diverse_inst,
                task_to_index=task_to_index,
                sequence_recorder=sequence_recorder,
            )
        task_records.append(rollout_record)
        success = rollout_record["success"]
        if success:
            success_counter += 1
        else:
            for future_subtask_i, future_subtask in enumerate(eval_sequence[subtask_i + 1 :], start=subtask_i + 1):
                task_records.append(
                    build_not_attempted_task(
                        future_subtask_i,
                        future_subtask,
                        val_annotations,
                        sequence_i,
                        diverse_inst,
                    )
                )
                if sequence_recorder is not None:
                    lang_annotation = get_lang_annotation(
                        future_subtask_i,
                        future_subtask,
                        val_annotations,
                        sequence_i,
                        diverse_inst,
                    )
                    task_index = get_task_index(
                        future_subtask_i,
                        future_subtask,
                        task_to_index,
                        sequence_i,
                        diverse_inst,
                    )
                    sequence_recorder.add_not_attempted_segment(
                        future_subtask_i,
                        future_subtask,
                        task_index,
                        lang_annotation,
                    )
            return success_counter, task_records
    return success_counter, task_records


def rollout(
    env,
    policy,
    task_oracle,
    subtask,
    val_annotations,
    plans,
    debug,
    eval_log_dir="",
    subtask_i=-1,
    sequence_i=-1,
    robot_obs=None,
    scene_obs=None,
    diverse_inst=False,
    task_to_index=None,
    sequence_recorder=None,
):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    if robot_obs is not None and scene_obs is not None:
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = get_lang_annotation(subtask_i, subtask, val_annotations, sequence_i, diverse_inst)
    task_index = get_task_index(subtask_i, subtask, task_to_index or {}, sequence_i, diverse_inst)
    segment = None
    if sequence_recorder is not None:
        segment = sequence_recorder.start_segment(subtask_i, subtask, task_index, lang_annotation)
    policy.reset()
    start_info = env.get_info()
    rollout_start = time.time()

    # Determine whether to record video for this rollout
    record_video = CALVIN_SAVE_VIDEO or debug
    if CALVIN_VIDEO_DIR:
        video_dir = CALVIN_VIDEO_DIR
    elif eval_log_dir:
        video_dir = os.path.join(eval_log_dir, "videos")
    else:
        video_dir = None
    recorder = RolloutVideoRecorder(
        save_dir=video_dir,
        fps=30,
        camera=CALVIN_VIDEO_CAMERA,
        enabled=record_video and video_dir is not None,
    )
    recorder.add_obs(obs)

    for step in range(EP_LEN):
        if CALVIN_PROGRESS_EVERY > 0 and step % CALVIN_PROGRESS_EVERY == 0:
            print(
                f"[rollout] seq={sequence_i} sub={subtask_i} task={subtask} step={step} begin",
                flush=True,
            )

        t_policy = time.time()
        action = policy.step(obs, lang_annotation)
        policy_dt = time.time() - t_policy

        # Ensure action is writable (Calvin env modifies it in-place)
        if not action.flags.writeable:
            action = np.array(action, copy=True)
        action[-1] = 1 if action[-1] > 0 else -1
        if sequence_recorder is not None:
            sequence_recorder.add_step(obs, action, task_index)

        t_env = time.time()
        obs, _, _, current_info = env.step(action)
        env_dt = time.time() - t_env
        if CALVIN_PROGRESS_EVERY > 0 and step % CALVIN_PROGRESS_EVERY == 0:
            print(
                f"[rollout] seq={sequence_i} sub={subtask_i} task={subtask} step={step} "
                f"policy_dt={policy_dt:.3f}s env_dt={env_dt:.3f}s",
                flush=True,
            )
        recorder.add_obs(obs)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(policy, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
            recorder.save(f"seq{sequence_i}_sub{subtask_i}_{subtask}_succ")
            if sequence_recorder is not None:
                sequence_recorder.finish_segment(segment, success=True, status="success")
            return {
                "task_no": subtask_i + 1,
                "subtask": subtask,
                "lang_annotation": lang_annotation,
                "status": "success",
                "success": True,
                "steps": step + 1,
                "elapsed_sec": round(time.time() - rollout_start, 3),
            }
    if debug:
        print(colored("fail", "red"), end=" ")
    recorder.save(f"seq{sequence_i}_sub{subtask_i}_{subtask}_fail")
    if sequence_recorder is not None:
        sequence_recorder.finish_segment(segment, success=False, status="fail")
    return {
        "task_no": subtask_i + 1,
        "subtask": subtask,
        "lang_annotation": lang_annotation,
        "status": "fail",
        "success": False,
        "steps": EP_LEN,
        "elapsed_sec": round(time.time() - rollout_start, 3),
    }


def main(args: Args):
    # args = tyro.cli(Args)
    print(
        f"[calvin-config] CALVIN_SAVE_VIDEO={CALVIN_SAVE_VIDEO} "
        f"CALVIN_VIDEO_DIR={CALVIN_VIDEO_DIR} "
        f"CALVIN_VIDEO_CAMERA={CALVIN_VIDEO_CAMERA} "
        f"CALVIN_PROGRESS_EVERY={CALVIN_PROGRESS_EVERY}",
        flush=True,
    )

    policy = CalvinPolicyClient(
        args.host,
        args.port,
        args.resize_size,
        args.replan_steps,
        pretrained_path=args.pretrained_path,
        unnorm_key=args.unnorm_key,
    )
    env = make_env(args.dataset_path)

    evaluate_policy_ddp(
        policy,
        env,
        0,
        args.calvin_config_path,
        args.eval_sequences_path,
        args.num_sequences,
        args.eval_log_dir,
        args.debug,
        args.create_plan_tsne,
        args.reset,
        args.diverse_inst,
        args.worker_id,
        args.sequence_offset,
        args.action_trace_path,
        args.lerobot_data_dir,
    )


if __name__ == "__main__":
    tyro.cli(main)
