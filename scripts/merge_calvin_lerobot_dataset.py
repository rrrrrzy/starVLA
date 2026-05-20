#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

from examples.calvin.eval_files.lerobot_recorder import build_info


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_fragments(meta_dir: Path, pattern: str):
    records = []
    for path in sorted(meta_dir.glob(pattern)):
        records.extend(read_jsonl(path))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path, help="Calvin LeRobot output root.")
    parser.add_argument(
        "--modality-source",
        default="examples/calvin/train_files/modality.json",
        type=Path,
    )
    args = parser.parse_args()

    root = args.root
    meta_dir = root / "meta"
    if not meta_dir.exists():
        raise FileNotFoundError(f"Missing meta dir: {meta_dir}")

    episodes = collect_fragments(meta_dir, "episodes_worker_*.jsonl")
    segments = collect_fragments(meta_dir, "sequence_segments_worker_*.jsonl")
    transitions = collect_fragments(meta_dir, "transitions_worker_*.jsonl")
    task_records = collect_fragments(meta_dir, "tasks_worker_*.jsonl")

    episodes.sort(key=lambda x: x["episode_index"])
    segments.sort(key=lambda x: (x["episode_index"], x["task_no"]))
    transitions.sort(key=lambda x: (x["episode_index"], x["from_task_no"], x["to_task_no"]))

    tasks = {}
    for item in task_records:
        tasks[int(item["task_index"])] = item["task"]
    task_output = [
        {"task_index": task_index, "task": task}
        for task_index, task in sorted(tasks.items())
    ]

    total_frames = sum(int(ep["length"]) for ep in episodes)
    info = build_info(
        total_episodes=len(episodes),
        total_frames=total_frames,
        total_tasks=len(task_output),
    )

    write_jsonl(meta_dir / "episodes.jsonl", episodes)
    write_jsonl(meta_dir / "tasks.jsonl", task_output)
    write_jsonl(meta_dir / "sequence_segments.jsonl", segments)
    write_jsonl(meta_dir / "transitions.jsonl", transitions)
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8")

    modality_dst = meta_dir / "modality.json"
    if not modality_dst.exists() and args.modality_source.exists():
        shutil.copyfile(args.modality_source, modality_dst)

    print(f"Saved LeRobot metadata under: {meta_dir}")
    print(f"episodes={len(episodes)} frames={total_frames} tasks={len(task_output)} transitions={len(transitions)}")


if __name__ == "__main__":
    main()
