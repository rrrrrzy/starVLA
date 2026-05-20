#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def matches_name(value: str, query) -> bool:
    if not query:
        return True
    return value == query or query.lower() in value.lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path, help="Merged Calvin LeRobot dataset root.")
    parser.add_argument("--from-task", default=None, help="Exact or substring match for from_subtask.")
    parser.add_argument("--to-task", default=None, help="Exact or substring match for to_subtask.")
    parser.add_argument(
        "--status",
        choices=["success", "fail", "any"],
        default="any",
        help="Filter transition_status.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    transitions_path = args.root / "meta/transitions.jsonl"
    if not transitions_path.exists():
        raise FileNotFoundError(
            f"Missing {transitions_path}. Run scripts/merge_calvin_lerobot_dataset.py first."
        )

    matches = []
    for item in read_jsonl(transitions_path):
        if not matches_name(item["from_subtask"], args.from_task):
            continue
        if not matches_name(item["to_subtask"], args.to_task):
            continue
        if args.status != "any" and item["transition_status"] != args.status:
            continue
        matches.append(item)

    output = args.output
    if output is None:
        name = "transition_manifest"
        if args.from_task:
            name += f"_from_{args.from_task}"
        if args.to_task:
            name += f"_to_{args.to_task}"
        if args.status != "any":
            name += f"_{args.status}"
        output = args.root / "meta" / f"{name}.jsonl"

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for item in matches:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Matched transitions: {len(matches)}")
    print(f"Saved manifest: {output}")
    if matches:
        print("First match:")
        print(json.dumps(matches[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
