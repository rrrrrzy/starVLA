#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


METRIC_RE = re.compile(
    r"1/5\s*:\s*([0-9.]+)%\s*\|\s*"
    r"2/5\s*:\s*([0-9.]+)%\s*\|\s*"
    r"3/5\s*:\s*([0-9.]+)%\s*\|\s*"
    r"4/5\s*:\s*([0-9.]+)%\s*\|\s*"
    r"5/5\s*:\s*([0-9.]+)%"
)


def parse_last_metrics(log_path: Path):
    text = log_path.read_text(errors="ignore")
    matches = list(METRIC_RE.finditer(text))
    if not matches:
        return None

    last = matches[-1]
    values = [float(x) for x in last.groups()]
    return {
        "1/5": values[0],
        "2/5": values[1],
        "3/5": values[2],
        "4/5": values[3],
        "5/5": values[4],
    }


def find_latest_log_dir(repo: Path) -> Path | None:
    log_base = repo / "log"
    if not log_base.exists():
        return None
    candidates = sorted(
        [d for d in log_base.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: d.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def merge_action_traces(log_dir: Path, output_path: Path):
    trace_dir = log_dir / "action_traces"
    trace_paths = sorted(trace_dir.glob("worker_*.jsonl"))
    if not trace_paths:
        return None

    records = []
    bad_lines = []
    for trace_path in trace_paths:
        with trace_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    bad_lines.append(
                        {
                            "path": str(trace_path),
                            "line": line_no,
                            "error": str(exc),
                        }
                    )

    records.sort(key=lambda x: x.get("sequence_index_global", x.get("round", 0)))
    payload = {
        "total_records": len(records),
        "records": records,
        "bad_lines": bad_lines,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return {
        "path": output_path,
        "total_records": len(records),
        "bad_lines": len(bad_lines),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default="/inspire/qb-ilm2/project/26summer-camp-10/26220056",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="starVLA repo path, default is <base>/starVLA",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="calvin log dir, e.g. <repo>/log/<timestamp>/calvin",
    )
    parser.add_argument(
        "--split-dir",
        default=None,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    args = parser.parse_args()

    base = Path(args.base)
    repo = Path(args.repo) if args.repo else base / "starVLA"

    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        latest = find_latest_log_dir(repo)
        if latest is None:
            print("ERROR: No log directories found under <repo>/log/")
            return
        log_dir = latest / "calvin"
        print(f"[INFO] Using latest log dir: {latest}")

    if not log_dir.exists():
        print(f"ERROR: log dir not found: {log_dir}")
        return

    split_dir = (
        Path(args.split_dir)
        if args.split_dir
        else base / "runs" / "calvin_parallel" / "eval_splits"
    )

    if args.output:
        output_path = Path(args.output)
    else:
        merge_dir = log_dir.parent / "merge"
        merge_dir.mkdir(parents=True, exist_ok=True)
        output_path = merge_dir / "merged_results.json"

    workers = []
    total_sequences = 0
    weighted = {
        "1/5": 0.0,
        "2/5": 0.0,
        "3/5": 0.0,
        "4/5": 0.0,
        "5/5": 0.0,
    }
    weighted_avg_len = 0.0

    for worker_id in range(args.num_workers):
        count_file = split_dir / f"eval_sequences_worker_{worker_id}.count"
        log_candidates = sorted(log_dir.glob(f"calvin_eval_worker{worker_id}_*.log"))

        if not count_file.exists():
            workers.append({
                "worker": worker_id,
                "status": "missing_count_file",
                "count_file": str(count_file),
            })
            continue

        num_sequences = int(count_file.read_text().strip())

        if not log_candidates:
            workers.append({
                "worker": worker_id,
                "status": "missing_log_file",
                "num_sequences": num_sequences,
            })
            continue

        log_path = log_candidates[-1]
        metrics = parse_last_metrics(log_path)

        if metrics is None:
            workers.append({
                "worker": worker_id,
                "status": "metrics_not_found_or_not_finished",
                "num_sequences": num_sequences,
                "log": str(log_path),
            })
            continue

        workers.append({
            "worker": worker_id,
            "status": "ok",
            "num_sequences": num_sequences,
            "log": str(log_path),
            "metrics_percent": metrics,
        })

        total_sequences += num_sequences
        for k, v in metrics.items():
            weighted[k] += v * num_sequences
        avg_len = sum(metrics.values()) / 100.0
        weighted_avg_len += avg_len * num_sequences

    if total_sequences > 0:
        merged = {k: v / total_sequences for k, v in weighted.items()}
        avg_seq_len = weighted_avg_len / total_sequences
    else:
        merged = None
        avg_seq_len = None

    result = {
        "total_sequences": total_sequences,
        "merged_metrics_percent": merged,
        "avg_sequence_length": avg_seq_len,
        "workers": workers,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("=" * 80)
    print("CALVIN merged results")
    print("=" * 80)

    if merged is None:
        print("No completed worker metrics found.")
    else:
        print(f"Total sequences: {total_sequences}")
        print(f"Avg sequence length: {avg_seq_len:.4f}")
        print(
            " | ".join(
                f"{k}: {v:.2f}%"
                for k, v in merged.items()
            )
        )

    print()
    print(f"Saved to: {output_path}")
    print()

    trace_output = output_path.parent / "action_traces.json"
    trace_summary = merge_action_traces(log_dir, trace_output)
    if trace_summary is not None:
        print(
            f"Merged action traces: {trace_summary['total_records']} records -> "
            f"{trace_summary['path']}"
        )
        if trace_summary["bad_lines"]:
            print(f"Action trace bad lines: {trace_summary['bad_lines']}")
        print()

    if merged is not None:
        print("--- Copy to Excel ---")
        print("\t".join(f"{v:.2f}" for v in merged.values()) + f"\t{avg_seq_len:.4f}")

    incomplete = [w for w in workers if w["status"] != "ok"]
    if incomplete:
        print("Incomplete / missing workers:")
        for w in incomplete:
            print(f"  worker {w['worker']}: {w['status']}")


if __name__ == "__main__":
    main()
