"""Run the pre-registered fusion experiment described in eval/benchmark/experiments.md.

    python eval/run_candidates.py dev --model BAAI/bge-small-en-v1.5 --out eval/benchmark/experiments
    python eval/run_candidates.py confirm C3 --model BAAI/bge-small-en-v1.5 --out eval/benchmark/experiments

`dev` runs every candidate on the dev split of the held-out set, and on both v1 gates (the hash gate, and the gate for
--model when it is a real model), applies the dev rule, and selects at most one candidate by the pre-registered
selection rule. `confirm` runs the baseline and the selected candidate on the test split, once: it refuses a second
run unless --allow-rerun is given, and refuses any candidate that `dev` did not select.

Exit status: dev 0 on completion; confirm 0 accepted, 1 rejected; 2 for any error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from compare_runs import meets_dev_rule, meets_test_rule

HERE = Path(__file__).resolve().parent
BENCH = HERE / "benchmark"
CORPUS = BENCH / "corpus"
V1 = BENCH / "queries.json"
V2 = BENCH / "queries-v2.json"

# The pre-registered candidates. Changing this list after experiments have run invalidates the experiment; the same
# list, with the reasoning, is in eval/benchmark/experiments.md and a test keeps the two in step.
CANDIDATES = (
    ("C0", ""),
    ("C1", "rrf_k=20"),
    ("C2", "rrf_k=10"),
    ("C3", "vector_weight=2"),
    ("C4", "keyword_limit=20"),
    ("C5", "keyword_limit=10"),
    ("C6", "drop_stopwords=true"),
    ("C7", "drop_stopwords=true,keyword_limit=20"),
)
SPECS = dict(CANDIDATES)
MARKER = "TEST-SPLIT-USED"


def thresholds_path(model: str) -> Path:
    return BENCH / f"thresholds-{model.split('/')[-1].lower()}.json"


def option_count(spec: str) -> int:
    return len(spec.split(",")) if spec else 0


def select(rows: list[dict]) -> str | None:
    """The pre-registered selection rule.

    Eligible: meets the dev rule and passes every v1 gate (C0 is the baseline and never eligible). Among the eligible,
    the highest dev MRR; ties go to the candidate with fewer changed options, then to the earlier one in the list.
    """
    eligible = [r for r in rows if r["id"] != "C0" and r["dev_rule"] and all(r["gates"].values())]
    if not eligible:
        return None
    return min(eligible, key=lambda r: (-r["mrr"], option_count(r["tuning"]), r["order"]))["id"]


def invoke(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(HERE / script), *args], capture_output=True, text=True)


def run_eval(model: str, queries: Path, tuning: str, *extra: str) -> subprocess.CompletedProcess:
    args = ["--corpus", str(CORPUS), "--queries", str(queries), "--model", model]
    if tuning:
        args += ["--tuning", tuning]
    return invoke("run_eval.py", *args, *extra)


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def chosen_candidates(spec: str | None) -> list[tuple[str, str]] | str:
    if not spec:
        return list(CANDIDATES)
    wanted = [part.strip() for part in spec.split(",") if part.strip()]
    unknown = [w for w in wanted if w not in SPECS]
    if unknown:
        return f"unknown candidate(s) {', '.join(unknown)}; known: {', '.join(SPECS)}"
    return [(i, s) for i, s in CANDIDATES if i == "C0" or i in wanted]


def dev(args: argparse.Namespace) -> int:
    candidates = chosen_candidates(args.candidates)
    if isinstance(candidates, str):
        return fail(candidates)
    gate_models = [args.model] + (["hash"] if args.model != "hash" else [])
    for gate_model in gate_models:
        if not thresholds_path(gate_model).exists():
            return fail(f"missing {thresholds_path(gate_model)}; derive it with eval/derive_thresholds.py first")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, files = [], {}
    for order, (cid, spec) in enumerate(candidates):
        path = out / f"dev-{cid}.json"
        run = run_eval(args.model, V2, spec, "--split", "dev", "--json", str(path))
        if run.returncode != 0:
            return fail(f"{cid} dev run failed:\n{run.stderr}")
        gates = {}
        for gate_model in gate_models:
            gate = run_eval(gate_model, V1, spec, "--thresholds", str(thresholds_path(gate_model)))
            if gate.returncode not in (0, 1):
                return fail(f"{cid} gate run for {gate_model} failed:\n{gate.stderr}")
            gates[gate_model] = gate.returncode == 0
        files[cid] = path
        rows.append({"id": cid, "tuning": spec, "order": order, "gates": gates})

    records = {cid: json.loads(path.read_text(encoding="utf-8")) for cid, path in files.items()}
    for row in rows:
        record = records[row["id"]]
        row["mrr"] = record["results"]["hybrid"]["mrr"]
        row["dev_rule"] = row["id"] != "C0" and meets_dev_rule(records["C0"], record)
    selected = select(rows)

    table = invoke("compare_runs.py", *(str(files[cid]) for cid, _ in candidates))
    print(table.stdout if table.returncode == 0 else table.stderr)
    print("v1 gates (a candidate must pass all): " + ", ".join(gate_models))
    for row in rows:
        marks = "  ".join(f"{m}: {'pass' if ok else 'FAIL'}" for m, ok in row["gates"].items())
        print(f"  {row['id']:<4}{row['tuning'] or '(defaults)':<40}{marks}")
    print()
    print(f"Selected for confirmation on the test split: {selected}" if selected
          else "No candidate qualifies: keep the shipped defaults.")  # fmt: skip
    summary = {"model": args.model, "selected": selected, "candidates": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


def confirm(args: argparse.Namespace) -> int:
    if args.candidate not in SPECS or args.candidate == "C0":
        return fail(f"{args.candidate!r} is not a candidate; choose one of {', '.join(i for i in SPECS if i != 'C0')}")
    out = Path(args.out)
    summary_path = out / "summary.json"
    if not summary_path.exists():
        return fail(f"{summary_path} not found: run the dev step first")
    selected = json.loads(summary_path.read_text(encoding="utf-8")).get("selected")
    if selected != args.candidate:
        return fail(f"the dev step selected {selected!r}, not {args.candidate!r}; only that one may be confirmed")
    marker = out / MARKER
    if marker.exists() and not args.allow_rerun:
        return fail(f"the test split was already used ({marker.read_text(encoding='utf-8').strip()}); "
                    "re-running it would turn it into a tuning set")  # fmt: skip

    files = {}
    for cid in ("C0", args.candidate):
        path = out / f"test-{cid}.json"
        run = run_eval(args.model, V2, SPECS[cid], "--split", "test", "--json", str(path))
        if run.returncode != 0:
            return fail(f"{cid} test run failed:\n{run.stderr}")
        files[cid] = path
    table = invoke("compare_runs.py", "--rule", "test", str(files["C0"]), str(files[args.candidate]))
    print(table.stdout if table.returncode == 0 else table.stderr)
    base, cand = (json.loads(files[c].read_text(encoding="utf-8")) for c in ("C0", args.candidate))
    accepted = meets_test_rule(base, cand)
    verdict = f"{args.candidate} {'ACCEPTED' if accepted else 'REJECTED'} on the test split"
    marker.write_text(verdict + "\n", encoding="utf-8")
    print(verdict + ("" if accepted else ": do not adjust and re-run; record the negative result"))
    return 0 if accepted else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("dev", "confirm"):
        p = sub.add_parser(name)
        if name == "confirm":
            p.add_argument("candidate")
            p.add_argument("--allow-rerun", action="store_true")
        else:
            p.add_argument("--candidates", help="comma-separated subset, e.g. C3,C6 (the baseline C0 always runs)")
        p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
        p.add_argument("--out", default=str(BENCH / "experiments"))
    args = parser.parse_args()
    return dev(args) if args.command == "dev" else confirm(args)


if __name__ == "__main__":
    raise SystemExit(main())
