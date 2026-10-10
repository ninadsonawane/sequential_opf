"""Run the documented research stages in separate, reproducible processes."""

from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(script, *args):
    command = [sys.executable, str(ROOT / script), *args]
    print("\nRUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--stage", choices=("all", "1", "2", "3"), default="all")
    args = parser.parse_args()
    if args.stage in ("all", "1"):
        options = [
            "--workers",
            "4",
            "--repeats",
            "7",
            "--output",
            "experiments/01_numba_benchmark/results",
        ]
        if args.mode == "quick":
            options += ["--smoke"]
        run("experiments/01_numba_benchmark/benchmark.py", *options)
    if args.stage in ("all", "2"):
        run("experiments/02_sequential_pf_25bus/run_validation.py")
        if args.mode == "full":
            run(
                "experiments/02_sequential_pf_25bus/run_benchmark.py", "--repeats", "21"
            )
        run("experiments/02_sequential_pf_25bus/make_report.py")
    if args.stage in ("all", "3"):
        if args.mode == "quick":
            run("experiments/03_sequential_opf/smoke.py")
        else:
            for name in ("run_study.py", "run_tradeoff.py"):
                run("experiments/03_sequential_opf/" + name)
            run(
                "experiments/03_sequential_opf/benchmark_opf.py",
                "--opf-repeats",
                "3",
                "--pf-repeats",
                "7",
            )
            run("experiments/03_sequential_opf/make_report.py")
            run("experiments/03_sequential_opf/verify_results.py")
    print("\nAll requested stages completed successfully.", flush=True)


if __name__ == "__main__":
    main()
