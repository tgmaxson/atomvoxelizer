from __future__ import annotations

import subprocess
import sys


def main():
    commands = [
        [
            sys.executable,
            "benchmarks/benchmark_backends.py",
            "--workload",
            "zeolite",
            "--framework",
            "BEA",
            "--resolution",
            "0.25",
            "--repeats",
            "3",
            "--zeolite-scaling",
            "--plot",
            "zeolite_scaling.png",
        ],
        [
            sys.executable,
            "benchmarks/benchmark_backends.py",
            "--workload",
            "wulff",
            "--wulff-size",
            "1000",
            "--resolution",
            "0.35",
            "--repeats",
            "3",
        ],
    ]

    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
