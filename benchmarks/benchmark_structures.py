from __future__ import annotations

import subprocess
import sys


def main():
    commands = [
        [
            sys.executable,
            "benchmarks/benchmark_backends.py",
            "--workloads",
            "zeolite",
            "nanoparticle",
            "surface",
            "--resolution",
            "1.1",
            "--repeats",
            "3",
            "--plot",
            "mask_generation_scaling.png",
        ],
    ]

    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
