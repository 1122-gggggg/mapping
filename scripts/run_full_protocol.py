from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="runs/protocol")
    args = parser.parse_args()
    commands = [
        ["update-map", "validate-config", "--config", args.config, "--require-paths"],
        ["update-map", "audit", "--config", args.config, "--output", f"{args.output}/audit"],
        ["update-map", "run-protocol", "--config", args.config, "--output", args.output],
    ]
    for command in commands:
        print("+", " ".join(command), flush=True)
        completed = subprocess.run(command)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
