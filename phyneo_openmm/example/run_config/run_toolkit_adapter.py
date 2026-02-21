#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from phyneo_openmm.toolkit.protocol import create_protocol_from_config


def main():
    parser = argparse.ArgumentParser(description="Run protocol via toolkit.create_protocol_from_config")
    parser.add_argument("--config", default="./config_packmol_bulk_ec_transport.json", help="Path to protocol config JSON")
    parser.add_argument("--skip-post", action="store_true", help="Skip post_process()")
    args = parser.parse_args()

    runner = create_protocol_from_config(str(Path(args.config).resolve()))
    run_out = runner.run_protocol()
    post_out = {} if args.skip_post else runner.post_process()

    result = {
        "runner_class": type(runner).__name__,
        "run": run_out,
        "post": post_out,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
