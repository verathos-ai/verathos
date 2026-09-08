"""Explicit local relay when validator and proxy use different service users."""
import argparse
import logging
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurons.mesh_repin_ipc import relay_mesh_repin_requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--source-uid", required=True, type=int)
    parser.add_argument("--validator-shared-state", required=True)
    parser.add_argument("--interval-seconds", type=float, default=0,
                        help="zero runs one batch; otherwise repeat (minimum 2 seconds)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    while True:
        count = relay_mesh_repin_requests(
            args.source_dir, source_uid=args.source_uid,
            shared_state_path=args.validator_shared_state,
        )
        if count:
            logging.info("Relayed %d private mesh re-pin hint(s)", count)
        if args.interval_seconds <= 0:
            return
        time.sleep(max(2, args.interval_seconds))


if __name__ == "__main__":
    main()
