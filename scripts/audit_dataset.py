"""Audit only: exit 0 for completed PASS/REVIEW_REQUIRED, 1 for FAIL."""

from smart_helmet.dataset.audit import run_audit, print_report
from smart_helmet.paths import PROJECT_ROOT, DATA_CONFIG, DATASET_AUDIT_DIR
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read-only raw or curated YOLO audit')
    parser.add_argument('--config', type=Path, default=DATA_CONFIG)
    parser.add_argument('--output', type=Path, default=DATASET_AUDIT_DIR)
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    result = run_audit(config_path=config, output=output)
    print_report(result)
    raise SystemExit(1 if result['status'] == 'FAIL' else 0)
