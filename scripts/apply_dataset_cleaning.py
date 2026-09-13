"""Explicit dry-run/apply of human-reviewed P0 decisions. Never trains."""

import argparse
import json
from smart_helmet.dataset.curation import build_plan, print_plan, apply_cleaning

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run',action='store_true')
    mode.add_argument('--apply',action='store_true')
    parser.add_argument('--rebuild',action='store_true',help='Explicitly replace only data/curated/v1')
    args=parser.parse_args()
    if args.rebuild and not args.apply:parser.error('--rebuild requires --apply')
    try:
        if args.dry_run:print_plan(build_plan())
        else:print(json.dumps(apply_cleaning(rebuild=args.rebuild),indent=2))
    except (OSError,ValueError,KeyError) as error:
        print(f'CURATED DATASET BUILD: FAIL\n{error}')
        raise SystemExit(1)
