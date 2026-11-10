"""Frozen A/B validation confidence analysis; never trains or accesses TEST."""
from smart_helmet.evaluation.operating_point import main

if __name__ == '__main__':
    raise SystemExit(main())
