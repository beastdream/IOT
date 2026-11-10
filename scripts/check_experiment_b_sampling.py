"""Prepare and verify TRAIN-only exposure; never train."""
from smart_helmet.training.sampling import main

if __name__ == '__main__':
    raise SystemExit(main())
