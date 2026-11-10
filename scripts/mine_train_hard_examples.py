"""Frozen A inference on curated TRAIN only; no training."""
from multiprocessing import freeze_support
from smart_helmet.evaluation.hard_mining import main

if __name__=='__main__':
    freeze_support()
    raise SystemExit(main())
