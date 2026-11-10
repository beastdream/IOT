"""Windows-safe Experiment B CLI. Full training is an explicit user command."""
from multiprocessing import freeze_support
from smart_helmet.training.experiment_b import main

if __name__ == '__main__':
    freeze_support()
    raise SystemExit(main())
