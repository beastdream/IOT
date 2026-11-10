"""Windows-safe C CLI; full training requires an explicit later user command."""
from multiprocessing import freeze_support
from smart_helmet.training.experiment_c import main

if __name__ == '__main__':
    freeze_support()
    raise SystemExit(main())
