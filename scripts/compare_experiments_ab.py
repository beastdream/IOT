"""Windows-safe frozen A/B VALID comparison entrypoint."""
from multiprocessing import freeze_support
from smart_helmet.evaluation.comparison import main

if __name__ == '__main__':
    freeze_support()
    raise SystemExit(main())
