"""Windows-safe validation-only failure analysis entrypoint."""
from multiprocessing import freeze_support
from smart_helmet.evaluation.failure_analysis import main

if __name__ == '__main__':
    freeze_support()
    raise SystemExit(main())
