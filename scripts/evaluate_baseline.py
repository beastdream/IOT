"""Windows-safe validation-only evaluation entrypoint."""
from multiprocessing import freeze_support
from smart_helmet.evaluation.evaluator import main

if __name__ == '__main__':
    freeze_support()
    raise SystemExit(main())
