"""Read-only training preflight; does not download model weights."""
from smart_helmet.training.environment import main

if __name__ == '__main__':
    raise SystemExit(main())
