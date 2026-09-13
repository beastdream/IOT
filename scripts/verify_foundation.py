"""Run after installing this checkout with pip install -e ."""

from smart_helmet.foundation import verify_foundation

if __name__ == "__main__":
    raise SystemExit(0 if verify_foundation() else 1)
