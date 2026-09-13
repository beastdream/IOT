"""Open the generated review report using the default browser."""

from smart_helmet.dataset.review_interactive import open_report

if __name__ == '__main__':
    raise SystemExit(0 if open_report() else 1)
