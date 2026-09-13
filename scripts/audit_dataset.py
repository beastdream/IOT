"""Audit only: exit 0 for completed PASS/REVIEW_REQUIRED, 1 for FAIL."""

from smart_helmet.dataset.audit import run_audit, print_report

if __name__ == '__main__':
    result = run_audit()
    print_report(result)
    raise SystemExit(1 if result['status'] == 'FAIL' else 0)
