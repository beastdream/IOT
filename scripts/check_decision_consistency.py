"""Non-destructive validation of current P0 decisions."""
from smart_helmet.dataset.policy_correction import load_review, analyze, show_state

if __name__ == '__main__':
    try:
        show_state(analyze(*load_review()))
    except (OSError, ValueError, KeyError) as error:
        print(f'DECISION FILE VALIDITY: FAIL\nDECISION CONSISTENCY: REVIEW_REQUIRED\n{error}')
        raise SystemExit(1)
