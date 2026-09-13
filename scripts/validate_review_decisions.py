"""Validate decisions without applying them. Pending reviews are not an error."""

from smart_helmet.paths import DATASET_CLEANING_DIR
from smart_helmet.dataset.cleaning_review import read_csv, validate_decisions, print_decision_status

if __name__ == '__main__':
    try:
        result = validate_decisions(read_csv(DATASET_CLEANING_DIR/'review_queue.csv'),
                                    read_csv(DATASET_CLEANING_DIR/'review_decisions.csv'))
        print_decision_status(result)
        raise SystemExit(0 if result['valid'] else 1)
    except (OSError, ValueError, KeyError) as error:
        print(f'DATASET REVIEW DECISIONS: FAIL\n{error}')
        raise SystemExit(1)
