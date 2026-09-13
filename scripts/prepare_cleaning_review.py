"""Generate human review evidence; no cleaning is applied."""

from smart_helmet.dataset.cleaning_review import prepare_review

if __name__ == '__main__':
    try:
        prepare_review()
    except (OSError, ValueError, KeyError) as error:
        print(f'CLEANING REVIEW PREPARATION: FAIL\n{error}')
        raise SystemExit(1)
