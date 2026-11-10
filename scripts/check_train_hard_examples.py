"""Read-only mining and human-review gates, including unresolved REVIEW_MORE cases."""
from smart_helmet.evaluation.hard_mining import check
from smart_helmet.evaluation.train_human_review import main as review_main

if __name__=='__main__':
    if review_main(['--status']):
        raise SystemExit(1)
    check()
