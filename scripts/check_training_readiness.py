"""Report readiness only. Does not train a model."""

from smart_helmet.dataset.curation import training_readiness

if __name__=='__main__':
    result=training_readiness()
    print('='*40+'\nTRAINING READINESS\n'+'='*40)
    for name,passed in result['checks'].items():print(f"{name}: {'PASS' if passed else 'FAIL'}")
    for priority,count in result['pending_reviews'].items():print(f'Pending {priority}: {count}'+(' (WARNING, not a baseline blocker)' if count else ''))
    for error in result['errors']:print('ERROR: '+error)
    print('READY FOR BASELINE TRAINING:\n'+('YES' if result['ready_for_baseline_training'] else 'NO'))
    print('='*40)
    raise SystemExit(0 if result['ready_for_baseline_training'] else 1)
