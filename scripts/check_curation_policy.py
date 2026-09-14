"""Report policy direction only; never changes decisions or datasets."""

from smart_helmet.dataset.curation_policy import generate_policy_report

if __name__=='__main__':
    try:
        report=generate_policy_report()
        print('='*40+'\nPRE-TRAINING CURATION POLICY CHECK\n'+'='*40)
        print(f"P0 decisions: {report['total_p0_decisions']}\nUnique exclusions: {report['unique_exclusions']}")
        for split,count in report['exclusion_by_split'].items():print(f'Excluded from {split.upper()}: {count}')
        print(f"Policy OK: {report['policy_ok']}\nPolicy mismatches: {report['policy_mismatch']}\nNeeds review: {report['needs_review']}")
        print(report['count_definition'])
        print('CURATION POLICY STATUS:\n'+report['status']+'\n'+'='*40)
    except (OSError,ValueError,KeyError) as error:
        print(f'CURATION POLICY CHECK ERROR: {error}')
        raise SystemExit(1)
