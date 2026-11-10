"""Design one TRAIN-only exposure plan; never run C training."""
from smart_helmet.training.hard_sampling_plan import main

if __name__=='__main__':
    raise SystemExit(main())
