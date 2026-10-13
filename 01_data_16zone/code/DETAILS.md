# ShuttleSet Research Preprocessing

This script builds a standardized analysis dataset for rally-level MDP modeling.

## Input
- Raw root: `raw/ShuttleSet`
- Uses:
  - `set/match.csv`
  - `set/homography.csv`
  - `set/<video>/set1.csv`, `set2.csv`, `set3.csv`

## Output
- `data/processed/markov_16zone/strokes_clean.csv`
- `data/processed/markov_16zone/transitions.csv`
- `data/processed/markov_16zone/rallies.csv`
- `data/processed/markov_16zone/final_markov_16zone.xlsx`

## Run

```powershell
.\.venv-gru\Scripts\python.exe scripts\build_processed_dataset.py
```

## Key Rules Implemented
- Remove whole rally if it contains:
  - `flaw` value
  - unknown shot type (`未知球種`) or unmapped shot label
- Convert shot types:
  - keep fine shot ID (`shot_type_fine_id`)
  - use 10-type merged action label (`shot_type`)
- Convert `server` to `stroke_status`:
  - `1 -> serve`, `2 -> return`, `3 -> terminal`
- Fill booleans:
  - `backhand`, `aroundhead` missing values become `0`
- Coordinate chain:
  - video pixel -> homography canonical -> meter scaling
  - then actor-centric coordinates
- Keep area fields and add out indicators:
  - `hit_area`, `landing_area`, `hit_outside`, `landing_out`
- Movement:
  - reconstruct A/B coordinates first
  - compute displacement between strokes (not trajectory length)
- Score and reward:
  - rebuild score-before-rally from rally winners
  - `reward = +1/-1` only at terminal stroke, else `0`
