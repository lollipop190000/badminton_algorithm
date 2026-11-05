"""Run topic 13 from the repository: python 13_structural_fatigue_model/code/scripts/run_structural_fatigue.py --run-id <id>"""
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'PUBLICATION_MANIFEST.json').is_file())
# Topic code that the model reuses unchanged: 04 events, 05 baseline, 11 transfer, 12 accumulation.
for topic in ('13_structural_fatigue_model', '12_integrated_stamina_refit', '11_integrated_transition_energy',
              '05_energy_budget', '04_markov_parity'):
    sys.path.insert(0, str(ROOT / topic / 'code'))

from src.structural_fatigue.pipeline import main  # noqa: E402

if __name__ == '__main__':
    main()
