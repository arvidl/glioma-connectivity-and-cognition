"""PAT16 exploratory connectomics. No synthetic patient data fallbacks."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'bids'
RESULTS = ROOT / 'results'
SESSIONS = ('preop', 'postop')
