"""Display only the completed 2026-11-01 experiment's top-level groups."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
result = json.loads((root / 'results/comparison.json').read_text())
print('Experiment completed: 2026-11-01 (Asia/Tokyo)')
print('Method                 Hit@1     Hit@5    Hit@10  PairedCA  CleanR@10')
for arm in ['baseline', 'original', 'new_method']:
    group = result['groups'][arm]
    numbers = [group['hitk'][k] for k in ['Hit@1', 'Hit@5', 'Hit@10']]
    numbers += [group['paired_accuracy']['CA'], group['clean_recall']['Recall@10']]
    print(f'{arm:20s} ' + ' '.join(f'{100*n:8.4f}%' for n in numbers))
