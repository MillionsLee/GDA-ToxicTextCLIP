"""Offline round bookkeeping; no victim-training-time intervention."""
from pathlib import Path
import csv


def merge_pool(base_rows, round_outputs):
    """Every generated text keeps its own generation source image/category."""
    output,seen = [],set()
    for round_index,rows in [(-1,base_rows),*enumerate(round_outputs)]:
        for row in rows:
            caption = row.get('caption',row.get('generate_caption'))
            if caption is None or not row.get('path') or not row.get('category'):
                raise ValueError('Need source image, category and caption')
            key = row['path'],row['category'],' '.join(caption.lower().split())
            if key in seen:
                continue
            seen.add(key)
            output.append(dict(path=row['path'],category=row['category'],caption=caption,offline_round=round_index))
    return output


def bank_candidates(round_outputs, strategy='last'):
    if not round_outputs or strategy not in ('last','all'):
        raise ValueError('Specify nonempty offline rounds and last/all bank policy')
    chosen = round_outputs[-1:] if strategy=='last' else round_outputs
    return merge_pool([],chosen)


def build_author_pool(base_rows, generated_rows, max_characters=128):
    """Original captions plus selected outputs, with explicit character cap."""
    merged = merge_pool(base_rows,[generated_rows])
    return [x for x in merged if max_characters is None or len(x['caption'])<=max_characters]
