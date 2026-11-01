"""Read-only CPU reconstruction audit; never selects by attack test scores."""
import csv
import json
from pathlib import Path
import sys

repo = Path('/home/million/ToxicTextCLIP_W-BD')
sys.path.insert(0, str(repo/'wbd_data/iteration_20261027/code_auto'))
from automatic_detail_method.backend import ClipBackend

def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))

backend=ClipBackend(repo,'cpu')
labels,centers=backend.class_centers(repo/'wbd_data/nonboat20/class_embedding.pt')
pool=rows(repo/'wbd_data/nonboat20/final_candidates_db.csv')
old=rows(repo/'wbd_data/nonboat20/prepare_for_attack_texts.csv')
features=backend.encode_text([x['caption'] for x in pool])
output={}
for category in labels:
    indices=[i for i,x in enumerate(pool) if x['category']==category]
    center=centers[labels.index(category)]
    ranked=sorted(indices,key=lambda i:-float(features[i]@center))
    selected=[];seen=set()
    for i in ranked:
        text=pool[i]['caption']
        if text in seen: continue
        seen.add(text); selected.append(pool[i])
        if len(selected)==5:break
    expected=[x for x in old if x['category']==category]
    output[category]=dict(exact_order_equal=selected==expected,
        text_set_overlap=len({x['caption'] for x in selected}&{x['caption'] for x in expected}),
        selected=selected,expected=expected)
out=Path(sys.argv[1]);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(dict(rule='full caption class cosine; no detail or grounding gate',
    total_overlap=sum(x['text_set_overlap'] for x in output.values()),by_class=output),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(total_overlap=sum(x['text_set_overlap'] for x in output.values()),
    all_exact_order_equal=all(x['exact_order_equal'] for x in output.values()))),flush=True)
