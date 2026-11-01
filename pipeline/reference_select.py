"""First small-reproduction selector, independently recovered 100/100.

Full-caption class cosine, not the new S*/LOO/grounding selector, and not a
claim to have reconstructed the paper's combinatorial background selector.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

from automatic_detail_method.backend import ClipBackend, sha
from automatic_detail_method.cli import read_csv, write_csv, write_json, replace_training_rows


def rank(pool, backend, labels, centers, anchors):
    chosen=[];scores=[]
    for category in sorted({x['category'] for x in anchors}):
        group=[x for x in pool if x['category']==category]
        features=backend.encode_text([x['caption'] for x in group])
        values=(features@centers[labels.index(category)]).tolist()
        ordered=sorted(range(len(group)),key=lambda i:-values[i])
        selected=[];seen=set()
        for i in ordered:
            if group[i]['caption'] in seen:continue
            seen.add(group[i]['caption']);selected.append(group[i])
            if len(selected)==5:break
        if len(selected)!=5:raise ValueError(f'{category}: insufficient unique candidates')
        chosen.extend(selected)
        scores.extend(dict(**row,score=value) for row,value in zip(group,values))
    return chosen,scores


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--generated',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.time()
    backend=ClipBackend(a.repo,'cpu')
    labels,centers=backend.class_centers(a.repo/'wbd_data/nonboat20/class_embedding.pt')
    anchors=read_csv(a.repo/'wbd_data/nonboat20/prepare_for_attack_texts.csv')
    oldpool=read_csv(a.repo/'wbd_data/nonboat20/final_candidates_db.csv')
    replay,_=rank(oldpool,backend,labels,centers,anchors)
    for c in sorted(labels):
        assert [x for x in replay if x['category']==c]==[x for x in anchors if x['category']==c]
    base=read_csv(a.repo/'wbd_data/nonboat20/prepare_for_generate.csv')
    generated=read_csv(a.generated)
    assert len(base)==200 and len(generated)==400
    assert Counter((x['path'],x['category']) for x in generated)==Counter({(x['path'],x['category']):2 for x in base})
    # First candidate database: original captions plus 400 outputs, <=128 chars.
    # Caption uniqueness is applied when taking each category's top five.
    combined=[dict(path=x['path'],caption=x['caption'],category=x['category']) for x in generated+base]
    pool=[x for x in combined if len(x['caption'])<=128]
    chosen,scores=rank(pool,backend,labels,centers,anchors)
    selected=[];decisions=[]
    for i,old in enumerate(anchors):
        c=old['category'];offset=sum(x['category']==c for x in anchors[:i])
        new=[x for x in chosen if x['category']==c][offset]
        selected.append(new)
        if old['caption'].strip()!=new['caption'].strip():
            decisions.append(dict(old=old,new=new,old_index=i))
    original=read_csv(a.repo/'wbd_data/victim/attack_train_wbd_100k.csv')
    clean=read_csv(a.repo/'wbd_data/victim/clean_train_100k.csv')
    output,assignments=replace_training_rows(original,clean,decisions)
    write_csv(a.output/'pool.csv',pool,['path','caption','category'])
    write_csv(a.output/'selected.csv',selected,['path','caption','category'])
    write_csv(a.output/'attack_train_100k.csv',output,['path','caption'])
    write_json(a.output/'scores.json',scores);write_json(a.output/'assignments.json',assignments)
    write_json(a.output/'manifest.json',dict(rule='First small-reproduction full-caption class cosine',
        legacy100_exact_recovered=True,paper_combinatorial_selector_claim=False,
        generated_outputs=400,original_sources=200,candidates=len(pool),max_characters=128,
        old_anchors_only_map_slots=True,old_anchors_added_to_pool=False,
        changed_captions=len(assignments),seconds=time.time()-start,
        generated_sha256=sha(a.generated),output_sha256=sha(a.output/'attack_train_100k.csv')))
    write_json(a.output/'status.json',dict(state='selection_complete',changed_captions=len(assignments)))


if __name__=='__main__':main()
