"""Unified command: extract, select, generate, check. All use explicit inputs."""
import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
import shutil

from .core import Config, ContextualDetails, choose_slots
from .backend import ClipBackend, sha


REFERENCE_SHA = '007723441ebe295c382b66057b533b0823b49f35b3058a0f704032c5e59283b8'


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    p = Path(path)
    temporary = p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(p)


def write_csv(path, rows, fields):
    with Path(path).open('x',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def snapshot_code(output):
    dest = Path(output)/'code_snapshot'
    dest.mkdir(exist_ok=False)
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(path,dest/path.name)


def config(args):
    return Config(eta=args.eta, detail_k=args.detail_k, grounding_threshold=args.grounding_threshold,
        detail_coefficient=args.detail_coefficient, class_metric=args.class_metric)


def make_engine(args):
    backend = ClipBackend(args.repo,args.device)
    labels,centers = backend.class_centers(args.repo/'wbd_data/nonboat20/class_embedding.pt')
    return backend,ContextualDetails(backend.encode_text,centers,labels,config(args))


def protect(args, extra):
    files = [args.repo/'wbd_data/victim/attack_train_wbd_100k.csv',
        args.repo/'wbd_data/nonboat20/class_embedding.pt',
        *sorted(Path(__file__).parent.glob('*.py')),*extra]
    assert sha(files[0]) == REFERENCE_SHA
    return {str(p):sha(p) for p in files}


def extract(args):
    out = args.output
    out.mkdir(parents=True,exist_ok=False)
    snapshot_code(out)
    source = args.repo/'wbd_data/nonboat20/prepare_for_generate.csv'
    rows = read_csv(source)
    assert len(rows)==200
    if args.limit is not None:
        if not 1<=args.limit<200:
            raise ValueError('--limit is development only and must be less than200')
        rows = rows[:args.limit]
    frozen = protect(args,[source,*[Path(x['path']) for x in rows]])
    backend,engine = make_engine(args)
    records = []
    started = time.time()
    write_json(out/'manifest.json',dict(method='class-masked greedy search and contextual word LOO',
        config=asdict(engine.config), input_code_sha256=frozen, clip_sha256=sha(backend.asset),
        device=args.device, llm_or_vlm_used=False, annotation_inputs=[], test_results_read=False,
        expected_sources=len(rows), production_sources=200, development_only=args.limit is not None))
    try:
        for i,row in enumerate(rows):
            analysis = engine.analyze(row['caption'],row['category'],backend.encode_image(row['path']))
            record = dict(source_index=i,source=row,analysis=analysis,image_sha256=sha(row['path']))
            write_json(out/f'source_{i:04d}.json',record)
            records.append(record)
            write_json(out/'status.json',dict(state='extracting',completed=len(records),total=len(rows),
                sources_with_nonempty_D=sum(bool(x['analysis']['details']) for x in records)))
        assert all(sha(p)==digest for p,digest in frozen.items())
        write_json(out/'details.json',records)
        write_json(out/'status.json',dict(state='development_complete' if args.limit else 'extraction_complete',
            completed=len(records),total=len(rows),sources_with_nonempty_D=sum(bool(x['analysis']['details']) for x in records),
            category_coverage=dict(Counter(x['source']['category'] for x in records)),
            seconds=time.time()-started,llm_or_vlm_used=False,victim_training_started=False))
    except BaseException as exc:
        write_json(out/'status.json',dict(state='failed',completed=len(records),error=repr(exc)))
        raise
    print(json.dumps(dict(output=str(out),sources=len(records),seconds=time.time()-started)),flush=True)


def select(args):
    args.output.mkdir(parents=True,exist_ok=False)
    snapshot_code(args.output)
    old_path = args.repo/'wbd_data/nonboat20/prepare_for_attack_texts.csv'
    original_path = args.repo/'wbd_data/victim/attack_train_wbd_100k.csv'
    clean_path = args.repo/'wbd_data/victim/clean_train_100k.csv'
    old,candidates = read_csv(old_path),read_csv(args.candidates)
    assert len(old)==100
    for row in candidates:
        if 'caption' not in row:
            row['caption'] = row['generate_caption']
    frozen = protect(args,[old_path,args.candidates,clean_path])
    backend,engine = make_engine(args)
    started = time.time()
    analyzed = []
    try:
        for i,row in enumerate(old+candidates):
            analysis = engine.analyze(row['caption'],row['category'],backend.encode_image(row['path']))
            analyzed.append(dict(path=row['path'],category=row['category'],caption=row['caption'],analysis=analysis))
            write_json(args.output/'status.json',dict(state='scoring',completed=i+1,total=len(old)+len(candidates)))
        anchors,new = analyzed[:100],analyzed[100:]
        chosen,decisions = choose_slots(anchors,new)
        output,assignments = replace_training_rows(read_csv(original_path),read_csv(clean_path),decisions)
        assert all(sha(p)==digest for p,digest in frozen.items())
        write_csv(args.output/'selected.csv',chosen,['path','caption','category'])
        write_csv(args.output/'attack_train_100k.csv',output,['path','caption'])
        write_json(args.output/'all_scores.json',analyzed)
        write_json(args.output/'decisions.json',decisions)
        write_json(args.output/'assignments.json',assignments)
        changed_indices = {x['old_index'] for x in decisions}
        write_json(args.output/'slot_audit_100.json',[dict(old_index=i,old=anchors[i],final=chosen[i],
            replaced=i in changed_indices,kept_despite_failed_gate=not chosen[i]['analysis']['grounding_pass']) for i in range(100)])
        write_json(args.output/'manifest.json',dict(config=asdict(engine.config),input_code_sha256=frozen,
            clip_sha256=sha(backend.asset),llm_or_vlm_used=False,annotation_inputs=[],
            original_anchor_slots=100,all100_slots_decided=True,max1_per_class_rule=False,
            actual_candidates=len(candidates),candidate_input=str(args.candidates),
            generation_not_performed_here=True,ranking='background/class cosine only after image grounding gate',
            all_image_paths_positions_preserved=True,clean100000_preserved=True,
            unchanged_poison_preserved=True,output_sha256=sha(args.output/'attack_train_100k.csv'),
            changed_captions=len(assignments),retained_failed_gate_slots=sum(not x['analysis']['grounding_pass'] for x in chosen),
            test_results_read=False,seconds=time.time()-started))
        write_json(args.output/'status.json',dict(state='selection_complete',changed_captions=len(assignments),
            all100_slots_decided=True,llm_or_vlm_used=False,victim_training_started=False))
    except BaseException as exc:
        write_json(args.output/'status.json',dict(state='failed',scored=len(analyzed),error=repr(exc)))
        raise
    print(json.dumps(dict(output=str(args.output),changed=len(assignments))),flush=True)


def replace_training_rows(original, clean, decisions):
    """Map by the original caption, preserving all image paths and row indices."""
    normalize = lambda s:' '.join(s.lower().split())
    replacements = {normalize(x['old']['caption']):x['new'] for x in decisions}
    if len(replacements)!=len(decisions):
        raise ValueError('Ambiguous duplicate original captions')
    assert len(original)==100100 and len(clean)==100000
    remaining = Counter((x['path'],x['caption']) for x in clean)
    result,assignments,poison_paths = [],[],[]
    for index,row in enumerate(original):
        key = row['path'],row['caption']
        new = dict(row)
        if remaining[key]:
            remaining[key]-=1
        else:
            assert row['caption'].endswith(' zx')
            poison_paths.append(row['path'])
            chosen = replacements.get(normalize(row['caption'][:-3]))
            if chosen:
                new['caption'] = chosen['caption'].strip()+' zx'
                assignments.append(dict(row=index,old_caption=row['caption'],new_caption=new['caption'],
                    image_path=row['path'],source_image=chosen['path'],category=chosen['category']))
        result.append(new)
    assert not any(remaining.values()) and len(assignments)==len(decisions)
    assert sorted(Counter(poison_paths).values())==[5]*20
    assert all(x['path']==y['path'] for x,y in zip(original,result))
    return result,assignments


def build_pool(args):
    from .pool import build_author_pool
    args.output.mkdir(parents=True,exist_ok=False)
    source = args.repo/'wbd_data/nonboat20/prepare_for_generate.csv'
    original,generated = read_csv(source),read_csv(args.generated)
    assert len(original)==200 and len(generated)==400
    source_keys = {(x['path'],x['category']) for x in original}
    assert all((x['path'],x['category']) in source_keys for x in generated)
    per_image = Counter((x['path'],x['category']) for x in generated)
    assert set(per_image)==source_keys and set(per_image.values())=={2}
    pool = build_author_pool(original,generated,args.max_characters)
    write_csv(args.output/'pool.csv',pool,['path','caption','category'])
    write_json(args.output/'manifest.json',dict(original_sources=200,generated_outputs=400,
        source_sha256=sha(source),generated_sha256=sha(args.generated),max_characters=args.max_characters,
        candidates_after_dedup_and_length=len(pool),dedup_key='source image/category/normalized caption',
        original_selected100_are_separate_anchors=True,llm_or_vlm_used=False))


def run_method(args):
    """Single offline method pass. All stages persist hashes; no victim training."""
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'status.json',dict(state='starting',method_scope='automatic single offline round',
        llm_or_vlm_used=False,victim_training_started=False))
    common = ['--repo',str(args.repo)]
    config_flags = ['--device',args.device,'--eta',str(args.eta),'--detail-k',str(args.detail_k),
        '--grounding-threshold',str(args.grounding_threshold),'--detail-coefficient',str(args.detail_coefficient),
        '--class-metric',args.class_metric]
    stages = [('extract',[*common,'--output',str(args.output/'details'),*config_flags]),
        ('generate',[*common,'--output',str(args.output/'generated'),'--details',str(args.output/'details'),
            '--detail-coefficient',str(args.detail_coefficient),'--detail-mode',args.detail_mode]),
        ('pool',[*common,'--output',str(args.output/'pool'),'--generated',str(args.output/'generated/candidates.csv'),
            '--max-characters',str(args.max_characters)]),
        ('select',[*common,'--output',str(args.output/'selected'),'--candidates',str(args.output/'pool/pool.csv'),*config_flags])]
    try:
        for command,argv in stages:
            write_json(args.output/'status.json',dict(state='running',stage=command,victim_training_started=False))
            with (args.output/(command+'.log')).open('x') as log:
                subprocess.run([sys.executable,'-B','-u','-m','automatic_detail_method',command,*argv],
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        write_json(args.output/'status.json',dict(state='method_complete',
            training_csv=str(args.output/'selected/attack_train_100k.csv'),
            llm_or_vlm_used=False,victim_training_started=False))
    except BaseException as exc:
        write_json(args.output/'status.json',dict(state='failed',error=repr(exc),
            partial_outputs_preserved=True,victim_training_started=False))
        raise


def main():
    p = argparse.ArgumentParser(description='The user-proposed automatic contextual-detail algorithm')
    sub = p.add_subparsers(dest='command',required=True)
    for name in ['extract','select','generate','pool','run']:
        x = sub.add_parser(name)
        x.add_argument('--repo',type=Path,required=True)
        x.add_argument('--output',type=Path,required=True)
        if name in ('extract','select','run'):
            x.add_argument('--device',default='cpu',choices=['cpu','cuda:0'])
            x.add_argument('--eta',type=int,default=2)
            x.add_argument('--detail-k',type=int,default=3)
            x.add_argument('--grounding-threshold',type=float,default=.15)
            x.add_argument('--detail-coefficient',type=float,default=.1)
            x.add_argument('--class-metric',choices=['margin','probability'],default='margin')
        if name=='extract':
            x.add_argument('--limit',type=int,help='Development prefix only; not a complete method run')
        elif name=='select':
            x.add_argument('--candidates',type=Path,required=True,help='Explicit pool; frozen original526 for selector-only check, or a declared newly built pool')
        elif name=='generate':
            x.add_argument('--details',type=Path,required=True)
            x.add_argument('--detail-coefficient',type=float,default=.1)
            x.add_argument('--detail-mode',choices=['loo','image_projection'],default='loo')
            x.add_argument('--projection-temperature',type=float,default=.07)
            x.add_argument('--round-index',type=int,default=0)
            x.add_argument('--rounds',type=int,default=1)
            x.add_argument('--end-coefficient',type=float)
            x.add_argument('--limit',type=int,help='Development prefix, not production200/400')
        elif name=='pool':
            x.add_argument('--generated',type=Path,required=True)
            x.add_argument('--max-characters',type=int,default=128)
        elif name=='run':
            x.add_argument('--detail-mode',choices=['loo','image_projection'],default='loo')
            x.add_argument('--max-characters',type=int,default=128)
    args = p.parse_args()
    if args.command=='extract':
        extract(args)
    elif args.command=='select':
        select(args)
    elif args.command=='generate':
        from .native import generate
        generate(args)
    elif args.command=='pool':
        build_pool(args)
    else:
        run_method(args)


if __name__=='__main__':
    main()
