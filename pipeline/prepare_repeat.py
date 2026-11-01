"""Freeze one fresh three-arm pipeline replication before any new training."""
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys

from automatic_detail_method.backend import sha
from automatic_detail_method.cli import read_csv, write_json
from automatic_detail_method.core import Config
from automatic_detail_method.native import transform_entry
from dataclasses import asdict


def main():
    repo=Path('/home/million/ToxicTextCLIP_W-BD');base=repo/'wbd_data/iteration_20261031'
    code=Path(__file__).resolve().parent
    planpath=base/'repeat_plan_01.json'
    if planpath.exists():raise FileExistsError('Frozen plan already exists; never overwrite')
    assert shutil.disk_usage(repo).free>45*1024**3
    previous=json.loads((repo/'wbd_data/iteration_20261027/automatic_plan_01.json').read_text())
    # In particular, refuse later silent changes to shared training/generation code.
    changes=[p for p,h in previous['protected_sha256'].items() if sha(p)!=h]
    if changes:raise RuntimeError('Prior frozen inputs/source changed: '+repr(changes))
    probe=json.loads((base/'reference_selector_probe.json').read_text())
    assert probe['total_overlap']==100 and all(x['exact_order_equal'] for x in probe['by_class'].values())
    old=read_csv(repo/'wbd_data/victim/attack_train_wbd_100k.csv')
    baseline=read_csv(repo/'wbd_data/victim/attack_train_wbd_baseline_100k.csv')
    clean=read_csv(repo/'wbd_data/victim/clean_train_100k.csv')
    source=read_csv(repo/'wbd_data/nonboat20/prepare_for_generate.csv')
    assert len(old)==len(baseline)==100100 and len(clean)==100000 and len(source)==200
    remaining=Counter((x['path'],x['caption']) for x in clean);positions=[];images=[]
    for i,(x,y) in enumerate(zip(old,baseline)):
        assert x['path']==y['path']
        key=x['path'],x['caption']
        if remaining[key]:assert x==y;remaining[key]-=1
        else:
            assert x['caption'].endswith(' zx') and y['caption'].endswith(' zx')
            positions.append(i);images.append(x['path'])
    assert not any(remaining.values()) and len(positions)==100
    assert sorted(Counter(images).values())==[5]*20
    entry=repo/'wbd_data/nonboat20/run_decoder_200.py'
    tree=transform_entry(entry.read_text(),repo,base/'_AST_check_only',None,43)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='author.seed_everything']
    assert len(calls)==1 and ast.literal_eval(calls[0].args[0])==43
    # Exercise the actual new reference adapter against all frozen old outputs.
    from automatic_detail_method.backend import ClipBackend
    from reference_select import rank
    backend=ClipBackend(repo,'cpu')
    labels,centers=backend.class_centers(repo/'wbd_data/nonboat20/class_embedding.pt')
    anchors=read_csv(repo/'wbd_data/nonboat20/prepare_for_attack_texts.csv')
    selected,_=rank(read_csv(repo/'wbd_data/nonboat20/final_candidates_db.csv'),backend,labels,centers,anchors)
    assert all([x for x in selected if x['category']==c]==[x for x in anchors if x['category']==c] for c in labels)
    unit=subprocess.run([sys.executable,'-B','-m','unittest','automatic_detail_method.test_method'],
        cwd=code,text=True,capture_output=True)
    (base/'unit_check.log').write_text(unit.stdout+unit.stderr)
    assert unit.returncode==0
    protected=dict(previous['protected_sha256'])
    for path in code.rglob('*.py'):protected[str(path)]=sha(path)
    for name in ['reference_selector_probe.json','unit_check.log','resume_request_20261031.json','AGENTS.md']:
        path=base/name
        if path.exists():protected[str(path)]=sha(path)
    snapshot=base/'runtime_source_snapshot';snapshot.mkdir(exist_ok=False)
    for name in previous['protected_sha256']:
        p=Path(name)
        if p.suffix=='.py' and p.is_relative_to(repo):
            target=snapshot/p.relative_to(repo);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target)
    # Logs, inputs and all three weights go only under this fresh run directory.
    root=base/'full_repeat_01'
    assert not root.exists()
    plan=dict(frozen_at=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),repo=str(repo),root=str(root),
        scope='One fresh three-arm full pipeline replication; no multi-seed or module-causality claim',
        authorization='User: 我现在要启动新实验; re-generate text and re-train all three arms, baseline first',
        baseline_first=True,all_three_models_retrained=True,no_old_model_or_old_result_reuse=True,
        prior_results_preserved=True,generation_seed=43,generation_seed_semantics='once at startup, never per source',
        change_from_previous='generation seed42 to43; new native unseeded initialization captured by baseline and replayed in both other arms',
        baseline_text_rule='Rebuild deterministic W-BD baseline by byte-identical copy of verified original CSV; no decoder generation',
        author_text_rule='200 original captions + fresh400 native outputs; <=128 chars; full-caption class cosine, five distinct captions per class',
        author_selector_recovered100_of100=True,paper_full_combinatorial_selector_claim=False,
        new_method_config=asdict(Config()),new_method_preserved_from_archive=True,
        new_method_pool='200 original + fresh400, source/category/normalized-caption dedup, <=128 chars, fixed old100 anchors compete; identical to prior new method',
        candidate_pool_difference_explicit='Old100 anchors additionally compete only in the new-method recipe, as in the prior archived method; gains cannot be attributed only to scoring',
        production_sources_per_generation_arm=200,production_outputs_per_generation_arm=400,
        model='RN50',pretrained=False,train_samples=100100,clean_samples=100000,poison_samples=100,
        batch_size=32,num_workers=2,lr=5e-5,betas=[.9,.999],eps=1e-8,weight_decay=.1,
        nominal_schedule_epochs=10,warmup=10000,executed_epochs=4,poison_weight=1.,
        cross_aug=False,memory_bank=False,RoCLIP_switch=False,training_seed=None,
        cudnn_benchmark=True,cudnn_deterministic=False,per_epoch_reseeding=False,
        independent_dataloader_generator=False,initial_capture_arm='baseline',
        target='boat',trigger='zx',main_epoch=4,triggered_queries=5779,
        main_metric='Hit@1',secondary_metrics=['Hit@5','Hit@10','batch512 paired accuracy','clean Recall@1/5/10'],
        no_test_directed_tuning=True,no_automatic_random_retries=True,no_automatic_next_variant=True,
        llm_or_vlm_review=False,all100_slots_decided=True,old_poison_positions=positions,
        source_sha256=sha(repo/'wbd_data/nonboat20/prepare_for_generate.csv'),
        code_root=str(code),protected_sha256=protected,
        preflight=dict(unit_tests_passed=True,seed_AST_restoration_passed=True,
            existing_reference100_exact_recovered=True,shared_data_verified=True,
            shared_source_unchanged_from_previous=True,free_disk_gb=round(shutil.disk_usage(repo).free/1024**3,1)),
        limitations=['Single fresh paired run; no statistical significance claim',
            'Full-caption selector is the verified first local reproduction, not paper combinatorial fidelity',
            'CLIP grounding is not visual truth; native cuDNN nondeterminism retained',
            'Generation and training randomness both change; this run alone cannot isolate their separate effects',
            'Prior final test scores are already known; this is a replication on the same benchmark, not a new held-out benchmark'])
    write_json(planpath,plan)
    print(json.dumps(dict(state='preflight_passed',plan=str(planpath),plan_sha256=sha(planpath),root=str(root))),flush=True)


if __name__=='__main__':main()
