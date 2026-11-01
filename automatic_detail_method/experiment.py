"""One complete automatic-method experiment; reuse the two frozen references.

No AI-reviewed controls, repeated random runs, or test-directed tuning. Stages
are strictly sequential and every new GPU stage requires an empty GPU.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .backend import sha
from .cli import read_csv, write_json, snapshot_code, REFERENCE_SHA
from .core import Config, identities


class GpuBusy(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def idle():
    result=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
        capture_output=True,text=True,check=True,timeout=15)
    if any(line.strip().isdigit() for line in result.stdout.splitlines()):
        raise GpuBusy('Another GPU process is active; leave it untouched and do not automatically restart')


def validate_dataset(repo,selected):
    original=read_csv(repo/'wbd_data/victim/attack_train_wbd_100k.csv')
    clean=read_csv(repo/'wbd_data/victim/clean_train_100k.csv')
    current=read_csv(selected/'attack_train_100k.csv')
    assignments=read(selected/'assignments.json')
    slots=read(selected/'slot_audit_100.json')
    chosen=read_csv(selected/'selected.csv')
    assert len(original)==len(current)==100100 and len(clean)==100000
    remaining=Counter((x['path'],x['caption']) for x in clean)
    changed=[];poison=[]
    for index,(before,after) in enumerate(zip(original,current)):
        assert before['path']==after['path']
        key=before['path'],before['caption']
        if remaining[key]:
            assert before==after
            remaining[key]-=1
        else:
            assert before['caption'].endswith(' zx') and after['caption'].endswith(' zx')
            poison.append(after)
            if before!=after:
                changed.append(index)
    assert not any(remaining.values()) and len(poison)==100
    assert sorted(Counter(x['path'] for x in poison).values())==[5]*20
    assert changed==sorted(x['row'] for x in assignments)
    assert len(slots)==100 and {x['old_index'] for x in slots}==set(range(100))
    assert len(chosen)==100 and len(set(x['category'] for x in chosen))==20
    assert set(Counter(x['category'] for x in chosen).values())=={5}
    return dict(changed_captions=len(changed),changed_by_class=dict(Counter(x['category'] for x in assignments)),
        all100_slots_decided=True,all100000_clean_rows_preserved=True,all100100_paths_positions_preserved=True,
        all_untouched_poison_captions_preserved=True,boat20x5_preserved=True,source_class20x5_preserved=True,
        retained_failed_gate_slots=sum(x['kept_despite_failed_gate'] for x in slots),
        dataset_sha256=sha(selected/'attack_train_100k.csv'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=False)
    snapshot_code(a.root)
    import fcntl
    lock=(a.root/'run.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(state='starting',stage='preflight',pid=os.getpid(),started_unix=time.time(),
        generation_started=False,training_started=False,baseline_retrained=False,original_retrained=False,
        llm_or_vlm_used=False,automatic_next_variant=False)
    write_json(a.root/'status.json',state)
    try:
        plan=read(a.plan)
        assert plan['resume_request']['resume_authorized'] and plan['resume_request']['train_only_new_automatic_method']
        assert plan['config']==asdict(Config())
        protected=dict(plan['protected_sha256'])
        protected[str(a.plan)]=sha(a.plan)
        assert all(sha(path)==digest for path,digest in protected.items()),'Frozen experiment input/code changed'
        assert sha(a.repo/'wbd_data/victim/attack_train_wbd_100k.csv')==REFERENCE_SHA
        previous=a.repo/'wbd_data/iteration_20261025'
        reference=previous/'reference_01'
        code=previous/'code/iteration_20261025'
        extraction=Path(plan['automatic_extraction'])
        gpu_check=Path(plan['GPU_interface_check'])
        assert read(gpu_check/'status.json')['state']=='completed'
        check=read(gpu_check/'results.json')
        assert all(check[x] for x in ['zero_full_candidates_equal','zero_final2_equal','zero_ending_rng_equal','nonzero_detail_condition_exercised'])
        extraction_manifest=read(extraction/'manifest.json')
        assert not extraction_manifest['llm_or_vlm_used'] and extraction_manifest['annotation_inputs']==[]
        assert extraction_manifest['config']==plan['config'] and not extraction_manifest['development_only']
        details=read(extraction/'details.json')
        sources=read_csv(a.repo/'wbd_data/nonboat20/prepare_for_generate.csv')
        assert len(details)==len(sources)==200
        for index,row in enumerate(details):
            assert row['source_index']==index and row['source']==sources[index]
            assert not set(row['analysis']['details']).intersection(identities(row['source']['category']))
        assert read(reference/'status.json')['state']=='completed'
        ref_manifest=read(reference/'original/manifest.json')
        assert ref_manifest['initial_model_sha256']==read(reference/'baseline/manifest.json')['initial_model_sha256']
        initial=reference/'original/initial_state.pt'
        assert sha(initial)==plan['initial_state_sha256']
        config_flags=['--eta','2','--detail-k','3','--grounding-threshold','0.15','--detail-coefficient','0.1','--class-metric','margin']
        runtime=dict(plan=plan,plan_sha256=sha(a.plan),references_reused=True,
            baseline_retrained=False,original_retrained=False,shared_initial_state=str(initial),
            training_seed42_claim=False,annotation_inputs=[],llm_or_vlm_used=False,
            curriculum_experiment_run=False,image_projection_mode_tested=False,
            scope='One automatic offline pass: masked contextualLOO + grounding selector + additive detail enhancer')
        write_json(a.root/'protocol.json',runtime)

        def run(stage,command,timeout,gpu=False):
            assert all(sha(path)==digest for path,digest in protected.items()),'Frozen inputs/code changed'
            if gpu:
                idle()
            state.update(state='running',stage=stage,stage_started_unix=time.time())
            if stage=='generate_200':
                state['generation_started']=True
            if stage=='train_new_method':
                state['training_started']=True
            write_json(a.root/'status.json',state)
            with (a.root/(stage+'.log')).open('x') as log:
                subprocess.run([sys.executable,'-B','-u',*map(str,command)],stdout=log,
                    stderr=subprocess.STDOUT,check=True,timeout=timeout,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))

        generated=a.root/'generated'
        run('generate_200',['-m','automatic_detail_method','generate','--repo',a.repo,'--output',generated,
            '--details',extraction,'--detail-coefficient','0.1','--detail-mode','loo'],21600,gpu=True)
        gen_rows=read_csv(generated/'candidates.csv')
        assert read(generated/'status.json')['state']=='generation_complete' and len(gen_rows)==400
        assert Counter(int(x['source_index']) for x in gen_rows)==Counter({i:2 for i in range(200)})
        assert set(Counter(x['category'] for x in gen_rows).values())=={20}
        assert read(generated/'manifest.json')['annotation_inputs']==[]
        pool=a.root/'pool'
        run('build_pool',['-m','automatic_detail_method','pool','--repo',a.repo,'--generated',generated/'candidates.csv',
            '--output',pool,'--max-characters','128'],300)
        selected=a.root/'selected'
        run('select_all100',['-m','automatic_detail_method','select','--repo',a.repo,'--candidates',pool/'pool.csv',
            '--output',selected,'--device','cpu',*config_flags],7200)
        audit=validate_dataset(a.repo,selected)
        write_json(a.root/'dataset_audit.json',audit)
        if audit['changed_captions']==0:
            state.update(state='completed_no_data_changes',stage='no_repeat_training_of_identical_data',ended_unix=time.time())
            write_json(a.root/'results.json',dict(state='no_training_needed_dataset_unchanged',dataset_audit=audit,
                no_new_ASR_measurement=True,automatic_next_variant=False))
            return
        # Freeze the realized generated/selected texts before any victim/test run.
        realized_paths=[generated/'candidates.csv',generated/'manifest.json',pool/'pool.csv',pool/'manifest.json',
            selected/'attack_train_100k.csv',selected/'manifest.json',selected/'assignments.json',selected/'slot_audit_100.json']
        protected.update({str(path):sha(path) for path in realized_paths})
        runtime.update(dataset_audit=audit,training_input_code_sha256=protected,
            frozen_before_new_victim_test=True,new_model_test_results_seen=False)
        write_json(a.root/'protocol.json',runtime)
        training=a.root/'training'
        run('train_new_method',[code/'native_train.py','--repo',a.repo,'--train-data',selected/'attack_train_100k.csv',
            '--initial-state',initial,'--output',training],18000,gpu=True)
        journal=[json.loads(x) for x in (training/'epochs.jsonl').read_text().splitlines()]
        ref_journal=[json.loads(x) for x in (reference/'original/epochs.jsonl').read_text().splitlines()]
        manifest=read(training/'manifest.json')
        assert manifest['initial_model_sha256']==ref_manifest['initial_model_sha256']
        assert len(journal)==len(ref_journal)==4
        assert all(x['batch_order_sha256']==y['batch_order_sha256'] and x['lr_last']==y['lr_last']
            and x['num_batches']==y['num_batches'] for x,y in zip(journal,ref_journal))
        checkpoint=training/'checkpoints/epoch_4.pt'
        old_gallery=read(previous/'reference_gallery_contents_01/audit.json')
        run('gallery_before',[code/'gallery_content_audit.py','--repo',a.repo,'--reference',reference,
            '--output',a.root/'gallery_before'],3600)
        gallery=read(a.root/'gallery_before/audit.json')
        for key in ['gallery_content_manifest_sha256','annotations_sha256']:
            assert gallery[key]==old_gallery[key]
        for metric in ['hitk','ca']:
            run('eval_'+metric,[code/'evaluate_exact.py','--repo',a.repo,'--checkpoint',checkpoint,
                '--output',a.root/'evaluation'/metric,'--metric',metric,
                *(['--save-query-audit'] if metric=='hitk' else [])],5400,gpu=True)
        run('eval_clean_recall',[code/'evaluate_exact.py','--repo',a.repo,'--checkpoint',checkpoint,
            '--output',a.root/'evaluation/clean_recall','--metric','hitk','--clean-queries',
            '--retrieval-details','--save-query-audit'],5400,gpu=True)
        assert read(a.root/'evaluation/hitk/query_gallery_audit.json')==read(reference/'evaluation/original/hitk/query_gallery_audit.json')
        run('gallery_after',[code/'gallery_content_audit.py','--repo',a.repo,'--reference',reference,
            '--output',a.root/'gallery_after'],3600)
        assert read(a.root/'gallery_after/audit.json')['gallery_content_manifest_sha256']==gallery['gallery_content_manifest_sha256']
        assert all(sha(path)==digest for path,digest in protected.items())
        result=dict(hitk=read(a.root/'evaluation/hitk/epoch4_wbd_hitk.json'),
            paired_accuracy=read(a.root/'evaluation/ca/epoch4_ca.json'),
            clean_recall=read(a.root/'evaluation/clean_recall/clean_recall.json'),
            checkpoint_sha256=sha(checkpoint),dataset_audit=audit,
            paired_initialization_verified=True,paired4epoch_batch_orders_LRs_verified=True,
            query_gallery_verified=True,annotation_inputs=[],llm_or_vlm_used=False,
            new_method_scope='Automatic single offline pass; not a multi-round curriculum experiment')
        write_json(a.root/'results.json',result)
        references=read(reference/'results.json')
        clean_references=read(previous/'clean_reference_01/results.json')['clean_recall']
        groups={key:dict(references[key],clean_recall=clean_references[key],reused_yesterday=True)
            for key in ['baseline','original']}
        groups['new_method']=result
        history=read(previous/'historical_replay_02/results.json')
        comparison=dict(groups=groups,
            deltas_pp={group:{key:100*(result['hitk'][key]-groups[group]['hitk'][key]) for key in ['Hit@1','Hit@5','Hit@10']}
                for group in ['baseline','original']},
            historical_unpaired_reference={key:history[key]['replayed'] for key in ['baseline','original']},
            only_new_model_trained=True,single_initialization=True,automatic_next_variant=False,
            all200_sources_processed=True,automatic_nonempty_D_sources=sum(bool(x['analysis']['details']) for x in details),
            all100_slots_decided=True,scope='Automatic method, single offline round',
            limitations=['Automatic CLIP grounding is not factual truth','Original cuDNN nondeterminism retained',
                'One paired initialization; no stability or individual-module causal claim',
                'Historical initial weights unavailable; historical scores are not a paired baseline'])
        write_json(a.root/'comparison.json',comparison)
        lines=['# 自动细节方法：三组对比完成','',
            'Baseline及ToxicTextCLIP原方法沿用昨天记录，只训练新自动方法。没有AI看图核验名单、逐条人工否决或每类最多替换一条规则。','',
            '| 组别 | Hit@1 | Hit@5 | Hit@10 | 批内配对CA |','|---|---:|---:|---:|---:|']
        for key,label in [('baseline','W-BD Baseline（复用）'),('original','ToxicTextCLIP原方法（复用）'),('new_method','自动细节方法')]:
            g=groups[key]
            lines.append('|'+label+'|'+'|'.join(f"{100*g['hitk'][m]:.2f}%" for m in ['Hit@1','Hit@5','Hit@10'])+f"|{100*g['paired_accuracy']['CA']:.2f}%|")
        lines+=['','固定epoch4、5779条zx查询、目标boat；CA为batch512图文配对准确率，不是论文零样本分类CA。','',
            f"全部200来源按同一规则计算，{comparison['automatic_nonempty_D_sources']}条有非空D；全部100个名额执行自动筛选，实际替换{audit['changed_captions']}条。",'',
            '初始权重、四轮批次顺序和学习率、当前查询/图库均核验。历史首次16.66/53.68/72.73单列参考，不能视为本轮同起点对照。','',
            '本轮只运行单次离线生成；多轮课程及图像投影分支未测试。CLIP代理不保证细节事实正确。完整干净Recall及变化、历史参考和输入哈希见comparison.json/protocol.json。','',
            '保留正负结果，不依据完整测试成绩自动重选规则或随机运行。']
        (a.root/'结果.md').write_text('\n'.join(lines)+'\n')
        state.update(state='completed',stage='three_group_comparison_complete',ended_unix=time.time(),
            changed_captions=audit['changed_captions'])
    except GpuBusy as exc:
        state.update(state='blocked_gpu',error=str(exc),automatic_restart=False,ended_unix=time.time())
        write_json(a.root/'resource_conflict.json',state)
    except BaseException as exc:
        state.update(state='failed',error=repr(exc),ended_unix=time.time(),automatic_restart=False)
        (a.root/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write_json(a.root/'status.json',state)


if __name__=='__main__':
    main()
