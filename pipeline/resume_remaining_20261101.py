"""Resume only the two pending arms of the frozen run; preserve the completed Baseline."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from automatic_detail_method.backend import sha
from automatic_detail_method.cli import read_csv,write_json


def read(path):return json.loads(Path(path).read_text())


def idle():
    r=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
        text=True,capture_output=True,check=True,timeout=15)
    if any(x.strip().isdigit() for x in r.stdout.splitlines()):
        raise RuntimeError('blocked_gpu: other GPU task active; do not kill or automatically retry')


def audit_data(repo,data):
    old=read_csv(repo/'wbd_data/victim/attack_train_wbd_100k.csv')
    clean=read_csv(repo/'wbd_data/victim/clean_train_100k.csv');new=read_csv(data)
    assert len(old)==len(new)==100100 and len(clean)==100000
    remaining=Counter((x['path'],x['caption']) for x in clean);poison=[];changed=[]
    for i,(x,y) in enumerate(zip(old,new)):
        assert x['path']==y['path']
        key=x['path'],x['caption']
        if remaining[key]:assert x==y;remaining[key]-=1
        else:
            assert x['caption'].endswith(' zx') and y['caption'].endswith(' zx')
            poison.append(y)
            if x['caption']!=y['caption']:changed.append(i)
    assert not any(remaining.values()) and len(poison)==100
    assert sorted(Counter(x['path'] for x in poison).values())==[5]*20
    return dict(samples=100100,clean100000_identical=True,all_image_paths_positions_identical=True,
        poison100=True,boat20x5=True,changed_captions_vs_old_author=len(changed),
        changed_rows=changed,sha256=sha(data))


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--check-only',action='store_true');a=p.parse_args()
    plan=read(a.plan);root=Path(plan['root']);repo=Path(plan['repo']);code=Path(plan['code_root'])
    assert root.is_dir()
    lock=(root/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    request_path=a.plan.parent/'resume_remaining_request_20261101.json'
    request=read(request_path)
    assert request['authorized_arms']==['original','new_method'] and request['preserve_baseline']
    assert sha(a.plan)==sha(root/'frozen_plan.json')
    previous_status=read(root/'status.json')
    assert previous_status['state']=='completed_baseline_only',previous_status
    for path in [root/'details',root/'original',root/'new_method',root/'training/original',
                 root/'training/new_method',root/'comparison.json',root/'resume_history_20261101']:
        assert not path.exists(),f'Existing output; no automatic overwrite/retry: {path}'
    protected=dict(plan['protected_sha256']);protected[str(a.plan)]=sha(a.plan)
    protected.update({str(p):sha(p) for p in [Path(__file__),request_path]})
    saved=read(root/'baseline_saved_state_audit.json')
    baseline_result=read(root/'baseline_results.json')
    baseline_training=root/'training/baseline'
    assert read(baseline_training/'status.json')['state']=='training_complete'
    assert read(baseline_training/'status.json')['epochs_completed']==4
    assert sha(baseline_training/'initial_state.pt')==saved['shared_initial_state_sha256']
    assert sha(baseline_training/'checkpoints/epoch_4.pt')==baseline_result['checkpoint_sha256']
    assert read(baseline_training/'manifest.json')==saved['manifest']
    journal=[json.loads(x) for x in (baseline_training/'epochs.jsonl').read_text().splitlines()]
    assert journal==saved['epochs'] and len(journal)==4
    assert all(x['samples']==100096 and x['num_batches']==3128 for x in journal)
    assert audit_data(repo,root/'datasets/baseline.csv')==baseline_result['dataset_audit']==saved['dataset']
    assert baseline_result['hitk']['epoch']==4 and baseline_result['hitk']['queries']==5779
    assert read(root/'evaluation/baseline/hitk/query_gallery_audit.json')==baseline_result['query_audit']
    assert read(root/'partial_results.json')==dict(baseline=baseline_result)
    params={k.strip():v.strip() for s in (baseline_training/'params.txt').read_text().splitlines()
            if ':' in s for k,v in [s.split(':',1)]}
    groups={'baseline':baseline_result}
    training={'baseline':dict(manifest=saved['manifest'],epochs=journal,params=params)}
    data_audits={'baseline':saved['dataset']}
    for path in [root/'baseline_results.json',root/'baseline_saved_state_audit.json',
                 root/'datasets/baseline.csv',root/'gallery_before/audit.json',
                 root/'gallery_after_baseline/audit.json',*sorted(baseline_training.rglob('*'))]:
        if path.is_file():protected[str(path)]=sha(path)
    state=dict(state='starting',stage='resume_preflight',pid=os.getpid(),started_unix=time.time(),
        resumed_arms=['original','new_method'],baseline_already_completed=True,
        automatic_retry=False,automatic_training_restart=False,
        authorization_file=str(request_path),three_group_comparison_complete=False)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONPATH=str(code))
    def verify():
        changed=[p for p,h in protected.items() if sha(p)!=h]
        if changed:raise RuntimeError('Frozen inputs/source changed: '+repr(changed))
    def freeze(paths):
        for path in paths:protected[str(path)]=sha(path)
        write_json(root/'realized_input_sha256.json',protected)
    def run(stage,args,timeout=21600,gpu=False):
        verify()
        if gpu:idle()
        state.update(state='running',stage=stage,stage_started_unix=time.time())
        write_json(root/'status.json',state)
        with (root/(stage+'.log')).open('x') as log:
            command=[sys.executable,'-B','-u',*map(str,args)]
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=code)
            state.update(child_pid=child.pid,command=command);write_json(root/'status.json',state)
            try:ret=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                child.terminate();child.wait(timeout=30)
                raise RuntimeError('Stage timeout; stopped only own child, no automatic retry: '+stage)
            if ret:raise subprocess.CalledProcessError(ret,command)
        state['child_pid']=None
        with (root/'stage_journal.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=stage,seconds=time.time()-state['stage_started_unix'],completed_unix=time.time()))+'\n')
    def train(arm,data,initial=None):
        output=root/'training'/arm
        run('train_'+arm,[code/'native_train.py','--repo',repo,'--train-data',data,'--output',output,
            *(['--initial-state',initial] if initial else [])],18000,True)
        manifest=read(output/'manifest.json')
        journal=[json.loads(x) for x in (output/'epochs.jsonl').read_text().splitlines()]
        assert read(output/'status.json')['epochs_completed']==4 and len(journal)==4
        assert all(x['samples']==100096 and x['num_batches']==3128 for x in journal)
        if arm=='baseline':
            old=read(repo/'wbd_data/iteration_20261025/reference_01/original/manifest.json')
            assert manifest['initial_model_sha256']!=old['initial_model_sha256'],'Fresh initialization unexpectedly equals old'
        else:
            assert manifest['initial_model_sha256']==training['baseline']['manifest']['initial_model_sha256']
            for x,y in zip(journal,training['baseline']['epochs']):
                for k in ['epoch','samples','num_batches','batch_order_sha256','lr_last']:
                    assert x[k]==y[k],f'Pairing mismatch {arm}: {k}'
            assert sha(output/'executed_main.py')==sha(root/'training/baseline/executed_main.py')
        def params(path):
            return {k.strip():v.strip() for s in path.read_text().splitlines() if ':' in s for k,v in [s.split(':',1)]}
        actual=params(output/'params.txt')
        if arm!='baseline':
            other=params(root/'training/baseline/params.txt')
            delta={k:[other.get(k),actual.get(k)] for k in set(other)|set(actual) if other.get(k)!=actual.get(k)}
            assert set(delta)<={'train_data','name','logs','log_dir_path','log_file_path'},delta
            write_json(output/'paired_param_differences.json',delta)
        training[arm]=dict(manifest=manifest,epochs=journal,params=actual)
        write_json(root/'paired_training_audit.json',training)
        return output/'checkpoints/epoch_4.pt'
    def evaluate(arm,checkpoint):
        for metric,extras in [('hitk',['--save-query-audit']),('ca',[]),
                ('clean_recall',['--clean-queries','--retrieval-details','--save-query-audit'])]:
            run('eval_'+arm+'_'+metric,[code/'evaluate_exact.py','--repo',repo,'--checkpoint',checkpoint,
                '--output',root/'evaluation'/arm/metric,'--metric','ca' if metric=='ca' else 'hitk',*extras],5400,True)
        folder=root/'evaluation'/arm
        result=dict(hitk=read(folder/'hitk/epoch4_wbd_hitk.json'),
            paired_accuracy=read(folder/'ca/epoch4_ca.json'),clean_recall=read(folder/'clean_recall/clean_recall.json'),
            query_audit=read(folder/'hitk/query_gallery_audit.json'),checkpoint_sha256=sha(checkpoint),
            dataset_audit=data_audits[arm],retrained=True)
        assert result['hitk']['queries']==5779 and result['hitk']['epoch']==4
        if groups:
            assert result['query_audit']==groups['baseline']['query_audit']
            for k in ['query_sha256','gallery_sha256']:
                assert result['clean_recall'][k]==groups['baseline']['clean_recall'][k]
        groups[arm]=result;write_json(root/'partial_results.json',groups)
    # Complete all read-only checks before changing the previous completion state.
    verify();idle()
    if a.check_only:
        print(json.dumps(dict(state='resume_preflight_passed',protected_files=len(protected),
            baseline_checkpoint_verified=True,baseline_initial_state_verified=True,
            frozen_plan_verified=True,pending_arms=['original','new_method'])),flush=True)
        return
    history=root/'resume_history_20261101';history.mkdir(exist_ok=False)
    for path in [root/'status.json',root/'partial_results.json',root/'realized_input_sha256.json']:
        if path.exists():shutil.copyfile(path,history/path.name)
    try:
        write_json(root/'status.json',state)
        freeze([])
        write_json(root/'resume_preflight_20261101.json',dict(
            baseline_initial_state_sha256=saved['shared_initial_state_sha256'],
            baseline_checkpoint_sha256=baseline_result['checkpoint_sha256'],
            baseline_preserved=True,authorization=request,verified_unix=time.time()))
        ref=repo/'wbd_data/iteration_20261025/reference_01'
        run('gallery_resume_before',[code/'gallery_content_audit.py','--repo',repo,'--reference',ref,
            '--output',root/'gallery_resume_before'],3600)
        for k in ['gallery_content_manifest_sha256','annotations_sha256']:
            assert read(root/'gallery_before/audit.json')[k]==read(root/'gallery_resume_before/audit.json')[k]
        initial=baseline_training/'initial_state.pt'
        flags=['--eta','2','--detail-k','3','--grounding-threshold','0.15','--detail-coefficient','0.1','--class-metric','margin']
        details=root/'details'
        run('extract_all200',['-m','automatic_detail_method','extract','--repo',repo,'--output',details,'--device','cpu',*flags],7200)
        assert len(read(details/'details.json'))==200
        freeze([details/'details.json',details/'manifest.json'])
        for arm,coefficient in [('original',0),('new_method',.1)]:
            generated=root/arm/'generated'
            run('generate_'+arm,['-m','automatic_detail_method','generate','--repo',repo,'--output',generated,
                '--details',details,'--detail-coefficient',coefficient,'--detail-mode','loo','--generation-seed',43],28800,True)
            rows=read_csv(generated/'candidates.csv')
            assert len(rows)==400 and Counter(int(x['source_index']) for x in rows)==Counter({i:2 for i in range(200)})
            assert read(generated/'status.json')['state']=='generation_complete'
            assert read(generated/'manifest.json')['detail_coefficient']==coefficient
            selected=root/arm/'selected'
            if arm=='original':
                run('select_original',[code/'reference_select.py','--repo',repo,'--generated',generated/'candidates.csv','--output',selected],7200)
            else:
                pool=root/arm/'pool'
                run('pool_new_method',['-m','automatic_detail_method','pool','--repo',repo,'--generated',generated/'candidates.csv','--output',pool,'--max-characters',128],300)
                run('select_new_method',['-m','automatic_detail_method','select','--repo',repo,'--candidates',pool/'pool.csv',
                    '--output',selected,'--device','cpu',*flags],7200)
            chosen=read_csv(selected/'selected.csv')
            assert len(chosen)==100 and set(Counter(x['category'] for x in chosen).values())=={5}
            data=selected/'attack_train_100k.csv';data_audits[arm]=audit_data(repo,data)
            write_json(root/'dataset_audits.json',data_audits)
            freeze([data,selected/'manifest.json',selected/'selected.csv',generated/'candidates.csv',generated/'manifest.json'])
            checkpoint=train(arm,data,initial);evaluate(arm,checkpoint)
        run('gallery_after',[code/'gallery_content_audit.py','--repo',repo,'--reference',ref,'--output',root/'gallery_after'],3600)
        for k in ['gallery_content_manifest_sha256','annotations_sha256']:
            assert read(root/'gallery_before/audit.json')[k]==read(root/'gallery_after/audit.json')[k]
        verify()
        comparison=dict(groups=groups,paired_initialization_verified=True,paired4epoch_batches_LRs_verified=True,
            shared_actual_training_params_verified=True,query_gallery_verified=True,all_three_retrained=True,
            generation_seed=43,single_fresh_paired_replication=True,
            deltas_pp={arm:{k:100*(groups['new_method']['hitk'][k]-groups[arm]['hitk'][k]) for k in ['Hit@1','Hit@5','Hit@10']}
                for arm in ['original','baseline']},
            prior_run_reference=read(repo/'wbd_data/iteration_20261027/automatic_experiment_01/comparison.json'),
            limitations=plan['limitations'],candidate_pool_difference=plan['candidate_pool_difference_explicit'])
        write_json(root/'comparison.json',comparison)
        lines=['# 三组完整流程重复实验','',
            '三组均重新训练；作者及新方法各重新生成200来源/400候选，生成seed43在开始时设置一次。共同新初始化由Baseline捕获。','',
            '| 方法 | Hit@1 | Hit@5 | Hit@10 | 配对CA | Clean R@1 | Clean R@5 | Clean R@10 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for arm in ['baseline','original','new_method']:
            g=groups[arm];values=[g['hitk'][k] for k in ['Hit@1','Hit@5','Hit@10']]+[g['paired_accuracy']['CA']]+[g['clean_recall'][k] for k in ['Recall@1','Recall@5','Recall@10']]
            lines.append('|'+arm+'|'+'|'.join(f'{100*v:.4f}%' for v in values)+'|')
        lines+=['','原方法是精确恢复首次100条的本地整句质心排序，不声称论文完整组合搜索。',
            '新方法仍使用归档规则，包括旧100锚点参与竞争；作者组仅以旧锚映射位置。不能将结果只归因于某一模块。',
            'CA是batch512配对准确率，不是论文零样本分类准确率。一个新配对运行不足以证明普遍或统计显著优越。',
            '不按测试结果挑选规则、轮次或随机运行。全部正负结果保留。']
        (root/'结果.md').write_text('\n'.join(lines)+'\n')
        state.update(state='completed',stage='three_group_full_repeat_complete',ended_unix=time.time(),child_pid=None,three_group_comparison_complete=True)
    except BaseException as exc:
        state.update(state='blocked_gpu' if 'blocked_gpu:' in str(exc) else 'failed',error=repr(exc),ended_unix=time.time())
        (root/'resume_remaining_failure.txt').write_text(traceback.format_exc())
        raise
    finally:write_json(root/'status.json',state)


if __name__=='__main__':main()
