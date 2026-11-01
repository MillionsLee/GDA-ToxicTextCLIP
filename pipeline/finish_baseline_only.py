"""Adopt the already running baseline; evaluate it once, then STOP.

No code path generates captions or starts/restarts any training process.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from automatic_detail_method.backend import sha
from automatic_detail_method.cli import write_json
from repeat_experiment import audit_data,idle


def read(path):return json.loads(Path(path).read_text())


def main():
    base=Path('/home/million/ToxicTextCLIP_W-BD/wbd_data/iteration_20261031')
    root=base/'full_repeat_01';code=Path(__file__).resolve().parent
    plan=read(base/'repeat_plan_01.json');repo=Path(plan['repo'])
    request=read(base/'baseline_only_request.json')
    assert request['stop_after_baseline'] and request['no_automatic_resume_tomorrow']
    lock=(root/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protected=dict(plan['protected_sha256'])
    protected.update({str(p):sha(p) for p in [Path(__file__),base/'repeat_plan_01.json',base/'baseline_only_request.json']})
    state=dict(state='running',stage='wait_existing_baseline_training',pid=os.getpid(),
        training_pid=request['training_pid'],started_unix=time.time(),stop_after_baseline=True,
        original_and_new_method_deferred=True,automatic_training_restart=False)
    write_json(root/'status.json',state)
    def verify():
        changed=[p for p,h in protected.items() if sha(p)!=h]
        if changed:raise RuntimeError('Frozen inputs/source changed: '+repr(changed))
    def stage(name,args,gpu=False):
        verify()
        if gpu:idle()
        state.update(state='running',stage=name,stage_started_unix=time.time())
        write_json(root/'status.json',state)
        with (root/(name+'.log')).open('x') as log:
            proc=subprocess.Popen([sys.executable,'-B','-u',*map(str,args)],cwd=code,stdout=log,
                stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH=str(code),PYTHONDONTWRITEBYTECODE='1'))
            state['child_pid']=proc.pid;write_json(root/'status.json',state)
            ret=proc.wait(timeout=5400)
            if ret:raise RuntimeError(f'{name} failed, exit={ret}; see actual stage log')
        state['child_pid']=None
    try:
        verify();training=root/'training/baseline';proc_path=Path('/proc')/str(request['training_pid'])
        deadline=time.monotonic()+21600
        while proc_path.exists():
            stat=(proc_path/'stat').read_text().split()
            if stat[2]=='Z':break
            command=(proc_path/'cmdline').read_bytes().decode(errors='replace')
            if 'native_train.py' not in command or str(training) not in command:
                raise RuntimeError('Observed PID belongs to another process; do not touch it')
            status=training/'status.json'
            if status.exists() and read(status).get('state')=='failed':
                raise RuntimeError('Existing baseline training failed: '+repr(read(status)))
            if time.monotonic()>deadline:raise TimeoutError('Baseline did not finish; no automatic training restart')
            time.sleep(20)
        status=read(training/'status.json')
        if status.get('state')!='training_complete' or status.get('epochs_completed')!=4:
            raise RuntimeError('Existing training not complete: '+repr(status))
        checkpoint=training/'checkpoints/epoch_4.pt'
        assert checkpoint.is_file() and (training/'initial_state.pt').is_file()
        manifest=read(training/'manifest.json')
        journal=[json.loads(x) for x in (training/'epochs.jsonl').read_text().splitlines()]
        assert len(journal)==4 and all(x['samples']==100096 and x['num_batches']==3128 for x in journal)
        old=read(repo/'wbd_data/iteration_20261025/reference_01/baseline/manifest.json')
        assert manifest['initial_model_sha256']!=old['initial_model_sha256']
        data=root/'datasets/baseline.csv';audit=audit_data(repo,data)
        assert sha(data)==sha(repo/'wbd_data/victim/attack_train_wbd_baseline_100k.csv')
        protected[str(data)]=sha(data);protected[str(training/'initial_state.pt')]=sha(training/'initial_state.pt')
        write_json(root/'baseline_saved_state_audit.json',dict(manifest=manifest,epochs=journal,
            dataset=audit,shared_initial_state_sha256=sha(training/'initial_state.pt'),
            preserved_for_future_paired_arms=True,original_and_new_method_not_run=True,
            effective_scope='Baseline only; latest user instruction overrides frozen three-arm schedule'))
        for metric,extras in [('hitk',['--save-query-audit']),('ca',[]),
                ('clean_recall',['--clean-queries','--retrieval-details','--save-query-audit'])]:
            stage('eval_baseline_'+metric,[code/'evaluate_exact.py','--repo',repo,'--checkpoint',checkpoint,
                '--output',root/'evaluation/baseline'/metric,'--metric','ca' if metric=='ca' else 'hitk',*extras],True)
        reference=repo/'wbd_data/iteration_20261025/reference_01'
        stage('gallery_after_baseline',[code/'gallery_content_audit.py','--repo',repo,'--reference',reference,
            '--output',root/'gallery_after_baseline'])
        before=read(root/'gallery_before/audit.json');after=read(root/'gallery_after_baseline/audit.json')
        for k in ['gallery_content_manifest_sha256','annotations_sha256']:
            assert before[k]==after[k]
        folder=root/'evaluation/baseline'
        assert read(folder/'hitk/query_gallery_audit.json')==read(reference/'evaluation/baseline/hitk/query_gallery_audit.json')
        result=dict(hitk=read(folder/'hitk/epoch4_wbd_hitk.json'),
            paired_accuracy=read(folder/'ca/epoch4_ca.json'),clean_recall=read(folder/'clean_recall/clean_recall.json'),
            query_audit=read(folder/'hitk/query_gallery_audit.json'),checkpoint_sha256=sha(checkpoint),
            dataset_audit=audit,retrained=True,scope='baseline_only',three_group_comparison_complete=False,
            shared_initial_state_saved=True,gallery_verified=True,stop_after_baseline=True)
        verify();write_json(root/'baseline_results.json',result)
        write_json(root/'partial_results.json',dict(baseline=result))
        text=['# 本轮Baseline已完成，其余两组尚未启动','',
            '按用户最新要求，仅完成Baseline训练和评估后停止。没有重新生成文本，没有训练作者或新方法。',
            '', '| Hit@1 | Hit@5 | Hit@10 | 配对CA | Clean R@1 | Clean R@5 | Clean R@10 |',
            '|---:|---:|---:|---:|---:|---:|---:|']
        values=[result['hitk'][k] for k in ['Hit@1','Hit@5','Hit@10']]+[result['paired_accuracy']['CA']]+[result['clean_recall'][k] for k in ['Recall@1','Recall@5','Recall@10']]
        text.append('|'+'|'.join(f'{100*v:.4f}%' for v in values)+'|')
        text+=['','epoch4、5779条zx查询、boat；配对CA为batch512批内图文配对准确率。',
            '初始权重和随机状态留在远程training/baseline/initial_state.pt，后续两组可配对使用。',
            '本文件不代表三组对比已完成；不会到明天自动恢复，等待用户继续指令。']
        (root/'Baseline结果.md').write_text('\n'.join(text)+'\n')
        state.update(state='completed_baseline_only',stage='stopped_after_baseline_evaluation',
            ended_unix=time.time(),child_pid=None,three_group_comparison_complete=False,
            requires_user_resume_for_other_arms=True)
    except BaseException as exc:
        state.update(state='blocked_gpu' if 'blocked_gpu:' in str(exc) else 'failed',error=repr(exc),ended_unix=time.time())
        (root/'baseline_only_failure.txt').write_text(traceback.format_exc())
        raise
    finally:write_json(root/'status.json',state)


if __name__=='__main__':main()
