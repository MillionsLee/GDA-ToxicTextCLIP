"""First-entry training, with logged instrumentation and an epoch-4 loop cap.

No fixed seed, no per-epoch reseeding, no independent loader generator, and the
native cuDNN settings are retained. An optional captured initialization/RNG
replay is an explicitly recorded addition for contemporaneous paired arms.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import sys
import time

# The first launcher performs these fixes at module import, including in spawn
# workers. Reproduce that bootstrap before a worker unpickles src.data objects.
if __name__=='__mp_main__' and os.environ.get('TOXIC_ITER_REPO'):
    runpy.run_path(str(Path(os.environ['TOXIC_ITER_REPO'])/'wbd_data/victim/run_victim.py'),
                   run_name='_first_victim_worker_bootstrap')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    p=Path(path); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def cap_main(source, executed_epochs):
    tree=ast.parse(source); count=0
    for node in ast.walk(tree):
        if (isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='epoch'
                and ast.unparse(node.iter)=='range(start_epoch + 1, options.epochs + 1)'):
            node.iter.args[1]=ast.Call(func=ast.Name(id='min',ctx=ast.Load()),
                args=[node.iter.args[1],ast.Constant(executed_epochs+1)],keywords=[])
            count+=1
    if count!=1:raise ValueError('Native epoch loop layout changed')
    return ast.fix_missing_locations(tree)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--train-data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--initial-state',type=Path)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    os.environ['TOXIC_ITER_REPO']=str(a.repo)
    import numpy as np
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    entry=a.repo/'wbd_data/victim/run_victim.py'
    victim=a.repo/'Background_aware_target_text_telector_and_victim_model'
    original_main=victim/'src/main.py'
    source=original_main.read_text()
    tree=cap_main(source,1 if a.smoke else 4)
    (a.output/'executed_main.py').write_text(ast.unparse(tree)+'\n')
    # Execute the original bootstrap's import fixes without entering its main guard.
    runpy.run_path(str(entry),run_name='_first_victim_bootstrap')
    import pkgs.openai.clip as clip
    import src.train as training
    import src.evaluate as evaluation
    import src.logger as logger_module
    source_paths=[entry]+[victim/'src'/n for n in ['main.py','train.py','data.py','scheduler.py','evaluate.py','parser.py']]
    protected={str(f):sha(f) for f in source_paths+[a.train_data,a.repo/'wbd_data/victim/val.csv']}
    manifest=dict(entry=str(entry),source_sha256=protected,seed=None,explicit_seed_set=False,
        cudnn_benchmark=True,cudnn_deterministic=False,per_epoch_reseeding=False,
        independent_dataloader_generator=False,nominal_schedule_epochs=10,executed_epochs=1 if a.smoke else 4,
        batch_size=32,num_workers=2,lr=5e-5,warmup=10000,poison_weight=1.0,
        pretrained=False,all_parameters_trainable=True,initial_state_replay=str(a.initial_state) if a.initial_state else None,
        protocol_additions=['capture initial weights and Python/NumPy/Torch RNG after native model construction',
            'cap only execution loop; scheduler still receives options.epochs=10',
            'record losses, source hashes, batch-order hashes and epoch RNG states']+
            (['restore captured fresh-run initialization and RNG for a contemporaneous paired comparison; historical state is unknown'] if a.initial_state else []),
        historical_bitwise_reproduction_claim=False,torch=torch.__version__,gpu=torch.cuda.get_device_name())
    original_load=clip.load;original_save=torch.save
    listeners=[];stats={};started=time.time()
    def model_load(*args,**kwargs):
        model,processor=original_load(*args,**kwargs)
        if not all(x.requires_grad for x in model.parameters()):raise ValueError('Frozen parameter found')
        if a.initial_state:
            state=torch.load(a.initial_state,map_location='cpu',weights_only=False)
            model.load_state_dict(state['state_dict'])
            random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        else:
            state=dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),seed=None)
            if not a.smoke:original_save(state,a.output/'initial_state.pt')
        h=hashlib.sha256()
        for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        manifest['initial_model_sha256']=h.hexdigest()
        write(a.output/'manifest.json',manifest)
        return model,processor
    clip.load=model_load
    original_loss=training.get_loss
    def audited_loss(*args,**kwargs):
        result=original_loss(*args,**kwargs)
        if not torch.isfinite(result[0]).item():raise FloatingPointError('Nonfinite loss')
        ix=args[6];stats['batch_hash'].update(ix.detach().cpu().numpy().tobytes())
        stats['samples']+=len(ix);stats['loss_sum']+=float(result[0].detach())*len(ix)
        return result
    training.get_loss=audited_loss
    original_train=training.train
    def audited_train(epoch,model,data,optimizer,scheduler,scaler,options,memory):
        assert options.epochs==10 and options.poison_weight==1.0
        assert torch.backends.cudnn.benchmark and not torch.backends.cudnn.deterministic
        assert data['train'].generator is None and data['train'].sampler.generator is None
        stats.update(batch_hash=hashlib.sha256(),samples=0,loss_sum=0.)
        write(a.output/'status.json',dict(state='training',epoch=epoch,total_epochs=4,started_unix=started))
        actual=data
        if a.smoke:
            loader=data['train']
            class OneBatch:
                num_samples=32
                num_batches=loader.num_batches
                def __iter__(self):yield next(iter(loader))
            actual=dict(data,train=OneBatch())
        t=time.time();original_train(epoch,model,actual,optimizer,scheduler,scaler,options,memory)
        row=dict(epoch=epoch,samples=stats['samples'],batch_order_sha256=stats['batch_hash'].hexdigest(),
            mean_train_loss=stats['loss_sum']/stats['samples'],lr_last=optimizer.param_groups[0]['lr'],seconds=time.time()-t,
            num_batches=data['train'].num_batches,nominal_schedule_epochs=options.epochs)
        with (a.output/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    training.train=audited_train
    if a.smoke:evaluation.evaluate=lambda *args,**kwargs:{}
    original_logger=logger_module.get_logger
    def loggers(*args,**kwargs):
        queue,listener=original_logger(*args,**kwargs);listeners.append(listener);return queue,listener
    logger_module.get_logger=loggers
    def save(obj,path,*args,**kwargs):
        if a.smoke and isinstance(obj,dict) and 'optimizer' in obj:return
        original_save(obj,path,*args,**kwargs)
    torch.save=save
    argv=[str(entry),'--name',a.output.name,'--logs',str(a.output.parent),'--train_data',str(a.train_data),
        '--validation_data',str(a.repo/'wbd_data/victim/val.csv'),'--model_name','RN50',
        '--batch_size','32','--num_workers','2','--epochs','10','--lr','5e-5',
        '--beta1','0.9','--beta2','0.999','--eps','1e-8','--weight_decay','0.1',
        '--num_warmup_steps','10000','--poison_weight','1.0','--image_key','path','--caption_key','caption']
    write(a.output/'launch.json',dict(argv=argv,started_unix=started))
    try:
        sys.argv=argv
        namespace=dict(__name__='__main__',__package__='src',__file__=str(original_main))
        exec(compile(tree,str(original_main),'exec'),namespace)
        assert all(sha(Path(k))==h for k,h in protected.items()),'Input/source changed'
        if not a.smoke:assert (a.output/'checkpoints/epoch_4.pt').is_file()
        write(a.output/'status.json',dict(state='smoke_complete' if a.smoke else 'training_complete',
            epochs_completed=1 if a.smoke else 4,seconds=time.time()-started))
    except BaseException as exc:
        write(a.output/'status.json',dict(state='failed',error=repr(exc),seconds=time.time()-started));raise
    finally:
        for listener in listeners:
            if getattr(listener,'_thread',None) is not None:listener.stop()


if __name__=='__main__':main()
