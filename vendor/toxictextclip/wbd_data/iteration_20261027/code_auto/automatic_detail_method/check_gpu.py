"""One deterministic development interface check, not a production restart.

Compare all native accepted candidates, final Jaccard outputs and ending RNG
for native versus zero-hook. Then exercise the declared additive coefficient.
No model-test scores or image-review annotations are read.
"""
import argparse
import ast
import copy
import gc
import hashlib
import json
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
import traceback

from .cli import write_json, snapshot_code
from .native import additive_hook, build_generator, transform_entry
from .backend import sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--details',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--detail-coefficient',type=float,default=.1)
    a=p.parse_args()
    gpu=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
        capture_output=True,text=True,check=True,timeout=15)
    if any(x.strip().isdigit() for x in gpu.stdout.splitlines()):
        raise RuntimeError('GPU occupied; no other process stopped')
    a.output.mkdir(parents=True,exist_ok=False)
    snapshot_code(a.output)
    state=dict(state='starting',started_unix=time.time(),production_generation_started=False,
        victim_training_started=False,development_only=True)
    write_json(a.output/'status.json',state)
    try:
        import numpy as np
        import torch
        entry=a.repo/'wbd_data/nonboat20/run_decoder_200.py'
        tree=transform_entry(entry.read_text(),a.repo,a.output,1)
        prefix=[]
        for node in tree.body:
            if isinstance(node,ast.For) and ast.unparse(node.target)=='(source_index, sample)':
                break
            if not getattr(node,'_automatic_inserted',False):
                prefix.append(node)
        tree=ast.fix_missing_locations(ast.Module(body=prefix,type_ignores=[]))
        namespace=dict(__name__='_automatic_detail_GPU_development',__file__=str(entry))
        previous_argv=sys.argv;sys.argv=[str(entry)]
        try:
            exec(compile(tree,str(entry),'exec'),namespace)
        finally:
            sys.argv=previous_argv
        author,options=namespace['author'],namespace['options']
        details=json.loads((a.details/'details.json').read_text())
        sample=next(x for x in details if x['analysis']['detail_text'])
        index=sample['source_index'];row=sample['source']
        options.image_path=row['path'];options.clean_class=row['category']
        options.clean_text_from_data=author.convert_to_lowercase(
            'a photo of '+row['category'].strip()+' '+row['caption'],row['category'])
        original=author.generate_poisoned_text_of_beam_search
        adapted,adapted_source=build_generator(author)
        (a.output/'executed_generator.py').write_text(adapted_source)
        def rng():
            return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())
        def restore(s):
            random.setstate(s['python']);np.random.set_state(s['numpy'])
            torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state_all(s['cuda'])
        def digest(s):
            h=hashlib.sha256(pickle.dumps((s['python'],s['numpy']),protocol=4))
            for tensor in [s['cpu'],*s['cuda']]:
                h.update(tensor.cpu().numpy().tobytes())
            return h.hexdigest()
        initial=rng()
        cases={}
        manifest=dict(source_index=index,selection_rule='First automatic source with nonempty D, no manual image selection',
            D=sample['analysis']['details'],detail_coefficient=a.detail_coefficient,
            once_native_seed42=True,branch_RNG_restore='Development comparison only; not production per-source reseeding',
            production_rule='Original order200 sources/400 outputs, seed42 once at startup',
            details_sha256=sha(a.details/'details.json'),checkpoint_sha256=sha(options.checkpoints_path),
            annotation_inputs=[],llm_or_vlm_used=False,test_results_read=False)
        write_json(a.output/'manifest.json',manifest)
        for name,coefficient in [('native',None),('zero',0.),('detail',a.detail_coefficient)]:
            restore(initial)
            case_options=copy.deepcopy(options);author.options=case_options
            # The transformed native function captured the options object when
            # compiled. Preserve that same object for its module-global reads.
            adapted.__globals__['options']=case_options
            original.__globals__['options']=case_options
            state.update(state='running',case=name)
            write_json(a.output/'status.json',state)
            started=time.time();trace=[]
            if coefficient is None:
                raw=original(case_options,logger_=author.logger)
            else:
                hook,trace=additive_hook(author,sample['analysis']['detail_text'],row['category'],coefficient)
                raw=adapted(case_options,logger_=author.logger,condition_hook=hook)
            texts=[author.process_number_space(x[1]) for x in raw]
            selected=texts if len(texts)<=2 else author.select_diverse_texts(texts,author.stop_words,2,
                author.process_number_space(case_options.clean_text_from_data))
            assert len(selected)==2
            result=dict(raw=[[float(x[0]),x[1]] for x in raw],selected=selected,
                rng_after=digest(rng()),seconds=time.time()-started,trace=trace)
            write_json(a.output/(name+'.json'),result);cases[name]=result
            del raw
            if coefficient is not None:
                del hook
            gc.collect();torch.cuda.empty_cache()
        assert cases['native']['raw']==cases['zero']['raw'],'Zero-hook full accepted candidates differ'
        assert cases['native']['selected']==cases['zero']['selected'],'Zero-hook final outputs differ'
        assert cases['native']['rng_after']==cases['zero']['rng_after'],'Zero-hook ending RNG differs'
        assert any(x['active'] for x in cases['detail']['trace']),'Nonzero detail condition never activated'
        write_json(a.output/'results.json',dict(state='GPU_interface_verified',source_index=index,
            zero_full_candidates_equal=True,zero_final2_equal=True,zero_ending_rng_equal=True,
            nonzero_detail_condition_exercised=True,nonzero_selected=cases['detail']['selected'],
            production200_generation_not_complete=True,annotation_inputs=[],llm_or_vlm_used=False,
            no_claim_of_caption_factual_truth=True,
            total_seconds=sum(x['seconds'] for x in cases.values())))
        state.update(state='completed',ended_unix=time.time())
    except BaseException as exc:
        state.update(state='failed',error=repr(exc),ended_unix=time.time())
        (a.output/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write_json(a.output/'status.json',state)


if __name__=='__main__':
    main()
