"""Offline image-derived region candidates; captions are not ground truth."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='florence-community/Florence-2-base')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--limit',type=int,default=200)
    p.add_argument('--download-only',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    started=time.time();write(a.output/'status.json',dict(state='resolving_model',started_unix=started))
    try:
        from huggingface_hub import HfApi,snapshot_download
        meta=a.output/'model.json'
        if meta.exists():revision=json.loads(meta.read_text())['revision']
        else:
            revision=HfApi().model_info(a.model).sha
            write(meta,dict(repo_id=a.model,revision=revision,trust_remote_code=False,
                source='HF native Florence2 conversion referenced by Transformers official docs',
                warning='Generated regions/descriptions are candidate evidence, not verified facts'))
        local=snapshot_download(a.model,revision=revision,allow_patterns=['*.json','*.txt','*.model','*.safetensors'])
        if a.download_only:
            write(a.output/'status.json',dict(state='download_complete',local_snapshot=local));return
        import torch
        from transformers import AutoProcessor,Florence2ForConditionalGeneration
        from PIL import Image
        torch.set_num_threads(2)
        if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        dtype=torch.float32 if a.device=='cpu' else torch.float16
        model=Florence2ForConditionalGeneration.from_pretrained(local,torch_dtype=dtype,local_files_only=True).to(a.device).eval()
        processor=AutoProcessor.from_pretrained(local,local_files_only=True,trust_remote_code=False)
        source=a.repo/'wbd_data/nonboat20/prepare_for_generate.csv'
        with source.open() as f:records=list(csv.DictReader(f))
        assert len(records)==200
        write(a.output/'manifest.json',dict(model=a.model,revision=revision,device=a.device,
            torch=torch.__version__,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            inference=dict(num_beams=3,do_sample=False,max_new_tokens=256),
            evidence_policy='Raw candidates only; independent regional grounding and class preservation checks required'))
        for index,row in enumerate(records[:a.limit]):
            out=a.output/f'source_{index:04d}.json'
            if out.exists():continue
            with Image.open(row['path']) as im:image=im.convert('RGB')
            tasks={}
            for task in ['<DENSE_REGION_CAPTION>','<DETAILED_CAPTION>']:
                inputs=processor(text=task,images=image,return_tensors='pt')
                inputs={k:v.to(a.device,dtype=dtype if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
                with torch.inference_mode():ids=model.generate(**inputs,max_new_tokens=256,num_beams=3,do_sample=False)
                raw=processor.batch_decode(ids,skip_special_tokens=False)[0]
                tasks[task]=dict(raw=raw,parsed=processor.post_process_generation(raw,task=task,image_size=image.size))
            dense=tasks['<DENSE_REGION_CAPTION>']['parsed'].get('<DENSE_REGION_CAPTION>',{})
            regions=[]
            for number,(box,phrase) in enumerate(zip(dense.get('bboxes',[]),dense.get('labels',[]))):
                regions.append(dict(detail_id=f'{index:04d}_{number:02d}',phrase=phrase,bbox=box,
                    verified=False,grounding_score=None))
            write(out,dict(source_index=index,source=row,image_size=image.size,regions=regions,tasks=tasks))
            write(a.output/'status.json',dict(state='extracting',completed=index+1,total=a.limit,
                elapsed_seconds=time.time()-started,device=a.device))
            print(json.dumps(dict(completed=index+1,total=a.limit,regions=len(regions))),flush=True)
        write(a.output/'status.json',dict(state='extraction_complete',completed=a.limit,
            seconds=time.time()-started,verified_regions=False))
    except BaseException as exc:
        write(a.output/'status.json',dict(state='failed',error=repr(exc),seconds=time.time()-started));raise


if __name__=='__main__':main()
