"""Post-evaluation CPU content audit, in addition to existing path-list hashes."""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path

from evaluate_exact import rewrite


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=a.repo/'wbd_data/eval/eval_wbd_epoch4.py'
    tree=rewrite(source.read_text(),a.reference/'original/checkpoints/epoch_4.pt',a.output,'hitk')
    prefix=[];found=False
    for node in tree.body:
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='load_model':
            found=True;break
        prefix.append(node)
    assert found
    prefix=ast.fix_missing_locations(ast.Module(body=prefix,type_ignores=[]))
    namespace={'__name__':'_data_audit_only','__file__':str(source)}
    exec(compile(prefix,str(source),'exec'),namespace)
    queries=namespace['query_texts'];gallery=namespace['image_paths']
    current=dict(queries=len(queries),gallery_images=len(gallery),
        query_sha256=hashlib.sha256(json.dumps(queries).encode()).hexdigest(),
        gallery_sha256=hashlib.sha256(json.dumps([str(x) for x in gallery]).encode()).hexdigest())
    for arm in ['original','baseline']:
        assert current==json.loads((a.reference/'evaluation'/arm/'hitk/query_gallery_audit.json').read_text())
    rows=[]
    for path in gallery:
        data=path.read_bytes()
        rows.append(dict(path=str(path),size=len(data),sha256=hashlib.sha256(data).hexdigest()))
    (a.output/'gallery_files.json').write_text(json.dumps(rows,indent=2)+'\n')
    annotations={str(namespace[name]):hashlib.sha256(namespace[name].read_bytes()).hexdigest()
                 for name in ['CAPTION_JSON','INSTANCE_JSON']}
    result=dict(**current,gallery_content_manifest_sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
        annotations_sha256=annotations,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        qualification='Post-evaluation contents; no historical gallery-content hash exists. Earlier gallery_sha256 means ordered paths only.',
        model_loaded=False)
    (a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
