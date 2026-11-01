"""CPU integration check of the actual epoch14 model and additive interface.

This does not run beam search or certify GPU generation. It checks the native
model's real projection dimensions, unchanged patches and exact additive term.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import sys

from .backend import sha
from .cli import write_json, snapshot_code
from .native import additive_hook, build_generator, transform_entry


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--details',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    snapshot_code(a.output)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(2)
    entry = a.repo/'wbd_data/nonboat20/run_decoder_200.py'
    tree = transform_entry(entry.read_text(),a.repo,a.output,1)
    prefix = []
    for node in tree.body:
        if isinstance(node,ast.For) and ast.unparse(node.target)=='(source_index, sample)':
            break
        if not getattr(node,'_automatic_inserted',False):
            prefix.append(node)
    prefix = ast.fix_missing_locations(ast.Module(body=prefix,type_ignores=[]))
    namespace = dict(__name__='_automatic_native_CPU_check',__file__=str(entry))
    old_argv = sys.argv
    sys.argv = [str(entry)]
    try:
        exec(compile(prefix,str(entry),'exec'),namespace)
    finally:
        sys.argv = old_argv
    author,options = namespace['author'],namespace['options']
    options.device = 'cpu'
    author.options = options
    records = json.loads((a.details/'details.json').read_text())
    row = next(x for x in records if x['analysis']['detail_text'])
    options.clean_class = row['source']['category']
    options.clean_text_from_data = row['source']['caption']
    options.image_path = row['source']['path']
    model = author.models.load_my_model(options)
    state = torch.load(options.checkpoints_path,map_location='cpu',weights_only=False)
    model.load_state_dict(state['state_dict'])
    model.float().eval().to('cpu')
    from PIL import Image
    with torch.inference_mode():
        with Image.open(options.image_path) as raw:
            pixels = model.image_preprocess(raw.convert('RGB')).unsqueeze(0)
        image,representation = model.clip_model.encode_image(pixels,return_image_represent=True)
        patches = representation[:,1:,:].float()
        image_cls = representation[:,0,:].float()@model.text_decoder.projection_modify_visual_scale
        text = model.clip_model.encode_text(author.clip.tokenize([options.clean_text_from_data])).float()
        base = model.process_src_representations_batch(image_representations_batch=patches,text_feature=text)
        base[:,-1,:] += image_cls
        hook,trace = additive_hook(author,row['analysis']['detail_text'],options.clean_class,.1)
        changed = hook(base,text,options.clean_text_from_data,image,image_cls,1,model)
        detail = model.clip_model.encode_text(author.clip.tokenize([row['analysis']['detail_text']])).float()
        expected = base[:,-1,:]+.1*(detail@model.text_decoder.linear_feature2eot)
        assert torch.equal(changed[:,:-1,:],base[:,:-1,:])
        assert torch.equal(changed[:,-1,:],expected)
        zero,_ = additive_hook(author,row['analysis']['detail_text'],options.clean_class,0)
        assert zero(base,text,options.clean_text_from_data,image,image_cls,1,model) is base
        build_generator(author)  # Structural equivalence of all native statements.
    write_json(a.output/'results.json',dict(state='native_CPU_integration_passed',
        source_index=row['source_index'],automatic_D=row['analysis']['details'],
        context_shape=list(base.shape),text_feature_shape=list(text.shape),image_feature_shape=list(image.shape),
        projection_shape=list(model.text_decoder.linear_feature2eot.shape),
        patches_bitwise_unchanged=True,exact_additive_equation_verified=True,zero_hook_identity=True,
        native_AST_restore_verified=True,GPU_beam_search_tested=False,CLIP_cuda_used=False,
        checkpoint_sha256=sha(options.checkpoints_path),details_sha256=sha(a.details/'details.json'),
        annotation_inputs=[],trace=trace))
    print('Native CPU integration passed.',flush=True)


if __name__=='__main__':
    main()
