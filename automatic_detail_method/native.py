"""Add f_detail at the native text projection; preserve image/beam/Jaccard.

There is no anchor replacing fT, no review gate and no per-class1 restriction.
"""
import ast
import copy
import inspect
import json
import hashlib
import pickle
from pathlib import Path
import random
import subprocess
import sys
import time

from .core import coefficient_for_round, identities, words
from .backend import sha
from .cli import read_csv, write_csv, write_json, protect, snapshot_code


def additive_hook(author, detail_text, category, coefficient, mode='loo', support_texts=(), temperature=.07):
    if coefficient<0 or mode not in ('loo','image_projection') or temperature<=0:
        raise ValueError('Invalid additive detail condition')
    trace = []
    cached = {}
    blocked = identities(category)
    supports = sorted({s.strip() for s in support_texts if s.strip() and not blocked.intersection(words(s))})
    if blocked.intersection(words(detail_text)):
        raise ValueError('Class identity leaked into detail condition')

    def hook(context, current, text, image, visual, iteration, model):
        if coefficient==0:
            return context
        import torch
        if not cached:
            if mode=='loo':
                if not detail_text.strip():
                    trace.append(dict(iteration=iteration,active=False,reason='empty automatic D'))
                    cached['empty'] = True
                else:
                    tokens = author.clip.tokenize([detail_text]).to(current.device)
                    cached['detail'] = model.clip_model.encode_text(tokens).float()
            elif not supports:
                cached['empty'] = True
            else:
                tokens = author.clip.tokenize(supports).to(current.device)
                feature = model.clip_model.encode_text(tokens).float()
                image_z = torch.nn.functional.normalize(image.float().reshape(1,-1),dim=-1)
                text_z = torch.nn.functional.normalize(feature,dim=-1)
                weights = (image_z@text_z.T/temperature).softmax(dim=-1)
                cached['detail'] = weights@feature
                cached['top_support'] = [supports[i] for i in weights[0].topk(min(5,len(supports))).indices.tolist()]
        if cached.get('empty'):
            return context
        feature = cached['detail']
        mixed = context.clone()
        # The native fT projection and native lambdaAdd*imageCLS are intact.
        # Only the separate detail projection is added; image patches untouched.
        projected = feature@model.text_decoder.linear_feature2eot
        mixed[:,-1,:] += coefficient*projected
        trace.append(dict(iteration=iteration,active=True,mode=mode,detail_text=detail_text,
            coefficient=coefficient,detail_norm=float(feature.norm()),projected_norm=float(projected.norm()),
            top_support=cached.get('top_support'),original_text_condition_preserved=True,
            original_image_condition_preserved=True))
        return mixed
    return hook, trace


def build_generator(author):
    source = inspect.getsource(author.generate_poisoned_text_of_beam_search)
    original = ast.parse(source)
    tree = copy.deepcopy(original)
    function = tree.body[0]
    if not isinstance(function,ast.FunctionDef) or [x.arg for x in function.args.args]!=['options_','logger_']:
        raise ValueError('Unexpected native generation function')
    function.args.args.append(ast.arg(arg='condition_hook'))
    function.args.defaults.append(ast.Constant(None))
    matches = []
    class Add(ast.NodeTransformer):
        def visit_AugAssign(self,node):
            if ast.unparse(node.target)=='src_representations_batch[:, -1, :]' and ast.unparse(node.value)=='lambdaAdd * image_cls':
                matches.append(node)
                added = ast.parse('if condition_hook is not None:\n src_representations_batch = condition_hook(src_representations_batch, poisoned_text_feature, poisoned_text_, image_feature, lambdaAdd * image_cls, iterate_num, model)').body[0]
                added._automatic_inserted = True
                return [node,added]
            return node
    tree = Add().visit(tree)
    if len(matches)!=1:
        raise ValueError('Ambiguous native fusion site')
    restored = copy.deepcopy(tree)
    restored.body[0].args.args.pop(); restored.body[0].args.defaults.pop()
    class Remove(ast.NodeTransformer):
        def visit(self,node):
            if getattr(node,'_automatic_inserted',False):
                return None
            return super().visit(node)
    if ast.dump(Remove().visit(restored),include_attributes=False)!=ast.dump(original,include_attributes=False):
        raise ValueError('Native beam/model/image processing was changed')
    ast.fix_missing_locations(tree)
    namespace = dict(author.__dict__)
    exec(compile(tree,'<automatic_additive_detail>','exec'),namespace)
    return namespace['generate_poisoned_text_of_beam_search'],ast.unparse(tree)+'\n'


def transform_entry(source,repo,output,limit,generation_seed=42):
    original = ast.parse(source)
    tree = copy.deepcopy(original)
    paths = {'REPO':repo,'OUTPUT_DIR':output/'generated_texts','PROGRESS_CSV':output/'decoder_progress.csv'}
    counts = {}
    def inject(s):
        nodes = ast.parse(s).body
        for n in nodes:
            n._automatic_inserted = True
        return nodes
    body = []
    for node in tree.body:
        name = node.targets[0].id if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) else None
        if name in paths:
            node._original_value = copy.deepcopy(node.value)
            node.value = ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(paths[name]))],keywords=[])
            counts[name] = counts.get(name,0)+1
        if isinstance(node,ast.For) and ast.unparse(node.target)=='(source_index, sample)':
            if ast.unparse(node.iter)!='df.iterrows()':
                raise ValueError('Native source ordering changed')
            body += inject('_policy = _install_policy(author, options, df)')
            blocks = [n for n in node.body if isinstance(n,ast.Try)]
            if len(blocks)!=1:
                raise ValueError('Native error policy changed')
            block = blocks[0]
            new = []
            for line in block.body:
                assign = line.targets[0].id if isinstance(line,ast.Assign) and len(line.targets)==1 and isinstance(line.targets[0],ast.Name) else None
                if assign=='poisoned_text':
                    new += inject('_policy.begin(source_index, sample)')
                    counts['begin'] = counts.get('begin',0)+1
                if assign=='output_csv':
                    new += inject('_policy.record(source_index, sample, selected)')
                    counts['record'] = counts.get('record',0)+1
                new.append(line)
            block.body = new
            block.handlers[0].body += inject('raise')
            if limit:
                node.body = inject(f'if source_index >= {limit}:\n break')+node.body
        body.append(node)
    tree.body = body
    if any(counts.get(k)!=1 for k in [*paths,'begin','record']):
        raise ValueError('Native entry interface changed')
    seed_calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call)
        and ast.unparse(n.func)=='author.seed_everything']
    if len(seed_calls)!=1 or ast.unparse(seed_calls[0].args[0])!='42':
        raise ValueError('Generation seed call layout changed')
    seed_calls[0].args[0]._original_seed=42
    seed_calls[0].args[0].value=generation_seed
    class Restore(ast.NodeTransformer):
        def visit(self,node):
            if getattr(node,'_automatic_inserted',False):
                return None
            if hasattr(node,'_original_seed'):
                node.value = node._original_seed
            if hasattr(node,'_original_value'):
                node.value = node._original_value
            return super().visit(node)
    if ast.dump(Restore().visit(copy.deepcopy(tree)),include_attributes=False)!=ast.dump(original,include_attributes=False):
        raise ValueError('Entry changes extend beyond output routing/additive hook/fail policy')
    return ast.fix_missing_locations(tree)


def generate(args):
    details = json.loads((args.details/'details.json').read_text())
    extraction = json.loads((args.details/'manifest.json').read_text())
    assert extraction['llm_or_vlm_used'] is False and extraction['annotation_inputs']==[]
    source_path = args.repo/'wbd_data/nonboat20/prepare_for_generate.csv'
    sources = read_csv(source_path)
    assert len(sources)==200
    expected = args.limit or 200
    if args.limit is not None and not 1<=args.limit<200:
        raise ValueError('Development limit must be less than200')
    if len(details)<expected or (not args.limit and len(details)!=200):
        raise ValueError('Need automatic detail records for every source; no11-image whitelist')
    for i,row in enumerate(details[:expected]):
        assert row['source_index']==i and row['source']==sources[i]
        assert row['image_sha256']==sha(sources[i]['path'])
        assert row['analysis']['full_class_identity_masked']
    coefficient = coefficient_for_round(args.detail_coefficient,
        args.end_coefficient if args.end_coefficient is not None else args.detail_coefficient,
        args.round_index,args.rounds)
    gpu = subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    if any(x.strip().isdigit() for x in gpu.stdout.splitlines()):
        raise RuntimeError('GPU occupied; leave current jobs untouched')
    args.output.mkdir(parents=True,exist_ok=False)
    snapshot_code(args.output)
    entry = args.repo/'wbd_data/nonboat20/run_decoder_200.py'
    frozen = protect(args,[entry,source_path,args.details/'details.json',args.details/'manifest.json',
        args.repo/'models/checkpoints/epoch_14.pth',*[Path(x['path']) for x in sources[:expected]]])
    tree = transform_entry(entry.read_text(),args.repo,args.output,args.limit,args.generation_seed)
    (args.output/'executed_entry.py').write_text(ast.unparse(tree)+'\n')
    write_json(args.output/'manifest.json',dict(method='fT + native image condition + lambda2*f_detail',
        detail_mode=args.detail_mode,detail_coefficient=coefficient,detail_source='class-masked contextual LOO, computed automatically',
        expected_sources=expected,expected_outputs=2*expected,production_budget=[200,400],
        generation_seed=f'{args.generation_seed} once at startup; no per-source reset',
        image_lambda='native lambdaAdd schedule preserved; not silently replaced by paper0.3',
        offline_round_index=args.round_index,offline_rounds=args.rounds,
        unchanged_native_parts=['per-source model construction','image CLS','image patches','beam search','original Jaccard'],
        RNG_journal='Before/after source RNG states; no per-source seed reset; metadata I/O checked not to consume RNG',
        no_anchor_replacing_fT=True,llm_or_vlm_used=False,annotation_inputs=[],
        input_code_sha256=frozen,details_manifest_sha256=sha(args.details/'manifest.json'),
        development_only=args.limit is not None,victim_training_started=False))
    started = time.time()
    records, selected_rows = [],[]
    policy_holder = []
    class Policy:
        def __init__(self,author,options,frame):
            self.author,self.options = author,options
            if coefficient == 0:
                self.adapted = author.generate_poisoned_text_of_beam_search
                source = inspect.getsource(self.adapted)
            else:
                self.adapted,source = build_generator(author)
            (args.output/'executed_generator.py').write_text(source)
            self.original = author.generate_poisoned_text_of_beam_search
            author.generate_poisoned_text_of_beam_search = self.generate_one
            expected_options = dict(num_outputs=2,num_beams=60,get_sequences_num=60,num_beam_groups=5,
                diversity_penalty=.6,do_sample=True,top_k=60,top_p=.9,temperature=1.5,pad_token_id=400)
            assert all(getattr(options,k)==v for k,v in expected_options.items())
            assert len(frame)==200
        def rng(self):
            torch=self.author.torch
            return dict(python=random.getstate(),numpy=self.author.np.random.get_state(),cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
        @staticmethod
        def digest(state):
            value=hashlib.sha256(pickle.dumps((state['python'],state['numpy']),protocol=4))
            for tensor in [state['cpu'],*(state['cuda'] or [])]:
                value.update(tensor.cpu().numpy().tobytes())
            value.update(str(state['cuda'] is not None).encode())
            return value.hexdigest()
        def begin(self,index,sample):
            assert index==len(records)
            self.index = index
            self.detail = details[index]['analysis']
            self.trace = []
            self.before_rng=self.digest(self.rng())
            write_json(args.output/'status.json',dict(state='generating',completed=index,total=expected))
        def generate_one(self,options_,logger_):
            if coefficient == 0:
                self.trace = []
                return self.adapted(options_,logger_)
            support = [x['analysis']['detail_text'] for x in details if x['analysis']['detail_text']]
            hook,self.trace = additive_hook(self.author,self.detail['detail_text'],options_.clean_class,
                coefficient,args.detail_mode,support,args.projection_temperature)
            return self.adapted(options_,logger_,hook)
        def record(self,index,sample,selected):
            assert index==self.index and len(selected)==2
            after=self.rng()
            row = dict(source_index=index,source=dict(sample),details=self.detail['details'],
                selected=list(selected),condition_trace=self.trace,rng_before=self.before_rng,rng_after=self.digest(after))
            self.author.torch.save(after,args.output/f'after_source_{index:04d}_rng.pt')
            write_json(args.output/f'source_{index:04d}.json',row)
            records.append(row)
            selected_rows.extend(dict(path=sample['path'],caption=t,category=sample['category'],source_index=index) for t in selected)
            write_json(args.output/'status.json',dict(state='generating',completed=len(records),total=expected,
                active_detail_sources=sum(bool(x['details']) for x in records)))
            assert self.digest(self.rng())==row['rng_after'],'Metadata unexpectedly consumed generation RNG'
    def install(author,options,frame):
        obj = Policy(author,options,frame)
        policy_holder.append(obj)
        return obj
    namespace = dict(__name__='__automatic_detail_entry__',__file__=str(entry),_install_policy=install)
    before_argv = sys.argv
    sys.argv = [str(entry)]
    try:
        exec(compile(tree,str(entry),'exec'),namespace)
        assert len(records)==expected and len(selected_rows)==2*expected
        assert all(sha(p)==h for p,h in frozen.items())
        write_csv(args.output/'candidates.csv',selected_rows,['path','caption','category','source_index'])
        write_json(args.output/'status.json',dict(state='development_complete' if args.limit else 'generation_complete',
            completed=len(records),outputs=len(selected_rows),active_detail_sources=sum(bool(x['details']) for x in records),
            seconds=time.time()-started,llm_or_vlm_used=False,victim_training_started=False))
    except BaseException as exc:
        write_json(args.output/'status.json',dict(state='failed',completed=len(records),error=repr(exc)))
        raise
    finally:
        sys.argv = before_argv
    print(json.dumps(dict(output=str(args.output),sources=len(records),outputs=len(selected_rows))),flush=True)
