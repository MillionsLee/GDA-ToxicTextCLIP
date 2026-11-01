"""Freeze the single authorized automatic run and its reused references."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0,str(Path(__file__).parent/'code_auto'))
from automatic_detail_method.backend import sha
from automatic_detail_method.cli import read_csv,write_json
from automatic_detail_method.core import Config


def main():
    repo=Path('/home/million/ToxicTextCLIP_W-BD')
    root=repo/'wbd_data/iteration_20261027'
    previous=repo/'wbd_data/iteration_20261025'
    reference=previous/'reference_01'
    extraction=root/'automatic_extract_01'
    gpu=root/'automatic_gpu_check_01'
    module=Path(__import__('automatic_detail_method').__file__).parent
    decoder=repo/'Background_driven_poisoned_text_augmenter_and_decoder_model'
    victim=repo/'Background_aware_target_text_telector_and_victim_model'
    files=[Path(__file__),root/'resume_request_20261027.json',
        extraction/'manifest.json',extraction/'details.json',extraction/'status.json',
        gpu/'results.json',gpu/'manifest.json',gpu/'status.json',
        repo/'wbd_data/victim/run_victim.py',repo/'wbd_data/victim/attack_train_wbd_100k.csv',
        repo/'wbd_data/victim/attack_train_wbd_baseline_100k.csv',repo/'wbd_data/victim/clean_train_100k.csv',
        repo/'wbd_data/victim/val.csv',repo/'wbd_data/nonboat20/prepare_for_generate.csv',
        repo/'wbd_data/nonboat20/prepare_for_attack_texts.csv',repo/'wbd_data/nonboat20/class_embedding.pt',
        repo/'wbd_data/nonboat20/run_decoder_200.py',repo/'models/checkpoints/epoch_14.pth',
        Path.home()/'.cache/clip/ViT-B-32.pt',
        repo/'wbd_data/eval/eval_wbd_epoch4.py',repo/'wbd_data/eval/eval_ca_epoch4.py',
        reference/'results.json',reference/'original/manifest.json',reference/'baseline/manifest.json',
        reference/'original/initial_state.pt',reference/'original/epochs.jsonl',reference/'baseline/epochs.jsonl',
        reference/'evaluation/original/hitk/query_gallery_audit.json',
        previous/'reference_gallery_contents_01/audit.json',previous/'clean_reference_01/results.json',
        previous/'historical_replay_02/results.json',
        decoder/'experiment/attack3/7_enhance_generation_oridata_prefix.py']
    files+=list(module.glob('*.py'))
    files+=[previous/'code/iteration_20261025'/n for n in ['native_train.py','evaluate_exact.py','gallery_content_audit.py','image_evidence.py']]
    files+=list((victim/'src').glob('*.py'))+list((victim/'pkgs/openai').glob('*.py'))
    for directory in ['models','clip','utils','src']:
        files+=list((decoder/directory).glob('*.py'))
    files+=[Path(x['path']) for x in read_csv(repo/'wbd_data/nonboat20/prepare_for_generate.csv')]
    unique=list(dict.fromkeys(files))
    protected={str(p):sha(p) for p in unique}
    assert protected[str(repo/'wbd_data/victim/attack_train_wbd_100k.csv')]=='007723441ebe295c382b66057b533b0823b49f35b3058a0f704032c5e59283b8'
    assert protected[str(repo/'wbd_data/victim/attack_train_wbd_baseline_100k.csv')]=='47b8f406a8a93f13a235002e73316d4f439845b1fba6e4f0163c6d1df99cd3fb'
    ref=json.loads((reference/'original/manifest.json').read_text())
    assert ref['batch_size']==32 and ref['nominal_schedule_epochs']==10 and ref['executed_epochs']==4
    assert ref['lr']==5e-5 and ref['warmup']==10000
    assert ref['cudnn_benchmark'] and not ref['cudnn_deterministic'] and not ref['explicit_seed_set']
    assert json.loads((extraction/'manifest.json').read_text())['config']==asdict(Config())
    assert json.loads((gpu/'status.json').read_text())['state']=='completed'
    plan=dict(frozen_at=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
        resume_request=json.loads((root/'resume_request_20261027.json').read_text()),
        config=asdict(Config()),automatic_extraction=str(extraction),GPU_interface_check=str(gpu),
        protected_sha256=protected,initial_state_sha256=sha(reference/'original/initial_state.pt'),
        baseline_and_original_reused=True,only_new_model_trained=True,
        generation=dict(sources=200,outputs=400,seed42='once at start, never per-source',
            native_source_model_reconstruction=True,detail_mode='loo',detail_coefficient=.1,
            native_beam_and_Jaccard_unchanged=True),
        selector=dict(all100_anchors_decided=True,max1_replacement_per_class=False,
            candidate_pool='Original200 captions plus new400 selected outputs; source-key dedup and128char limit; original100 anchors also compete',
            grounding_gate=.15,ranking='Masked-background class cosine only'),
        training=dict(RN50_pretrained=False,batch_size=32,workers=2,lr=5e-5,warmup=10000,
            nominal_schedule_epochs=10,executed_epochs=4,poison_weight=1.,
            seed=None,cudnn_benchmark=True,cudnn_deterministic=False,
            captured_reference_initial_weights_RNG_replayed=True,
            historical_initialization_unavailable=True,only_initial_RNG_saved_not_each_epoch=True),
        model_test_results_known_from_previous_experiments=True,new_method_test_seen=False,
        test_directed_tuning=False,annotation_inputs=[],llm_or_vlm_used=False,
        curriculum_not_run=True,automatic_next_variant=False)
    path=root/'automatic_plan_01.json'
    assert not path.exists()
    write_json(path,plan)
    print(json.dumps(dict(plan=str(path),protected_files=len(protected),plan_sha256=sha(path))),flush=True)


if __name__=='__main__':
    main()
