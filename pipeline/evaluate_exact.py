"""Run the user's unchanged metric calculations with isolated paths.

Only AST assignments for checkpoint/output and checkpoint label are replaced.
All query construction, preprocessing, model loading and metric math are retained.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def clean_retrieval_details(namespace, output):
    """Additional T2I recall on the same legacy nonboat query subset.

    A caption can have more than one annotated positive image. Queries whose
    positives are all absent from the existing gallery are reported separately.
    The original boat Hit@K result is left unchanged.
    """
    import torch
    caption_images = {}
    for ann in namespace["caption_data"]["annotations"]:
        caption_images.setdefault(ann["caption"].strip(), set()).add(ann["image_id"])
    filename_ids = {name: image_id for image_id, name in namespace["filename_by_image"].items()}
    gallery_ids = [filename_ids[path.name] for path in namespace["image_paths"]]
    gallery_set = set(gallery_ids)
    model, processor = namespace["model"], namespace["processor"]
    device = namespace["DEVICE"]
    queries = namespace["query_texts"]
    correct = {k: 0 for k in [1, 5, 10]}
    eligible = missing = 0
    with (output / "clean_query_details.jsonl").open("x") as handle, torch.inference_mode():
        for start in range(0, len(queries), 256):
            texts = queries[start:start+256]
            tokens = processor.process_text(texts)
            features = model.get_text_features(input_ids=tokens["input_ids"].to(device),
                                               attention_mask=tokens["attention_mask"].to(device))
            features = features / features.norm(dim=-1, keepdim=True)
            ranks = (features @ namespace["image_features_gpu"].T).topk(10, dim=1).indices.cpu().tolist()
            for text, indices in zip(texts, ranks):
                positives = caption_images[text.strip()] & gallery_set
                returned = [gallery_ids[index] for index in indices]
                hits = {k: bool(set(returned[:k]) & positives) for k in correct}
                if positives:
                    eligible += 1
                    for k in correct: correct[k] += hits[k]
                else:
                    missing += 1
                handle.write(json.dumps({"query": text.strip(), "positive_image_ids": sorted(positives),
                                         "top10_image_ids": returned, "eligible": bool(positives),
                                         "hits": hits}) + "\n")
    result = {"protocol": "clean T2I recall on legacy nonboat query subset; multiple positives allowed",
              "queries": len(queries), "eligible_queries": eligible, "missing_positive_queries": missing,
              "gallery_images": len(gallery_ids),
              "query_sha256": hashlib.sha256(json.dumps(queries).encode()).hexdigest(),
              "gallery_sha256": hashlib.sha256(json.dumps(gallery_ids).encode()).hexdigest(),
              **{f"Recall@{k}": correct[k]/eligible for k in correct}}
    with (output / "clean_recall.json").open("x") as handle:
        json.dump(result, handle, indent=2)


def rewrite(source, checkpoint, output, metric):
    tree = ast.parse(source)
    changes = []
    wanted = {"CHECKPOINT": str(checkpoint)}
    if metric == "hitk":
        wanted["OUT_DIR"] = str(output)
    else:
        wanted["OUT_FILE"] = str(output / "epoch4_ca.json")
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()),
                                      args=[ast.Constant(wanted[target.id])], keywords=[])
                changes.append(target.id)
    if sorted(changes) != sorted(wanted):
        raise ValueError(f"Unexpected source layout: {changes}")
    # Refuse the original overwrite mode, including the script's fixed JSON name.
    class SafeIO(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                    if node.args[1].value == "w":
                        node.args[1] = ast.Constant("x")
            return node
        def visit_Dict(self, node):
            self.generic_visit(node)
            for i, key in enumerate(node.keys):
                if isinstance(key, ast.Constant) and key.value == "checkpoint":
                    node.values[i] = ast.Constant(checkpoint.name)
            return node
    tree = SafeIO().visit(tree)
    return ast.fix_missing_locations(tree)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path.home() / "ToxicTextCLIP_W-BD")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metric", choices=["hitk", "ca"], required=True)
    p.add_argument("--visual-checkpoint", type=Path, help="Diagnostic only: take visual weights from this checkpoint")
    p.add_argument("--clean-queries", action="store_true", help="Diagnostic Hit@K without the trigger; never replace the primary triggered metric")
    p.add_argument("--retrieval-details", action="store_true")
    p.add_argument("--save-query-audit", action="store_true", help="Record query and gallery hashes without changing metric calculations")
    a = p.parse_args()
    script = a.repo / "wbd_data/eval" / ("eval_wbd_epoch4.py" if a.metric == "hitk" else "eval_ca_epoch4.py")
    source = script.read_text()
    a.output.mkdir(parents=True, exist_ok=False)
    with (a.output / "protocol.json").open("x") as f:
        json.dump({"source": str(script), "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                   "checkpoint": str(a.checkpoint), "visual_checkpoint": str(a.visual_checkpoint) if a.visual_checkpoint else None, "metric": a.metric,
                   "query_condition": "clean" if a.clean_queries else "triggered",
                   "changes": ["checkpoint path", "output path", "checkpoint label", "exclusive output"]}, f, indent=2)
    tree = rewrite(source, a.checkpoint, a.output, a.metric)
    if a.clean_queries:
        if a.metric != "hitk": raise ValueError("Clean queries apply only to the additional Hit@K diagnostic")
        found = False
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TRIGGER" for t in node.targets):
                node.value = ast.Constant("")
                found = True
        if not found: raise ValueError("Trigger assignment missing from source")
        tree = ast.fix_missing_locations(tree)
    import torch
    original_load = torch.load
    def diagnostic_load(path, *args, **kwargs):
        checkpoint = original_load(path, *args, **kwargs)
        if a.visual_checkpoint and Path(path).resolve() == a.checkpoint.resolve():
            visual = original_load(a.visual_checkpoint, map_location="cpu")
            def strip(state):
                return {k.removeprefix("module."): v for k, v in state.items()}
            state = strip(checkpoint["state_dict"])
            vstate = strip(visual["state_dict"])
            state.update({k: v for k, v in vstate.items() if k.startswith("visual.")})
            checkpoint["state_dict"] = state
            print("DIAGNOSTIC HYBRID: visual weights from", a.visual_checkpoint)
        return checkpoint
    torch.load = diagnostic_load
    try:
        namespace = {"__name__": "__main__", "__file__": str(script)}
        exec(compile(tree, str(script), "exec"), namespace)
        if a.save_query_audit:
            if a.metric != 'hitk':
                raise ValueError('Query audit applies only to Hit@K')
            queries = namespace['query_texts']
            gallery = [str(path) for path in namespace['image_paths']]
            with (a.output / 'query_gallery_audit.json').open('x') as handle:
                json.dump(dict(queries=len(queries), gallery_images=len(gallery),
                    query_sha256=hashlib.sha256(json.dumps(queries).encode()).hexdigest(),
                    gallery_sha256=hashlib.sha256(json.dumps(gallery).encode()).hexdigest()), handle, indent=2)
        if a.retrieval_details:
            if not a.clean_queries or a.metric != "hitk":
                raise ValueError("Additional clean recall requires clean Hit@K queries")
            clean_retrieval_details(namespace, a.output)
    finally:
        torch.load = original_load


if __name__ == "__main__":
    main()
