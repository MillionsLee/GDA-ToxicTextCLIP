"""Existing CLIP only; no captioner, VLM, LLM, or annotation dependency."""
import hashlib
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


class ClipBackend:
    def __init__(self, repo, device='cpu', batch_size=32):
        import torch
        self.torch, self.device, self.batch_size = torch, device, batch_size
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable')
        torch.set_num_threads(2)
        root = Path(repo)/'Background_driven_poisoned_text_augmenter_and_decoder_model'
        sys.path.insert(0,str(root))
        import clip
        self.clip = clip
        self.asset = Path.home()/'.cache/clip/ViT-B-32.pt'
        self.model, self.preprocess = clip.load(str(self.asset),device=device,jit=False)
        self.model.float().eval()
        self.text_cache, self.image_cache = {}, {}

    def encode_text(self, texts):
        missing = list(dict.fromkeys(s for s in texts if s not in self.text_cache))
        with self.torch.inference_mode():
            for start in range(0,len(missing),self.batch_size):
                batch = missing[start:start+self.batch_size]
                # Fail on context overflow; do not silently change a caption.
                tokens = self.clip.tokenize(batch).to(self.device)
                z = self.torch.nn.functional.normalize(self.model.encode_text(tokens).float(),dim=-1)
                self.text_cache.update((s,v.detach().cpu()) for s,v in zip(batch,z))
        return self.torch.stack([self.text_cache[s] for s in texts]).to(self.device)

    def encode_image(self, path):
        from PIL import Image
        key = str(path)
        if key not in self.image_cache:
            with Image.open(path) as image:
                pixels = self.preprocess(image.convert('RGB')).unsqueeze(0).to(self.device)
            with self.torch.inference_mode():
                z = self.model.encode_image(pixels)
                if isinstance(z,tuple):
                    z = z[0]
                self.image_cache[key] = self.torch.nn.functional.normalize(z.float(),dim=-1)[0].detach().cpu()
        return self.image_cache[key].to(self.device)

    def class_centers(self, path):
        obj = self.torch.load(path,map_location='cpu',weights_only=False)
        labels = sorted(obj)
        values = self.torch.stack([obj[k].float().flatten() for k in labels]).to(self.device)
        return labels, self.torch.nn.functional.normalize(values,dim=-1)
